#!/usr/bin/env python3
"""
DanzarAI Vision Pipeline
========================

Real-time game vision processing with YOLO object detection, CLIP video understanding, 
template matching, and OCR. Designed for gaming commentary and HUD element detection.

Features:
- Fullscreen or window-specific capture at configurable FPS
- GPU-accelerated YOLO-Nano object detection
- CLIP-based semantic video understanding
- Template matching for static UI elements
- OCR on specific regions of interest
- Event debouncing and JSON output
- Async/threaded design for non-blocking operation
"""

import cv2
import numpy as np
import torch
import mss
import time
import json
import threading
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from ultralytics import YOLO
import pytesseract
from PIL import Image
import queue
import uuid

# Import CLIP vision enhancer
try:
    from services.clip_vision_enhancer import CLIPVisionEnhancer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPVisionEnhancer = None

@dataclass
class DetectionEvent:
    """Represents a detected object or event"""
    event_id: str
    timestamp: float
    object_type: str  # 'yolo', 'template', 'ocr', 'clip'
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CLIPVideoUpdate:
    """Represents a CLIP-based video understanding update"""
    timestamp: float
    frame_id: str
    clip_insights: List[Dict[str, Any]]
    semantic_understanding: Dict[str, Any]
    visual_descriptions: List[str]
    confidence_scores: Dict[str, float]
    game_context: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class VisionPipeline:
    """
    Real-time vision processing pipeline for game analysis with CLIP video understanding.
    
    Captures screenshots, runs object detection, CLIP semantic analysis, 
    template matching, and OCR in parallel with event debouncing and JSON output.
    """
    
    def __init__(self, config_path: str = "config/vision_config.yaml", 
                 event_callback: Optional[Callable[[DetectionEvent], None]] = None,
                 clip_callback: Optional[Callable[[CLIPVideoUpdate], None]] = None):
        """
        Initialize the vision pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            event_callback: Callback function for detection events
            clip_callback: Callback function for CLIP video updates
        """
        self.config_path = config_path
        self.event_callback = event_callback
        self.clip_callback = clip_callback
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Pipeline state
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        
        # Frame capture - initialize MSS properly
        try:
            self.sct = mss.mss()
            # Test capture to ensure it works
            test_monitor = self.sct.monitors[1]
            test_screenshot = self.sct.grab(test_monitor)
            self.logger.info("MSS screen capture initialized successfully")
        except Exception as e:
            self.logger.error(f"MSS initialization failed: {e}")
            self.sct = None
        
        # NDI capture support
        self.ndi_service = None
        self.use_ndi = self.config.get('capture', {}).get('use_ndi', False)
        self.ndi_source_name = self.config.get('capture', {}).get('ndi_source_name', None)
        
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.frame_timestamp = 0
        
        # Models and detection
        self.yolo_model = None
        self.templates = {}
        self.ocr_roi = None
        
        # CLIP video understanding
        self.clip_enhancer = None
        self.clip_enabled = False  # TEMPORARILY DISABLED TO RESTORE COMMENTARY
        self.clip_processing_fps = self.config.get('clip', {}).get('processing_fps', 1)  # Process every N frames
        self.clip_frame_counter = 0
        self.last_clip_update = 0
        self.clip_update_interval = 1.0 / self.clip_processing_fps if self.clip_processing_fps > 0 else 1.0
        
        # Event tracking and debouncing
        self.detection_history = {}
        self.debounce_timers = {}
        self.event_counter = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.processing_times = []
        
        self.logger.info("VisionPipeline initialized with CLIP video understanding")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("VisionPipeline")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'capture': {
                'fps': 10,
                'region': 'fullscreen',  # or 'window' with window_name
                'window_name': None,
                'monitor': 1,
                'use_ndi': False,
                'ndi_source_name': None
            },
            'yolo': {
                'model_path': 'models/yolo-nano.pt',
                'confidence_threshold': 0.5,
                'device': 'cuda:0',
                'classes': ['health_bar', 'minimap', 'boss', 'player', 'enemy']
            },
            'template_matching': {
                'enabled': True,
                'templates_dir': 'assets/templates/',
                'threshold': 0.8,
                'max_matches': 5
            },
            'ocr': {
                'enabled': True,
                'roi': [100, 100, 500, 200],  # x1, y1, x2, y2
                'tesseract_config': '--psm 6',
                'min_confidence': 0.6
            },
            'debouncing': {
                'enabled': True,
                'timeout_ms': 1000,
                'min_confidence_change': 0.1
            },
            'output': {
                'format': 'json',
                'include_metadata': True,
                'save_frames': False,
                'debug_mode': False
            },
            'clip': {
                'enabled': True,
                'processing_fps': 1
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the vision pipeline components"""
        try:
            self.logger.info("Initializing vision pipeline...")
            
            # Initialize NDI if enabled
            if self.use_ndi:
                if not await self._initialize_ndi():
                    self.logger.error("Failed to initialize NDI service")
                    return False
            
            # Initialize YOLO model
            if not await self._initialize_yolo():
                self.logger.error("Failed to initialize YOLO model")
                return False
            
            # Load template images
            if not await self._load_templates():
                self.logger.error("Failed to load templates")
                return False
            
            # Setup OCR ROI
            self._setup_ocr_roi()
            
            # Validate capture region
            if not self._validate_capture_region():
                self.logger.error("Invalid capture region")
                return False
            
            # Initialize CLIP video understanding
            if self.clip_enabled and not await self._initialize_clip():
                self.logger.error("Failed to initialize CLIP video understanding")
                return False
            
            self.logger.info("Vision pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _initialize_ndi(self) -> bool:
        """Initialize NDI service for OBS capture"""
        try:
            # Import NDI service
            from services.ndi_service import NDIService
            
            # Create a minimal app context for NDI service
            class MinimalAppContext:
                def __init__(self, ndi_source_name):
                    self.logger = self._setup_logging()
                    self.global_settings = {
                        'TARGET_NDI_SOURCE_NAME': ndi_source_name,
                        'NDI_CONNECTION_TIMEOUT_MS': 5000,
                        'NDI_RECEIVE_TIMEOUT_MS': 1000
                    }
                    # Add required attributes for NDI service
                    self.shutdown_event = threading.Event()
                    self.ndi_commentary_enabled = threading.Event()
                    self.frame_queue = queue.Queue(maxsize=10)
                
                def _setup_logging(self):
                    logging.basicConfig(level=logging.INFO)
                    return logging.getLogger("NDI-Vision")
            
            # Initialize NDI service
            app_context = MinimalAppContext(self.ndi_source_name)
            self.ndi_service = NDIService(app_context)
            
            # Initialize NDI connection
            if self.ndi_service.initialize_ndi():
                self.logger.info("NDI service initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize NDI connection")
                return False
                
        except Exception as e:
            self.logger.error(f"NDI initialization failed: {e}")
            return False
    
    async def _initialize_yolo(self) -> bool:
        """Initialize YOLO model on GPU"""
        try:
            model_path = self.config['yolo']['model_path']
            
            # Get optimal device from GPU memory manager or configuration
            device = self._get_optimal_device()
            
            # Check if CUDA is available
            if device.startswith('cuda') and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                device = 'cpu'
            
            # Load YOLO model
            if Path(model_path).exists():
                self.yolo_model = YOLO(model_path)
                self.logger.info(f"YOLO model loaded from {model_path}")
            else:
                # Use default YOLO model if custom model not found
                self.yolo_model = YOLO('yolov8n.pt')
                self.logger.info("Using default YOLOv8n model")
            
            # Move to device
            self.yolo_model.to(device)
            self.logger.info(f"YOLO model loaded on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"YOLO initialization failed: {e}")
            return False
    
    def _get_optimal_device(self) -> str:
        """Get the optimal device for vision processing using GPU memory manager."""
        try:
            # Check if we have access to app context with GPU memory manager
            if hasattr(self, 'app_context') and self.app_context and hasattr(self.app_context, 'gpu_memory_manager'):
                device, reason = self.app_context.gpu_memory_manager.get_best_vision_device()
                self.logger.info(f"[VisionPipeline] GPU Memory Manager selected device: {device} - {reason}")
                return device
            else:
                # Fallback to configuration
                device = self.config['yolo'].get('device', 'cuda:1')  # Default to cuda:1 to avoid main LLM
                self.logger.info(f"[VisionPipeline] Using configured device: {device}")
                return device
        except Exception as e:
            self.logger.warning(f"[VisionPipeline] Error getting optimal device: {e}")
            # Safe fallback to avoid main LLM
            return "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
    
    async def _load_templates(self) -> bool:
        """Load template images for matching"""
        try:
            if not self.config['template_matching']['enabled']:
                return True
            
            templates_dir = Path(self.config['template_matching']['templates_dir'])
            if not templates_dir.exists():
                self.logger.warning(f"Templates directory not found: {templates_dir}")
                return True
            
            # Load all template images
            for template_file in templates_dir.glob("*.png"):
                template_name = template_file.stem
                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    self.templates[template_name] = template_img
                    self.logger.debug(f"Loaded template: {template_name}")
            
            self.logger.info(f"Loaded {len(self.templates)} templates")
            return True
            
        except Exception as e:
            self.logger.error(f"Template loading failed: {e}")
            return False
    
    def _setup_ocr_roi(self):
        """Setup OCR region of interest"""
        if self.config['ocr']['enabled']:
            roi = self.config['ocr']['roi']
            self.ocr_roi = tuple(roi)
            self.logger.info(f"OCR ROI set to {self.ocr_roi}")
    
    def _validate_capture_region(self) -> bool:
        """Validate the capture region configuration"""
        try:
            region_config = self.config['capture']['region']
            
            if region_config == 'fullscreen':
                # Fullscreen capture is always valid
                return True
            elif region_config == 'window':
                window_name = self.config['capture']['window_name']
                if not window_name:
                    self.logger.error("Window name required for window capture")
                    return False
                # TODO: Validate window exists
                return True
            elif region_config == 'ndi':
                # NDI capture is valid if NDI is enabled
                if self.config['capture'].get('use_ndi', False):
                    return True
                else:
                    self.logger.error("NDI capture requires use_ndi to be enabled")
                    return False
            else:
                self.logger.error(f"Invalid capture region: {region_config}")
                return False
                
        except Exception as e:
            self.logger.error(f"Capture region validation failed: {e}")
            return False
    
    async def _initialize_clip(self) -> bool:
        """Initialize CLIP video understanding"""
        try:
            if not CLIP_AVAILABLE or not self.clip_enabled:
                self.logger.warning("CLIP video understanding not available or disabled")
                return True
            
            # Create a minimal app context for CLIP
            class MinimalAppContext:
                def __init__(self):
                    self.logger = self._setup_logging()
                    self.global_settings = {}
                
                def _setup_logging(self):
                    return logging.getLogger("CLIPVisionEnhancer")
            
            app_context = MinimalAppContext()
            self.clip_enhancer = CLIPVisionEnhancer(app_context)
            self.logger.info("CLIP video understanding initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"CLIP initialization failed: {e}")
            return False
    
    def start(self):
        """Start the vision pipeline"""
        if self.running:
            self.logger.warning("Vision pipeline already running")
            return
        
        self.running = True
        self.logger.info("Starting vision pipeline...")
        
        # AGENTIC MODE: Disable automatic capture loop - only capture when tools request it
        self.agentic_mode = True
        self.logger.info("🎯 AGENTIC MODE ENABLED: Automatic frame capture disabled - screenshots only when tools request them")
        
        # Start processing thread only (no capture loop)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Vision pipeline started (agentic mode - no automatic capture)")
    
    def stop(self):
        """Stop the vision pipeline"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping vision pipeline...")
        
        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Cleanup
        if self.sct:
            self.sct.close()
        self.logger.info("Vision pipeline stopped")
    
    def capture_frame_on_demand(self) -> Optional[np.ndarray]:
        """Capture a frame on demand (for agentic tools)"""
        try:
            self.logger.info("🎯 Capturing frame on demand (agentic mode)")
            frame = self._capture_frame()
            if frame is not None:
                # Process the frame immediately
                events = self._process_frame(frame, time.time())
                for event in events:
                    self._output_event(event)
                return frame
            else:
                self.logger.warning("Failed to capture frame on demand")
                return None
        except Exception as e:
            self.logger.error(f"Error capturing frame on demand: {e}")
            return None
    
    def _capture_loop(self):
        """Main capture loop - runs in separate thread"""
        fps = self.config['capture']['fps']
        frame_interval = 1.0 / fps
        
        self.logger.info(f"Starting capture loop at {fps} FPS")
        frame_count = 0
        
        while self.running:
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self._capture_frame()
                if frame is not None:
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        self.logger.info(f"Captured frame {frame_count} (shape: {frame.shape})")
                    
                    # Add to processing queue
                    try:
                        self.frame_queue.put_nowait((frame, start_time))
                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait((frame, start_time))
                        except queue.Empty:
                            pass
                else:
                    # If no frame captured, add a small delay to avoid busy waiting
                    time.sleep(0.01)
                    if frame_count % 100 == 0:  # Log every 100 failed attempts
                        self.logger.debug("No frame available from NDI, waiting...")
                
                # Maintain FPS
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Capture loop stopped after {frame_count} frames")
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from NDI or screen"""
        try:
            # Try NDI capture first if enabled
            if self.use_ndi and self.ndi_service and self.ndi_service.is_initialized:
                return self._capture_ndi_frame()
            
            # Fallback to screen capture
            if self.sct is None:
                self.logger.error("Screen capture not initialized")
                return None
                
            region_config = self.config['capture']['region']
            
            if region_config == 'fullscreen':
                # Capture fullscreen
                monitor = self.sct.monitors[self.config['capture']['monitor']]
                screenshot = self.sct.grab(monitor)
            elif region_config == 'window':
                # Capture specific window
                window_name = self.config['capture']['window_name']
                # TODO: Implement window-specific capture
                screenshot = self.sct.grab(self.sct.monitors[1])
            else:
                return None
            
            # Convert to numpy array
            frame = np.array(screenshot)
            # Convert from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
    
    def _capture_ndi_frame(self) -> Optional[np.ndarray]:
        """Capture frame from NDI source"""
        try:
            if not self.ndi_service or not self.ndi_service.is_initialized:
                return None
            
            # Get the latest frame from NDI service
            frame = self.ndi_service.last_captured_frame
            if frame is not None:
                # Convert to BGR if needed
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame
            
            # If no frame available, try to get from frame queue
            if hasattr(self.ndi_service.app_context, 'frame_queue'):
                try:
                    frame = self.ndi_service.app_context.frame_queue.get_nowait()
                    if frame is not None:
                        # Convert to BGR if needed
                        if len(frame.shape) == 3 and frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        return frame
                except queue.Empty:
                    pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"NDI frame capture failed: {e}")
            return None
    
    def _processing_loop(self):
        """Main processing loop - runs in separate thread"""
        self.logger.info("Starting processing loop")
        processed_count = 0
        
        while self.running:
            try:
                # Get frame from queue
                try:
                    frame, timestamp = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame
                start_time = time.time()
                events = self._process_frame(frame, timestamp)
                processing_time = time.time() - start_time
                
                processed_count += 1
                if processed_count % 30 == 0:  # Log every 30 frames
                    self.logger.info(f"Processed frame {processed_count} in {processing_time:.3f}s, found {len(events)} events")
                
                # Track performance
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # Output events
                for event in events:
                    self.logger.info(f"Detected event: {event.object_type} - {event.label} (conf: {event.confidence:.2f})")
                    self._output_event(event)
                
                # Update FPS counter
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                    self.logger.debug(f"FPS: {self.fps_counter}, Avg processing time: {avg_processing_time:.3f}s")
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Processing loop stopped after {processed_count} frames")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Process a single frame through all detection methods"""
        events = []
        
        # Run YOLO detection
        yolo_events = self._run_yolo_detection(frame, timestamp)
        events.extend(yolo_events)
        
        # Run template matching
        template_events = self._run_template_matching(frame, timestamp)
        events.extend(template_events)
        
        # Run OCR
        ocr_events = self._run_ocr(frame, timestamp)
        events.extend(ocr_events)
        
        # Run CLIP video understanding (at reduced frequency)
        if self.clip_enabled and self.clip_enhancer:
            clip_events = self._run_clip_analysis(frame, timestamp)
            events.extend(clip_events)
        else:
            # Ensure CLIP is completely disabled
            self.clip_enhancer = None
        
        return events
    
    def _run_yolo_detection(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Run YOLO object detection"""
        events = []
        
        try:
            if self.yolo_model is None:
                return events
            
            # Run inference
            results = self.yolo_model(frame, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Check confidence threshold
                        if confidence < self.config['yolo']['confidence_threshold']:
                            continue
                        
                        # Get class name
                        class_names = self.config['yolo']['classes']
                        if class_id < len(class_names):
                            label = class_names[class_id]
                        else:
                            label = f"class_{class_id}"
                        
                        # Create event
                        event = DetectionEvent(
                            event_id=str(uuid.uuid4()),
                            timestamp=timestamp,
                            object_type='yolo',
                            label=label,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            metadata={'class_id': class_id}
                        )
                        
                        # Check debouncing
                        if self._should_emit_event(event):
                            events.append(event)
                            self._update_detection_history(event)
        
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
        
        return events
    
    def _run_template_matching(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Run template matching"""
        events = []
        
        try:
            if not self.config['template_matching']['enabled'] or not self.templates:
                return events
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold = self.config['template_matching']['threshold']
            max_matches = self.config['template_matching']['max_matches']
            
            for template_name, template_img in self.templates.items():
                # Perform template matching
                result = cv2.matchTemplate(gray_frame, template_img, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                
                # Get top matches
                matches = []
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    w, h = template_img.shape[::-1]
                    confidence = result[y, x]
                    matches.append((x, y, w, h, confidence))
                
                # Sort by confidence and take top matches
                matches.sort(key=lambda x: x[4], reverse=True)
                matches = matches[:max_matches]
                
                for x, y, w, h, confidence in matches:
                    # Create event
                    event = DetectionEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        object_type='template',
                        label=template_name,
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h),
                        metadata={'template_size': (w, h)}
                    )
                    
                    # Check debouncing
                    if self._should_emit_event(event):
                        events.append(event)
                        self._update_detection_history(event)
        
        except Exception as e:
            self.logger.error(f"Template matching error: {e}")
        
        return events
    
    def _run_ocr(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Run OCR on region of interest with enhanced preprocessing"""
        events = []
        
        try:
            if not self.config['ocr']['enabled'] or self.ocr_roi is None:
                return events
            
            # Extract ROI
            x1, y1, x2, y2 = self.ocr_roi
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return events
            
            # Enhanced preprocessing for better OCR
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Basic thresholding
            _, thresh1 = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive thresholding
            thresh2 = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Method 3: Enhanced contrast and noise reduction
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray_roi)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Apply threshold to enhanced image
            _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 4: Monochrome conversion for better text recognition
            # Convert to black and white with high contrast
            _, monochrome = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
            
            # Try OCR with different preprocessing methods
            config = self.config['ocr']['tesseract_config']
            best_text = ""
            best_confidence = 0.0
            
            # Test different preprocessing methods
            preprocessing_methods = [
                ("basic", thresh1),
                ("adaptive", thresh2), 
                ("enhanced", thresh3),
                ("monochrome", monochrome)
            ]
            
            for method_name, processed_image in preprocessing_methods:
                try:
                    # Try different PSM modes for better text recognition
                    psm_modes = ['--psm 6', '--psm 7', '--psm 8', '--psm 13']
                    
                    for psm in psm_modes:
                        full_config = f"{config} {psm}"
                        text = pytesseract.image_to_string(processed_image, config=full_config).strip()
                        
                        if text and len(text) > 2:  # Only consider meaningful text
                            # Simple confidence estimation based on text quality
                            confidence = self._estimate_ocr_confidence(text)
                            
                            if confidence > best_confidence:
                                best_text = text
                                best_confidence = confidence
                                self.logger.debug(f"OCR method {method_name} with {psm} found: '{text}' (conf: {confidence:.2f})")
                
                except Exception as e:
                    self.logger.debug(f"OCR method {method_name} failed: {e}")
                    continue
            
            if best_text and best_confidence > 0.3:  # Lower threshold to catch more text
                # Create event
                event = DetectionEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    object_type='ocr',
                    label=best_text,
                    confidence=best_confidence,
                    bbox=self.ocr_roi,
                    metadata={
                        'text_length': len(best_text),
                        'preprocessing_method': 'enhanced',
                        'ocr_confidence': best_confidence
                    }
                )
                
                # Check debouncing
                if self._should_emit_event(event):
                    events.append(event)
                    self._update_detection_history(event)
                    self.logger.info(f"OCR detected: '{best_text}' (conf: {best_confidence:.2f})")
        
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
        
        return events
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text quality"""
        if not text:
            return 0.0
        
        # Remove common OCR artifacts
        clean_text = text.replace('|', 'I').replace('0', 'O').replace('1', 'l')
        
        # Check for common OCR issues
        issues = 0
        total_chars = len(clean_text)
        
        # Check for repeated characters (common OCR artifact)
        for i in range(1, len(clean_text)):
            if clean_text[i] == clean_text[i-1]:
                issues += 1
        
        # Check for mixed case (indicates good recognition)
        has_upper = any(c.isupper() for c in clean_text)
        has_lower = any(c.islower() for c in clean_text)
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in clean_text if c.isalpha())
        digit_chars = sum(1 for c in clean_text if c.isdigit())
        space_chars = sum(1 for c in clean_text if c.isspace())
        
        # Calculate confidence
        base_confidence = 0.5
        
        # Bonus for mixed case
        if has_upper and has_lower:
            base_confidence += 0.2
        
        # Bonus for reasonable character distribution
        if alpha_chars > 0 and alpha_chars / total_chars > 0.3:
            base_confidence += 0.1
        
        # Penalty for repeated characters
        if total_chars > 0:
            repeat_penalty = (issues / total_chars) * 0.3
            base_confidence -= repeat_penalty
        
        # Bonus for longer, meaningful text
        if len(clean_text) > 5:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _run_clip_analysis(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Run CLIP-based video understanding analysis"""
        # CLIP IS COMPLETELY DISABLED TO RESTORE COMMENTARY
        return []
    
    def _should_emit_event(self, event: DetectionEvent) -> bool:
        """Check if event should be emitted (debouncing logic)"""
        if not self.config['debouncing']['enabled']:
            return True
        
        # Create key for this detection
        key = f"{event.object_type}_{event.label}_{event.bbox}"
        
        # Check if we have a recent detection
        if key in self.detection_history:
            last_event = self.detection_history[key]
            time_diff = event.timestamp - last_event.timestamp
            timeout_ms = self.config['debouncing']['timeout_ms']
            
            # Check if enough time has passed
            if time_diff * 1000 < timeout_ms:
                # Check if confidence changed significantly
                confidence_change = abs(event.confidence - last_event.confidence)
                min_change = self.config['debouncing']['min_confidence_change']
                
                if confidence_change < min_change:
                    return False
        
        return True
    
    def _update_detection_history(self, event: DetectionEvent):
        """Update detection history for debouncing"""
        key = f"{event.object_type}_{event.label}_{event.bbox}"
        self.detection_history[key] = event
        
        # Clean up old entries
        current_time = time.time()
        keys_to_remove = []
        for k, v in self.detection_history.items():
            if current_time - v.timestamp > 5.0:  # Keep 5 seconds of history
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.detection_history[k]
    
    def _output_event(self, event: DetectionEvent):
        """Output detection event"""
        try:
            # Convert to JSON
            event_dict = asdict(event)
            
            # Call callback if provided
            if self.event_callback:
                self.event_callback(event)
            
            # Output to stdout if debug mode enabled (with safe fallback)
            try:
                debug_mode = self.config.get('output', {}).get('debug_mode', False)
                if debug_mode:
                    print(json.dumps(event_dict))
            except (KeyError, TypeError):
                # If output config is missing, just skip debug output
                pass
            
            # Save frame if enabled (with safe fallback)
            try:
                save_frames = self.config.get('output', {}).get('save_frames', False)
                if save_frames:
                    self._save_debug_frame(event)
            except (KeyError, TypeError):
                # If output config is missing, just skip frame saving
                pass
        
        except Exception as e:
            self.logger.error(f"Event output error: {e}")
    
    def _save_debug_frame(self, event: DetectionEvent):
        """Save debug frame with detection annotation"""
        try:
            if self.latest_frame is None:
                return
            
            # Create annotated frame
            frame = self.latest_frame.copy()
            x1, y1, x2, y2 = event.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if event.confidence > 0.8 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{event.label} ({event.confidence:.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save frame
            timestamp = int(event.timestamp * 1000)
            filename = f"debug_frames/frame_{timestamp}_{event.event_id[:8]}.jpg"
            Path("debug_frames").mkdir(exist_ok=True)
            cv2.imwrite(filename, frame)
        
        except Exception as e:
            self.logger.error(f"Debug frame save error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'running': self.running,
            'fps': self.fps_counter,
            'queue_size': self.frame_queue.qsize(),
            'processing_time_avg': np.mean(self.processing_times) if self.processing_times else 0,
            'detection_history_size': len(self.detection_history),
            'templates_loaded': len(self.templates),
            'yolo_model_loaded': self.yolo_model is not None
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example event callback
        def event_callback(event: DetectionEvent):
            print(f"Detection: {event.object_type} - {event.label} (conf: {event.confidence:.2f})")
        
        # Create pipeline
        pipeline = VisionPipeline(event_callback=event_callback)
        
        # Initialize
        if await pipeline.initialize():
            # Start pipeline
            pipeline.start()
            
            try:
                # Run for 30 seconds
                await asyncio.sleep(30)
            except KeyboardInterrupt:
                print("Stopping pipeline...")
            finally:
                pipeline.stop()
        else:
            print("Failed to initialize pipeline")
    
    # Run the example
    asyncio.run(main()) 