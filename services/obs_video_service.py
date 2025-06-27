#!/usr/bin/env python3
"""
OBS Video Service for DanzarAI - Real-time OBS capture with Qwen2.5-VL analysis
Integrates with OBS Studio for live video processing and gaming commentary
"""

import cv2
import numpy as np
import time
import tempfile
import os
import asyncio
import base64
import json
from typing import Optional, List, Tuple, Dict, Any, Callable
from pathlib import Path
import logging
from datetime import datetime
import requests
import aiohttp

try:
    import obsws_python as obs
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    logging.warning("obsws-python not available - install with: pip install obsws-python")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("mss library not available - screen capture disabled")

class OBSVideoService:
    """OBS integration service for real-time video analysis with Qwen2.5-VL"""
    
    def __init__(self, app_context):
        """Initialize OBS video service"""
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Video analysis settings
        video_config = self.config.get('VIDEO_ANALYSIS', {})
        self.analysis_enabled = video_config.get('enabled', True)
        self.analysis_interval = video_config.get('analysis_interval', 10.0)
        self.save_debug_frames = video_config.get('save_debug_frames', True)
        
        # Hybrid model configuration
        self.use_hybrid_models = video_config.get('use_hybrid_models', False)
        self.fast_model_endpoint = video_config.get('fast_model_endpoint', 'http://localhost:8083')
        self.detailed_model_endpoint = video_config.get('detailed_model_endpoint', 'http://localhost:8083')
        
        # Model selection
        self.use_qwen_vl = video_config.get('use_qwen_vl', True)  # Use Qwen2.5-VL model
        self.use_llamacpp_vl = video_config.get('use_llamacpp_vl', True)  # Use llama.cpp version
        
        # OBS connection settings
        obs_config = self.config.get('OBS_SETTINGS', {})
        self.obs_host = obs_config.get('host', 'localhost')
        self.obs_port = obs_config.get('port', 4455)
        self.obs_password = obs_config.get('password', '')
        
        # OBS source configuration - use configured source names
        self.obs_source_names = obs_config.get('source_names', [
            "NDI® Source",      # Primary NDI source (from our test)
            "Game Capture",     # Common game capture source
            "Display Capture",  # Display capture source
            "Window Capture",   # Window capture source
            "Screen Capture"    # Screen capture source
        ])
        self.active_obs_source = None
        
        # Capture settings
        capture_config = obs_config.get('capture_settings', {})
        self.capture_format = capture_config.get('format', 'jpg')
        self.capture_width = capture_config.get('width', 1920)
        self.capture_height = capture_config.get('height', 1080)
        self.capture_quality = capture_config.get('quality', 95)
        
        # Processing state
        self.is_processing = False
        self.obs_client = None
        self.current_scene = None
        self.analysis_callbacks = []
        
        # VL model services
        self.qwen_vl_service = None
        self.llamacpp_vl_service = None
        
        self.logger.info(f"[OBSVideoService] Initialized - Host: {self.obs_host}:{self.obs_port}")
        self.logger.info(f"[OBSVideoService] Hybrid models: {self.use_hybrid_models}")
        self.logger.info(f"[OBSVideoService] Capture: {self.capture_width}x{self.capture_height} {self.capture_format} Q{self.capture_quality}")
        if self.use_hybrid_models:
            self.logger.info(f"[OBSVideoService] Fast model: {self.fast_model_endpoint}")
            self.logger.info(f"[OBSVideoService] Detailed model: {self.detailed_model_endpoint}")
    
    async def initialize(self) -> bool:
        """Initialize OBS connection and video service"""
        try:
            if not OBS_AVAILABLE:
                self.logger.error("[OBSVideoService] obsws-python not available")
                return False
            
            # Connect to OBS
            self.logger.info(f"[OBSVideoService] Connecting to OBS at {self.obs_host}:{self.obs_port}")
            
            self.obs_client = obs.ReqClient(
                host=self.obs_host,
                port=self.obs_port,
                password=self.obs_password,
                timeout=3
            )
            
            # Test connection
            version_info = self.obs_client.get_version()
            self.logger.info(f"[OBSVideoService] Connected to OBS Studio {version_info.obs_version}")
            
            # Get current scene
            current_scene = self.obs_client.get_current_program_scene()
            self.current_scene = current_scene.current_program_scene_name
            self.logger.info(f"[OBSVideoService] Current scene: {self.current_scene}")
            
            # Find active OBS source
            self.active_obs_source = self.find_active_obs_source()
            if self.active_obs_source:
                self.logger.info(f"[OBSVideoService] Using source: {self.active_obs_source}")
            else:
                self.logger.warning("[OBSVideoService] No suitable video source found - will use scene capture")
                self.active_obs_source = self.current_scene
            
            # Initialize VL models if enabled
            if self.use_qwen_vl:
                try:
                    from services.qwen_vl_service import QwenVLService
                    self.qwen_vl_service = QwenVLService(self.app_context)
                    vl_success = await self.qwen_vl_service.initialize()
                    if vl_success:
                        self.logger.info("[OBSVideoService] Qwen2.5-VL model initialized successfully")
                    else:
                        self.logger.warning("[OBSVideoService] Failed to initialize Qwen2.5-VL model")
                except Exception as e:
                    self.logger.error(f"[OBSVideoService] Error initializing Qwen2.5-VL: {e}")
                    self.use_qwen_vl = False
            
            if self.use_llamacpp_vl:
                try:
                    from services.llamacpp_qwen_vl_service import LlamaCppQwenVLService
                    self.llamacpp_vl_service = LlamaCppQwenVLService(self.app_context)
                    llamacpp_success = await self.llamacpp_vl_service.initialize()
                    if llamacpp_success:
                        self.logger.info("[OBSVideoService] LlamaCpp Qwen2.5-VL model initialized successfully")
                    else:
                        self.logger.warning("[OBSVideoService] Failed to initialize LlamaCpp Qwen2.5-VL model")
                except Exception as e:
                    self.logger.error(f"[OBSVideoService] Error initializing LlamaCpp Qwen2.5-VL: {e}")
                    self.use_llamacpp_vl = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Failed to connect to OBS: {e}")
            self.logger.info("[OBSVideoService] Make sure OBS Studio is running with WebSocket server enabled")
            self.logger.info("[OBSVideoService] Go to Tools > WebSocket Server Settings in OBS")
            return False
    
    def capture_obs_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot from OBS using the active video source"""
        if not self.obs_client or not self.active_obs_source:
            return None
            
        try:
            # First try to capture from the scene directly (more reliable with NDI)
            try:
                current_scene = self.obs_client.get_current_program_scene()
                scene_name = current_scene.current_program_scene_name
                
                screenshot_request = self.obs_client.get_source_screenshot(
                    scene_name,
                    self.capture_format,
                    self.capture_width,
                    self.capture_height,
                    self.capture_quality
                )
                
                # Decode base64 image  
                image_data_str = screenshot_request.image_data
                if not image_data_str:
                    self.logger.warning("[OBSVideoService] Scene screenshot returned no data, trying source directly")
                    raise ValueError("No image data in scene screenshot")
                    
                if image_data_str.startswith('data:image/'):
                    # Remove data URL prefix
                    image_data_str = image_data_str.split(',')[1]
                
                image_data = base64.b64decode(image_data_str)
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.logger.debug(f"[OBSVideoService] Captured frame from scene '{scene_name}': {frame.shape}")
                    return frame
                else:
                    self.logger.warning("[OBSVideoService] Failed to decode scene screenshot")
                    raise ValueError("Failed to decode scene screenshot")
                    
            except Exception as scene_error:
                self.logger.warning(f"[OBSVideoService] Scene screenshot failed: {scene_error}")
                
                # Try source capture as fallback
                screenshot_request = self.obs_client.get_source_screenshot(
                    self.active_obs_source,
                    self.capture_format,
                    self.capture_width,
                    self.capture_height,
                    self.capture_quality
                )
                
                # Decode base64 image  
                image_data_str = screenshot_request.image_data
                if not image_data_str:
                    self.logger.error("[OBSVideoService] Source screenshot also returned no data")
                    return None
                    
                if image_data_str.startswith('data:image/'):
                    # Remove data URL prefix
                    image_data_str = image_data_str.split(',')[1]
                
                image_data = base64.b64decode(image_data_str)
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.logger.debug(f"[OBSVideoService] Captured frame from source '{self.active_obs_source}': {frame.shape}")
                    return frame
                else:
                    self.logger.warning("[OBSVideoService] Failed to decode source screenshot")
                    return None
                
        except Exception as e:
            self.logger.error(f"[OBSVideoService] OBS screenshot failed: {e}")
            
            # Try screen capture fallback
            return self.capture_screen_fallback()
    
    def capture_screen_fallback(self) -> Optional[np.ndarray]:
        """Fallback to screen capture if OBS not available"""
        if not MSS_AVAILABLE:
            return None
            
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                
                # Convert BGRA to BGR
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Resize for performance
                height, width = frame.shape[:2]
                new_width = int(width * 0.6)  # 60% size
                new_height = int(height * 0.6)
                frame = cv2.resize(frame, (new_width, new_height))
                
                return frame
                
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Screen capture fallback failed: {e}")
            return None
    
    def frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Convert frame to base64 for Qwen2.5-VL analysis"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode as JPEG
            success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                self.logger.error("[OBSVideoService] Failed to encode frame as JPEG")
                return None
                
            # Convert to base64
            image_bytes = buffer.tobytes()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Add data URL prefix
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error converting frame to base64: {e}")
            return None
    
    def save_debug_frame(self, frame: np.ndarray, prefix: str = "obs") -> Optional[str]:
        """Save frame for debugging purposes"""
        if not self.save_debug_frames:
            return None
            
        try:
            debug_dir = self.config.get('DEBUG_OUTPUT_PATH', './debug_frames')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(debug_dir, filename)
            
            cv2.imwrite(filepath, frame)
            self.logger.debug(f"[OBSVideoService] Saved debug frame to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error saving debug frame: {e}")
            return None

    async def clear_llama_context(self) -> bool:
        """Clear llama.cpp context to start fresh analysis"""
        try:
            # Reset server context by sending a simple query
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.detailed_model_endpoint}/v1/chat/completions",
                    json={
                        "messages": [{"role": "system", "content": "Reset context"}],
                        "max_tokens": 1,
                        "temperature": 0.0
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    await response.text()
            return True
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error clearing context: {e}")
            return False

    async def analyze_frame_with_omni(self, frame: np.ndarray, 
                                    prompt: str = "Analyze this gaming footage and provide commentary",
                                    use_fast_model: bool = False) -> str:
        """
        Analyze a frame using Qwen2.5-VL
        
        Args:
            frame: The frame to analyze
            prompt: The prompt to use for analysis
            use_fast_model: Whether to use the fast model endpoint
            
        Returns:
            Analysis text
        """
        try:
            # First try using Qwen2.5-VL HuggingFace model if available
            if self.use_qwen_vl and self.qwen_vl_service and self.qwen_vl_service.is_available():
                self.logger.info("[OBSVideoService] Using Qwen2.5-VL HuggingFace model for analysis")
                return await self.qwen_vl_service.analyze_frame(frame, prompt)
            
            # Then try using LlamaCpp Qwen2.5-VL if available
            if self.use_llamacpp_vl and self.llamacpp_vl_service and self.llamacpp_vl_service.is_available():
                self.logger.info("[OBSVideoService] Using LlamaCpp Qwen2.5-VL model for analysis")
                return await self.llamacpp_vl_service.analyze_frame(frame, prompt)
            
            # Fallback to llama.cpp server API
            self.logger.info(f"[OBSVideoService] Using llama.cpp server API for analysis (fast_model: {use_fast_model})")
            
            # Save frame for debugging
            debug_path = self.save_debug_frame(frame, "analysis")
            
            # Convert frame to base64
            base64_image = self.frame_to_base64(frame)
            if not base64_image:
                return "Error: Failed to convert frame to base64"
                
            # Select endpoint based on analysis type
            endpoint = self.fast_model_endpoint if use_fast_model else self.detailed_model_endpoint
            endpoint = f"{endpoint}/v1/chat/completions"
            
            # Create payload with image
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that analyzes gaming footage."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_image}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": 256 if use_fast_model else 512,
                "temperature": 0.7,
                "stream": False
            }
            
            # Send request with timeout
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30 if use_fast_model else 60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        self.logger.info(f"[OBSVideoService] Analysis generated: {content[:50]}...")
                        return content
                    else:
                        error_text = await response.text()
                        self.logger.error(f"[OBSVideoService] API error: {response.status}, {error_text}")
                        return f"Error: API returned {response.status}"
                        
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error analyzing frame: {str(e)}"

    async def get_current_analysis(self, prompt: str = "What's happening in this game right now?", 
                                 use_fast_model: bool = False) -> str:
        """Get analysis of the current frame"""
        frame = self.capture_obs_screenshot()
        if frame is None:
            return "Error: Failed to capture screenshot from OBS"
            
        return await self.analyze_frame_with_omni(frame, prompt, use_fast_model)
    
    async def get_fast_analysis(self, prompt: str = "Game?") -> str:
        """Get quick analysis of the current frame"""
        return await self.get_current_analysis(prompt, use_fast_model=True)
    
    async def get_detailed_analysis(self, prompt: str = "What's happening in this game right now?") -> str:
        """Get detailed analysis of the current frame"""
        return await self.get_current_analysis(prompt, use_fast_model=False)
    
    async def analyze_current_frame(self, prompt: str = "What's happening in this game right now?") -> Dict[str, Any]:
        """Analyze current frame and return detailed result"""
        try:
            start_time = time.time()
            
            # Capture frame
            frame = self.capture_obs_screenshot()
            if frame is None:
                return {"success": False, "error": "Failed to capture screenshot from OBS"}
            
            # Save debug frame
            debug_path = self.save_debug_frame(frame, "analysis")
            
            # Get analysis
            analysis = await self.analyze_frame_with_omni(frame, prompt, use_fast_model=False)
            
            # Calculate time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "analysis": analysis,
                "frame_path": debug_path,
                "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_obs_info(self) -> Dict[str, Any]:
        """Get OBS connection and source information"""
        try:
            if not self.obs_client:
                return {"connected": False}
            
            # Get OBS version
            version_info = self.obs_client.get_version()
            
            # Get current scene
            current_scene = self.obs_client.get_current_program_scene()
            scene_name = current_scene.current_program_scene_name
            
            # Get sources in current scene
            sources = self.obs_client.get_scene_item_list(scene_name)
            source_names = [item['sourceName'] for item in sources.scene_items]
            
            return {
                "connected": True,
                "obs_version": version_info.obs_version,
                "websocket_version": version_info.obs_web_socket_version,
                "current_scene": scene_name,
                "active_source": self.active_obs_source,
                "available_sources": source_names
            }
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error getting OBS info: {e}")
            return {"connected": False, "error": str(e)}
    
    def stop_analysis(self):
        """Stop ongoing analysis"""
        self.is_processing = False
    
    def disconnect(self):
        """Disconnect from OBS"""
        try:
            if self.obs_client:
                self.obs_client.disconnect()
                self.obs_client = None
                
            # Clean up VL services
            if self.qwen_vl_service:
                asyncio.create_task(self.qwen_vl_service.cleanup())
                
            if self.llamacpp_vl_service:
                asyncio.create_task(self.llamacpp_vl_service.cleanup())
                
            self.logger.info("[OBSVideoService] Disconnected from OBS")
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error disconnecting: {e}")
    
    def find_active_obs_source(self) -> Optional[str]:
        """Find the first available video source from the configured list"""
        if not self.obs_client:
            return None
            
        try:
            # Get current scene
            current_scene = self.obs_client.get_current_program_scene()
            scene_name = current_scene.current_program_scene_name
            
            # Get sources in current scene
            sources = self.obs_client.get_scene_item_list(scene_name)
            scene_sources = [item['sourceName'] for item in sources.scene_items]
            
            # Find first matching source
            for source_name in self.obs_source_names:
                if source_name in scene_sources:
                    self.logger.info(f"[OBSVideoService] Found source: {source_name}")
                    return source_name
            
            # If no match, return the first source
            if scene_sources:
                self.logger.info(f"[OBSVideoService] Using first available source: {scene_sources[0]}")
                return scene_sources[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"[OBSVideoService] Error finding active source: {e}")
            return None
