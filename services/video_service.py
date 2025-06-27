#!/usr/bin/env python3
"""
Video Service for DanzarAI - Real-time video processing with Qwen2.5-Omni
Handles live screen capture, video analysis, and gaming commentary
"""

import cv2
import numpy as np
import time
import tempfile
import os
import asyncio
import base64
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("mss library not available - screen capture disabled")

class VideoService:
    """Real-time video processing service using Qwen2.5-Omni capabilities"""
    
    def __init__(self, app_context):
        """Initialize video service with app context"""
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Video processing settings
        self.is_processing = False
        self.frame_buffer = []
        self.max_buffer_size = 60  # 2 seconds at 30fps
        self.chunk_duration = 2.0  # Process 2-second chunks as per Qwen2.5-Omni specs
        self.target_fps = 15  # Reduced FPS for better performance
        
        # Video analysis settings
        self.analysis_enabled = self.config.get('VIDEO_ANALYSIS_ENABLED', False)
        self.gaming_mode = self.config.get('VIDEO_GAMING_MODE', True)
        self.save_debug_frames = self.config.get('VIDEO_SAVE_DEBUG_FRAMES', True)
        
        # Screen capture settings
        self.capture_region = self.config.get('VIDEO_CAPTURE_REGION', None)  # (x, y, width, height)
        self.capture_quality = self.config.get('VIDEO_CAPTURE_QUALITY', 0.8)
        
        self.logger.info(f"[VideoService] Initialized - Analysis: {self.analysis_enabled}, Gaming Mode: {self.gaming_mode}")
    
    async def initialize(self) -> bool:
        """Initialize video service components"""
        try:
            if not MSS_AVAILABLE:
                self.logger.error("[VideoService] mss library not available - screen capture disabled")
                return False
                
            # Test screen capture
            test_frame = self.capture_screen()
            if test_frame is not None:
                self.logger.info(f"[VideoService] Screen capture working: {test_frame.shape}")
                
                # Save test frame for debugging
                if self.save_debug_frames:
                    debug_frames_dir = getattr(self.app_context, 'debug_frames_dir', './debug_frames')
                    os.makedirs(debug_frames_dir, exist_ok=True)
                    debug_path = os.path.join(debug_frames_dir, "video_test_capture.jpg")
                    cv2.imwrite(debug_path, test_frame)
                    self.logger.info(f"[VideoService] Test frame saved: {debug_path}")
                    
                return True
            else:
                self.logger.error("[VideoService] Screen capture test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"[VideoService] Initialization failed: {e}")
            return False
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Capture screen region for gaming analysis
        
        Args:
            region: (x, y, width, height) or None for configured region/full screen
            
        Returns:
            Captured frame as numpy array
        """
        if not MSS_AVAILABLE:
            return None
            
        try:
            with mss.mss() as sct:
                # Use provided region, configured region, or full screen
                capture_region = region or self.capture_region
                
                if capture_region:
                    monitor = {
                        "top": capture_region[1], 
                        "left": capture_region[0], 
                        "width": capture_region[2], 
                        "height": capture_region[3]
                    }
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                    
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                
                # Convert BGRA to BGR for OpenCV
                if frame.shape[2] == 4:  # BGRA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.shape[2] == 3:  # Already BGR
                    pass
                else:
                    self.logger.warning(f"[VideoService] Unexpected frame format: {frame.shape}")
                
                # Resize if needed for performance
                if self.capture_quality < 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * self.capture_quality)
                    new_height = int(height * self.capture_quality)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                return frame
                
        except Exception as e:
            self.logger.error(f"[VideoService] Screen capture failed: {e}")
            return None
    
    def frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """
        Convert frame to base64 for LLM processing
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Base64 encoded image or None if failed
        """
        try:
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None
            
            # Convert to base64
            base64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
            return base64_image
            
        except Exception as e:
            self.logger.error(f"[VideoService] Frame to base64 conversion failed: {e}")
            return None
    
    async def analyze_video_with_llm(self, video_path: str = None, frame: np.ndarray = None, 
                                   prompt: str = "Analyze this gaming footage and provide commentary") -> str:
        """
        Process video/frame with Qwen2.5-Omni LLM service
        
        Args:
            video_path: Path to video file (for video analysis)
            frame: Single frame for analysis (alternative to video)
            prompt: Text prompt for analysis
            
        Returns:
            Analysis result from Qwen2.5-Omni
        """
        try:
            # Get LLM service from app context
            llm_service = getattr(self.app_context, 'llm_service', None)
            if not llm_service:
                self.logger.error("[VideoService] LLM service not available")
                return "LLM service not available"
            
            # Prepare multimodal input
            if frame is not None:
                # Single frame analysis
                base64_image = self.frame_to_base64(frame)
                if not base64_image:
                    return "Failed to process frame"
                
                # Create multimodal prompt
                multimodal_prompt = f"""
                <image>data:image/jpeg;base64,{base64_image}</image>
                
                {prompt}
                
                Focus on:
                - Game UI elements (health, mana, inventory)
                - Character status and actions
                - Environmental context
                - Strategic opportunities or threats
                - Provide gaming commentary as if you're a knowledgeable gaming assistant
                """
                
                self.logger.debug(f"[VideoService] Analyzing frame with LLM: {prompt[:50]}...")
                
                # Get response from LLM
                response = await llm_service.get_response(multimodal_prompt, "VideoAnalysis")
                
                if response and hasattr(response, 'content'):
                    result = response.content
                elif isinstance(response, str):
                    result = response
                else:
                    result = str(response)
                
                self.logger.info(f"[VideoService] LLM analysis: {result[:100]}...")
                return result
                
            else:
                return "No valid input provided for analysis"
                
        except Exception as e:
            self.logger.error(f"[VideoService] Video analysis failed: {e}")
            return f"Analysis error: {str(e)}"
    
    async def get_current_screen_analysis(self, prompt: str = "What's happening on screen right now?") -> str:
        """
        Get immediate analysis of current screen
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Analysis result
        """
        frame = self.capture_screen()
        if frame is None:
            return "Unable to capture screen"
            
        return await self.analyze_video_with_llm(frame=frame, prompt=prompt)
    
    def get_video_stats(self) -> Dict[str, Any]:
        """Get video service statistics"""
        return {
            "analysis_enabled": self.analysis_enabled,
            "gaming_mode": self.gaming_mode,
            "is_processing": self.is_processing,
            "target_fps": self.target_fps,
            "chunk_duration": self.chunk_duration,
            "capture_quality": self.capture_quality,
            "mss_available": MSS_AVAILABLE,
            "capture_region": self.capture_region
        }
