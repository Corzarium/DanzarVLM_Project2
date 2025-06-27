"""
Qwen2.5-VL Service Client for DanzarVLM
Handles vision-language processing by calling external Qwen2.5-VL server
"""

import asyncio
import logging
import os
import tempfile
import base64
import aiohttp
import json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

class QwenVLService:
    """Client service for Qwen2.5-VL vision-language processing via external server"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Server configuration
        self.server_endpoint = self.config.get('EXTERNAL_SERVERS', {}).get('QWEN_VL_SERVER', {}).get('endpoint', 'http://localhost:8083/chat/completions')
        self.server_timeout = self.config.get('EXTERNAL_SERVERS', {}).get('QWEN_VL_SERVER', {}).get('timeout', 120)
        self.server_enabled = self.config.get('EXTERNAL_SERVERS', {}).get('QWEN_VL_SERVER', {}).get('enabled', True)
        
        # HTTP session
        self.session = None
        self.server_available = False
        
    async def initialize(self) -> bool:
        """Initialize the Qwen2.5-VL service client"""
        try:
            if not self.server_enabled:
                self.logger.warning("Qwen2.5-VL server is disabled in configuration")
                return False
                
            self.logger.info(f"🚀 Initializing Qwen2.5-VL service client...")
            self.logger.info(f"📡 Server endpoint: {self.server_endpoint}")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.server_timeout)
            )
            
            # Test server connectivity
            await self._test_server_connection()
            
            if self.server_available:
                self.logger.info("✅ Qwen2.5-VL service client initialized successfully")
                return True
            else:
                self.logger.error("❌ Failed to connect to Qwen2.5-VL server")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Qwen2.5-VL service client: {e}")
            return False
    
    async def _test_server_connection(self):
        """Test connection to the external server"""
        try:
            if not self.session:
                return
                
            # Try to get server health status
            async with self.session.get(f"{self.server_endpoint.replace('/chat/completions', '/health')}") as response:
                if response.status == 200:
                    self.server_available = True
                    self.logger.info("✅ Qwen2.5-VL server is available")
                else:
                    self.server_available = False
                    self.logger.warning(f"⚠️ Qwen2.5-VL server health check failed: {response.status}")
                    
        except Exception as e:
            self.server_available = False
            self.logger.warning(f"⚠️ Could not connect to Qwen2.5-VL server: {e}")
    
    async def _call_server(self, image_base64: str, prompt: str) -> str:
        """
        Call the external Qwen2.5-VL server
        
        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt for the image
            
        Returns:
            Analysis text
        """
        try:
            if not self.server_available or not self.session:
                return "Error: Server not available"
            
            # Prepare the request payload
            payload = {
                "model": "qwen2.5-vl-7b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False
            }
            
            # Make the request
            async with self.session.post(
                self.server_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        return "Error: Invalid response format from server"
                else:
                    error_text = await response.text()
                    self.logger.error(f"Server error {response.status}: {error_text}")
                    return f"Error: Server returned status {response.status}"
                    
        except asyncio.TimeoutError:
            self.logger.error("Timeout calling Qwen2.5-VL server")
            return "Error: Server timeout"
        except Exception as e:
            self.logger.error(f"❌ Error calling Qwen2.5-VL server: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Analyze image using external Qwen2.5-VL server
        
        Args:
            image_path: Path to image file or base64 string
            prompt: Text prompt for analysis
            
        Returns:
            Analysis text
        """
        try:
            if not self.server_available:
                return "Error: Qwen2.5-VL server not available"
            
            # Convert image to base64
            if image_path.startswith('data:image'):
                # Already base64 encoded
                image_base64 = image_path.split(',')[1]
            elif os.path.exists(image_path):
                # Local file - convert to base64
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            else:
                return f"Error: Image file not found: {image_path}"
            
            # Call external server
            result = await self._call_server(image_base64, prompt)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Image analysis error: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 for model input"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode as JPEG
            success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                self.logger.error("Failed to encode frame as JPEG")
                return None
                
            # Convert to base64
            image_bytes = buffer.tobytes()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_data
            
        except Exception as e:
            self.logger.error(f"Error converting frame to base64: {e}")
            return None
    
    async def analyze_frame(self, frame: np.ndarray, prompt: str = "What's happening in this game?") -> str:
        """
        Analyze video frame using external Qwen2.5-VL server
        
        Args:
            frame: OpenCV frame (numpy array)
            prompt: Text prompt for analysis
            
        Returns:
            Analysis text
        """
        try:
            if not self.server_available:
                return "Error: Qwen2.5-VL server not available"
            
            # Convert frame to base64
            image_base64 = self.frame_to_base64(frame)
            if not image_base64:
                return "Error: Failed to convert frame to base64"
            
            # Call external server
            result = await self._call_server(image_base64, prompt)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Frame analysis error: {e}")
            return f"Error analyzing frame: {str(e)}"
    
    def save_debug_frame(self, frame: np.ndarray, prefix: str = "debug") -> str:
        """Save frame for debugging purposes"""
        try:
            debug_path = self.config.get('DEBUG_OUTPUT_PATH', './debug_frames')
            os.makedirs(debug_path, exist_ok=True)
            
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(debug_path, filename)
            
            cv2.imwrite(filepath, frame)
            self.logger.debug(f"Saved debug frame: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving debug frame: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.server_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "Qwen2.5-VL Client",
            "server_available": self.server_available,
            "server_endpoint": self.server_endpoint,
            "server_enabled": self.server_enabled,
            "session_active": self.session is not None
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None