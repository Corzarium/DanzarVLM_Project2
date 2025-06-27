#!/usr/bin/env python3
"""
Streaming Video Analysis Service - Regular LLM Commentary
"""

import asyncio
import aiohttp
import base64
import time
import logging
import json
from typing import List, Dict, Optional, Callable
from datetime import datetime
import obsws_python as obs
import threading
from collections import deque
from PIL import Image

class StreamingVideoService:
    """Optimized streaming video service with vision analysis."""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # OBS WebSocket connection
        self.obs_client = None
        self.obs_source_name = "NDI® Source"  # Default NDI source name
        
        # Vision service - Use MiniGPT-4 Video Service
        self.vision_service = None
        
        # Performance settings
        self.screenshot_interval = 30.0  # Reduced from 60s for more frequent updates
        self.frames_per_analysis = 1  # Only analyze most recent frame for speed
        self.max_screenshot_size = (480, 270)  # Small for speed
        self.jpeg_quality = 80  # Good balance of quality and speed
        
        # Commentary loop control
        self.commentary_task = None
        self.is_streaming = False
        
        self.logger.info("[StreamingVideoService] Initialized for FAST screenshot analysis")

    async def initialize(self, vision_service=None) -> bool:
        """Initialize the streaming video service with hybrid vision support."""
        try:
            self.logger.info("[StreamingVideoService] Initializing with hybrid vision support...")
            
            # Initialize OBS connection
            if not await self.connect_obs():
                self.logger.warning("[StreamingVideoService] OBS connection failed, but continuing...")
            
            # Initialize vision service
            if vision_service:
                self.vision_service = vision_service
                self.logger.info("[StreamingVideoService] Using provided vision service")
            else:
                # Import and initialize Hybrid Vision Service
                try:
                    from services.hybrid_vision_service import HybridVisionService
                    self.vision_service = HybridVisionService(self.app_context)
                    if not await self.vision_service.initialize():
                        self.logger.error("Failed to initialize Hybrid Vision Service")
                        return False
                    self.logger.info("[StreamingVideoService] Successfully initialized with Hybrid Vision Service")
                except ImportError:
                    self.logger.error("Hybrid Vision Service not available")
                    return False
            
            self.logger.info("✅ Streaming Video Service initialized with Hybrid Vision")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming video service: {e}")
            return False

    async def connect_obs(self) -> bool:
        """Connect to OBS WebSocket."""
        try:
            self.obs_client = obs.ReqClient(host='localhost', port=4455, password='', timeout=5)
            
            # Test connection
            version_info = self.obs_client.get_version()
            self.logger.info(f"[StreamingVideoService] OBS connected - Version: {version_info.obs_version}")
            
            # Verify NDI source exists
            try:
                source_settings = self.obs_client.get_input_settings(self.obs_source_name)
                self.logger.info(f"[StreamingVideoService] Source verified: {self.obs_source_name}")
                return True
            except Exception as e:
                self.logger.warning(f"NDI source '{self.obs_source_name}' not found: {e}")
                # Try to find any available source
                inputs = self.obs_client.get_input_list()
                if inputs.inputs:
                    self.obs_source_name = inputs.inputs[0]['inputName']
                    self.logger.info(f"[StreamingVideoService] Using alternative source: {self.obs_source_name}")
                    return True
                else:
                    self.logger.error("No OBS inputs available")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to OBS: {e}")
            return False

    async def capture_screenshot(self) -> Optional[str]:
        """Capture optimized screenshot from OBS."""
        try:
            if not self.obs_client:
                if not await self.connect_obs():
                    return None
            
            # Get screenshot with optimized settings
            screenshot_response = self.obs_client.get_source_screenshot(
                self.obs_source_name,
                img_format='jpg',
                width=self.max_screenshot_size[0],
                height=self.max_screenshot_size[1],
                quality=60  # Good quality for analysis
            )
            
            # Extract base64 data (remove data:image/jpeg;base64, prefix)
            image_data = screenshot_response.image_data
            if image_data.startswith('data:image/jpeg;base64,'):
                image_data = image_data[len('data:image/jpeg;base64,'):]
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None

    async def analyze_current_frame(self) -> Optional[str]:
        """Analyze the current frame with hybrid vision service (auto mode for optimal performance)."""
        try:
            # Capture screenshot
            screenshot_data = await self.capture_screenshot()
            if not screenshot_data:
                return "Unable to capture screenshot from OBS"
            
            # Use Hybrid Vision Service for analysis (auto mode - tries Phi-4 first, falls back to MiniGPT-4)
            if hasattr(self.vision_service, 'analyze_screenshot'):
                analysis_result = await self.vision_service.analyze_screenshot(screenshot_data, "everquest", mode="auto")
                if analysis_result and 'analysis' in analysis_result:
                    service_used = analysis_result.get('service_used', 'unknown')
                    analysis_time = analysis_result.get('analysis_time', 0)
                    return f"🎮 EverQuest Analysis ({service_used.upper()}): {analysis_result['analysis']} [⏱️{analysis_time:.1f}s]"
                else:
                    return "🎮 EverQuest scene detected but no specific details available"
            else:
                self.logger.error("Hybrid vision service does not have analyze_screenshot method")
                return "Hybrid vision service not properly configured"
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    async def analyze_current_frame_detailed(self) -> Optional[str]:
        """Analyze the current frame with hybrid vision service in detailed mode."""
        try:
            # Capture screenshot
            screenshot_data = await self.capture_screenshot()
            if not screenshot_data:
                return "Unable to capture screenshot from OBS"
            
            # Use Hybrid Vision Service for detailed analysis (prefers MiniGPT-4 for detailed analysis)
            if hasattr(self.vision_service, 'analyze_screenshot'):
                analysis_result = await self.vision_service.analyze_screenshot(screenshot_data, "everquest", mode="detailed", fast_mode=False)
                if analysis_result and 'analysis' in analysis_result:
                    service_used = analysis_result.get('service_used', 'unknown')
                    analysis_time = analysis_result.get('analysis_time', 0)
                    return f"🎮 EverQuest Detailed Analysis ({service_used.upper()}): {analysis_result['analysis']} [⏱️{analysis_time:.1f}s]"
                else:
                    return "🎮 EverQuest scene detected but no specific details available"
            else:
                self.logger.error("Hybrid vision service does not have analyze_screenshot method")
                return "Hybrid vision service not properly configured"
            
        except Exception as e:
            self.logger.error(f"Detailed frame analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    async def start_commentary_loop(self) -> bool:
        """Start the commentary loop with working vision analysis."""
        try:
            if self.is_streaming:
                self.logger.warning("Commentary loop already running")
                return True
            
            if not await self.connect_obs():
                return False
            
            self.is_streaming = True
            self.start_time = time.time()
            self.frames_collected = 0
            self.batches_completed = 0
            self.commentary_task = asyncio.create_task(self._commentary_loop())
            
            self.logger.info(f"[StreamingVideoService] Started regular commentary (every {self.screenshot_interval}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start commentary loop: {e}")
            return False

    async def start_streaming(self, analysis_callback=None) -> bool:
        """Start streaming with callback support - compatible with Discord commands."""
        try:
            self.analysis_callback = analysis_callback
            return await self.start_commentary_loop()
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False

    async def stop_streaming(self):
        """Stop streaming - compatible with Discord commands."""
        try:
            await self.stop_commentary_loop()
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {e}")

    def is_active(self) -> bool:
        """Check if streaming is active."""
        return self.is_streaming

    def get_status(self) -> dict:
        """Get streaming status information."""
        return {
            'is_active': self.is_streaming,
            'frames_collected': getattr(self, 'frames_collected', 0),
            'batches_completed': getattr(self, 'batches_completed', 0),
            'running_time': getattr(self, 'running_time', 0.0),
            'collection_progress': 'Active' if self.is_streaming else 'Inactive',
            'next_analysis': f'{self.screenshot_interval}s intervals'
        }

    async def _commentary_loop(self):
        """Main commentary loop - now with working MiniGPT-4 analysis."""
        while self.is_streaming:
            try:
                start_time = time.time()
                
                # Analyze current frame
                self.logger.info(f"[StreamingVideoService] Using MiniGPT-4 for detailed analysis of 1 frame")
                commentary = await self.analyze_current_frame()
                
                if commentary:
                    analysis_time = time.time() - start_time
                    self.logger.info(f"[StreamingVideoService] MiniGPT-4 analysis completed in {analysis_time:.2f}s")
                    
                    # Call the analysis callback if provided (for Discord integration)
                    if hasattr(self, 'analysis_callback') and self.analysis_callback:
                        try:
                            await self.analysis_callback(commentary, 1, analysis_time)
                        except Exception as e:
                            self.logger.error(f"Analysis callback error: {e}")
                    else:
                        # For now, just log it
                        self.logger.info(f"[Commentary] {commentary}")
                else:
                    self.logger.warning("No commentary generated")
                
                # Update statistics
                self.frames_collected = getattr(self, 'frames_collected', 0) + 1
                self.running_time = time.time() - getattr(self, 'start_time', start_time)
                
                # Wait for next analysis
                await asyncio.sleep(self.screenshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Commentary loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def stop_commentary_loop(self):
        """Stop the commentary loop."""
        try:
            if self.commentary_task:
                self.commentary_task.cancel()
                try:
                    await self.commentary_task
                except asyncio.CancelledError:
                    pass
                self.commentary_task = None
                self.logger.info("[StreamingVideoService] Commentary loop cancelled")
            
            self.is_streaming = False
            self.logger.info("[StreamingVideoService] Stopped streaming")
            
        except Exception as e:
            self.logger.error(f"Error stopping commentary loop: {e}")

    async def get_single_analysis(self) -> Optional[str]:
        """Get a single frame analysis immediately."""
        try:
            return await self.analyze_current_frame()
        except Exception as e:
            self.logger.error(f"Single analysis failed: {e}")
            return None

    async def get_detailed_analysis(self) -> Optional[str]:
        """Get a detailed frame analysis immediately (slower but more comprehensive)."""
        try:
            return await self.analyze_current_frame_detailed()
        except Exception as e:
            self.logger.error(f"Detailed analysis failed: {e}")
            return None

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.stop_commentary_loop()
            
            if self.obs_client:
                self.obs_client.disconnect()
                self.obs_client = None
            
            if self.vision_service and hasattr(self.vision_service, 'cleanup'):
                await self.vision_service.cleanup()
            
            self.logger.info("[StreamingVideoService] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Test function
async def test_streaming_video():
    """Test streaming video service with working vision."""
    print("Testing Streaming Video Service with Simple Vision...")
    
    # Mock app context
    class MockContext:
        def __init__(self):
            self.logger = type('MockLogger', (), {
                'info': lambda self, msg: print(f"[INFO] {msg}"),
                'error': lambda self, msg: print(f"[ERROR] {msg}"),
                'warning': lambda self, msg: print(f"[WARN] {msg}"),
                'debug': lambda self, msg: print(f"[DEBUG] {msg}")
            })()
            self.global_settings = {}
    
    service = StreamingVideoService(MockContext())
    
    if await service.initialize():
        print("✓ Streaming video service initialized successfully")
        
        # Test single analysis
        result = await service.get_single_analysis()
        if result:
            print(f"✓ Single analysis result: {result}")
        else:
            print("✗ Single analysis failed")
        
        await service.cleanup()
    else:
        print("✗ Streaming video service initialization failed")

if __name__ == "__main__":
    asyncio.run(test_streaming_video()) 