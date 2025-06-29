#!/usr/bin/env python3
"""
Streaming Video Service - Real-time game commentary with Qwen2.5-VL
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from core.app_context import AppContext
from services.qwen_vision_service import QwenVisionService

class StreamingVideoService:
    """Real-time video streaming and commentary service"""
    
    def __init__(self, app_context: AppContext):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Vision service
        self.vision_service = QwenVisionService(app_context)
        
        # Commentary settings
        self.commentary_interval = self.config.get('vision_commentary_interval', 10)  # seconds
        self.last_commentary_time = 0
        self.is_running = False
        
        self.logger.info("[StreamingVideoService] Initialized")
    
    async def initialize(self) -> bool:
        """Initialize the streaming video service"""
        try:
            # Initialize vision service
            vision_ready = await self.vision_service.initialize()
            if not vision_ready:
                self.logger.error("[StreamingVideoService] Vision service initialization failed")
                return False
            
            self.logger.info("[StreamingVideoService] Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"[StreamingVideoService] Initialization failed: {e}")
            return False
    
    async def start_commentary_loop(self):
        """Start the real-time commentary loop"""
        if self.is_running:
            self.logger.warning("[StreamingVideoService] Commentary loop already running")
            return
        
        self.is_running = True
        self.logger.info("[StreamingVideoService] Starting commentary loop")
        
        try:
            while self.is_running and not self.app_context.shutdown_event.is_set():
                await self._process_commentary_cycle()
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            self.logger.error(f"[StreamingVideoService] Commentary loop error: {e}")
        finally:
            self.is_running = False
            self.logger.info("[StreamingVideoService] Commentary loop stopped")
    
    async def _process_commentary_cycle(self):
        """Process one commentary cycle"""
        current_time = time.time()
        
        # Check if it's time for commentary
        if current_time - self.last_commentary_time < self.commentary_interval:
            return
        
        # Get latest frame
        frame = await self._get_latest_frame()
        if frame is None:
            return
        
        # Generate commentary
        commentary = await self._generate_commentary(frame)
        if commentary:
            await self._deliver_commentary(commentary)
            self.last_commentary_time = current_time
    
    async def _get_latest_frame(self) -> Optional[str]:
        """Get the latest frame for analysis"""
        try:
            # Get frame from NDI service
            if hasattr(self.app_context, 'ndi_service') and self.app_context.ndi_service:
                frame = self.app_context.ndi_service.last_captured_frame
                if frame is not None:
                    # Convert to base64
                    import cv2
                    import base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    return base64.b64encode(buffer).decode('utf-8')
            
            return None
            
        except Exception as e:
            self.logger.error(f"[StreamingVideoService] Frame capture error: {e}")
            return None
    
    async def _generate_commentary(self, frame_data: str) -> Optional[str]:
        """Generate commentary for the frame"""
        try:
            # Get game type from context
            game_type = self.config.get('current_game', 'generic')
            
            # Use vision service for analysis
            result = await self.vision_service.analyze_screenshot(frame_data, game_type, "auto")
            
            if result and result.get('analysis'):
                return result['analysis']
            
            return None
            
        except Exception as e:
            self.logger.error(f"[StreamingVideoService] Commentary generation error: {e}")
            return None
    
    async def _deliver_commentary(self, commentary: str):
        """Deliver commentary through TTS"""
        try:
            # Use TTS service to speak commentary
            if hasattr(self.app_context, 'tts_service') and self.app_context.tts_service:
                await self.app_context.tts_service.speak_text(commentary)
            else:
                self.logger.info(f"[StreamingVideoService] Commentary: {commentary}")
                
        except Exception as e:
            self.logger.error(f"[StreamingVideoService] Commentary delivery error: {e}")
    
    def stop_commentary_loop(self):
        """Stop the commentary loop"""
        self.is_running = False
        self.logger.info("[StreamingVideoService] Stopping commentary loop")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "streaming_video",
            "is_running": self.is_running,
            "commentary_interval": self.commentary_interval,
            "last_commentary_time": self.last_commentary_time,
            "vision_service_status": self.vision_service.get_status()
        }

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
        
        await service.start_commentary_loop()
        print("✓ Commentary loop started successfully")
        
        await service.stop_commentary_loop()
        print("✓ Commentary loop stopped successfully")
        
        status = service.get_status()
        print(f"✓ Streaming video service status: {status}")
        
        await service.cleanup()
    else:
        print("✗ Streaming video service initialization failed")

if __name__ == "__main__":
    asyncio.run(test_streaming_video()) 