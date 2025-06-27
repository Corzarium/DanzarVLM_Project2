#!/usr/bin/env python3
"""
Qwen2.5-VL Streaming Service for DanzarVLM
Collects frames for 1 minute, then provides comprehensive analysis and commentary
"""

import asyncio
import base64
import cv2
import logging
import numpy as np
import time
from typing import Optional, Dict, Any, List, Callable, Union, Awaitable
import aiohttp

class QwenStreamingService:
    """Qwen2.5-VL streaming service with 1-minute collection and comprehensive analysis
    
    Architecture:
    - Uses OBS service for frame capture (VL models disabled to prevent multiple loading)
    - Uses CUDA server directly for Qwen2.5-VL analysis (no local model loading)
    - Collects frames every 5 seconds for 1 minute, then provides comprehensive commentary
    - Broadcasts final commentary through Discord voice (TTS) and text channels
    """
    
    def __init__(self, app_context):
        """Initialize Qwen2.5-VL streaming service"""
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Service configuration
        self.cuda_server_url = "http://127.0.0.1:8083/v1/chat/completions"
        self.collection_duration = 60.0  # 1 minute collection period
        self.frame_interval = 5.0  # Capture frame every 5 seconds during collection (increased frequency)
        self.image_quality = 95  # High quality JPEG
        self.image_width = 1920
        self.image_height = 1080
        
        # Streaming state
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        self.analysis_callback: Optional[Union[Callable[[str, int, float], None], Callable[[str, int, float], Awaitable[None]]]] = None
        self.start_time: Optional[float] = None
        self.frame_count = 0
        self.collected_analyses: List[Dict[str, Any]] = []  # Store individual frame analyses
        
        # Performance tracking
        self.analysis_times: List[float] = []
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.comprehensive_analyses = 0
        
        # Reusable OBS service instance (VL models disabled to prevent multiple loading)
        self.obs_service = None
        
        self.logger.info("[QwenStreaming] Initialized with 1-minute collection periods")
    
    async def initialize(self) -> bool:
        """Initialize the streaming service"""
        try:
            # Test CUDA server connectivity
            cuda_available = await self._test_cuda_server()
            if cuda_available:
                self.logger.info("[QwenStreaming] CUDA server is available")
                return True
            else:
                self.logger.error("[QwenStreaming] CUDA server not available")
                return False
                
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Initialization failed: {e}")
            return False
    
    async def _test_cuda_server(self) -> bool:
        """Test CUDA server connectivity"""
        try:
            test_payload = {
                "model": "models-gguf\\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.cuda_server_url,
                    json=test_payload,
                    timeout=10
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"[QwenStreaming] CUDA server test failed: {e}")
            return False
    
    async def start_streaming(self, analysis_callback: Union[Callable[[str, int, float], None], Callable[[str, int, float], Awaitable[None]]]) -> bool:
        """Start streaming with 1-minute collection periods"""
        if self.is_streaming:
            self.logger.warning("[QwenStreaming] Already streaming")
            return False
        
        try:
            self.analysis_callback = analysis_callback
            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0
            self.collected_analyses = []
            
            # Start streaming task
            self.stream_task = asyncio.create_task(self._streaming_loop())
            
            self.logger.info(f"[QwenStreaming] Started streaming with {self.collection_duration}s collection periods")
            return True
            
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Failed to start streaming: {e}")
            self.is_streaming = False
            return False
    
    async def stop_streaming(self) -> bool:
        """Stop streaming"""
        if not self.is_streaming:
            return True
        
        try:
            self.is_streaming = False
            
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    if hasattr(self.stream_task, '__await__'):
                        await self.stream_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("[QwenStreaming] Streaming stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Error stopping streaming: {e}")
            return False
    
    async def _streaming_loop(self):
        """Main streaming loop with 1-minute collection periods"""
        while self.is_streaming:
            try:
                # Start a new collection period
                await self._collection_period()
                
                # Brief pause before next collection period
                if self.is_streaming:
                    await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[QwenStreaming] Streaming loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collection_period(self):
        """Collect and analyze frames for 1 minute, then provide comprehensive analysis"""
        collection_start = time.time()
        self.collected_analyses = []
        period_frame_count = 0
        analysis_tasks = []  # Store analysis tasks to wait for completion
        
        self.logger.info(f"[QwenStreaming] Starting new collection period ({self.collection_duration}s)")
        
        # Collect frames for the duration
        while self.is_streaming and (time.time() - collection_start) < self.collection_duration:
            try:
                elapsed_time = time.time() - collection_start
                remaining_time = self.collection_duration - elapsed_time
                
                # Capture frame
                frame = await self._capture_obs_frame()
                if frame is not None:
                    period_frame_count += 1
                    self.frame_count += 1
                    
                    # Start frame analysis as non-blocking task
                    analysis_task = asyncio.create_task(self._analyze_frame_async(frame, period_frame_count, elapsed_time))
                    analysis_tasks.append(analysis_task)
                    
                    self.logger.info(f"[QwenStreaming] Frame {period_frame_count} captured and analysis started")
                
                # Wait for next frame interval
                await asyncio.sleep(self.frame_interval)
                
            except Exception as e:
                self.logger.error(f"[QwenStreaming] Frame collection error: {e}")
                await asyncio.sleep(2)
        
        # Wait for all analysis tasks to complete
        self.logger.info(f"[QwenStreaming] Collection period ended, waiting for {len(analysis_tasks)} frame analyses to complete")
        
        if analysis_tasks:
            # Wait for all analyses with timeout
            try:
                await asyncio.wait_for(asyncio.gather(*analysis_tasks, return_exceptions=True), timeout=120)
            except asyncio.TimeoutError:
                self.logger.warning("[QwenStreaming] Some frame analyses timed out, proceeding with available results")
        
        # Filter out failed analyses and get successful results
        successful_analyses = []
        for task in analysis_tasks:
            try:
                if task.done() and not task.exception():
                    result = task.result()
                    if result:
                        successful_analyses.append(result)
            except Exception as e:
                self.logger.error(f"[QwenStreaming] Analysis task failed: {e}")
        
        self.collected_analyses = successful_analyses
        
        # Generate comprehensive analysis if we have collected frames
        if self.collected_analyses and self.analysis_callback:
            await self._generate_comprehensive_analysis(len(self.collected_analyses), time.time() - collection_start)
    
    async def _analyze_frame_async(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Optional[Dict[str, Any]]:
        """Analyze frame asynchronously and return structured result"""
        try:
            analysis = await self._analyze_frame(frame)
            if analysis:
                return {
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'analysis': analysis
                }
            return None
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Frame {frame_number} analysis failed: {e}")
            return None
    
    async def _generate_comprehensive_analysis(self, frame_count: int, duration: float):
        """Generate comprehensive analysis from all collected frame analyses"""
        try:
            self.logger.info(f"[QwenStreaming] Generating comprehensive analysis from {frame_count} frames")
            
            # Create summary of all frame analyses
            frame_summaries = []
            for i, frame_data in enumerate(self.collected_analyses, 1):
                frame_summaries.append(f"Frame {i} ({frame_data['timestamp']:.1f}s): {frame_data['analysis']}")
            
            combined_summary = "\n".join(frame_summaries)
            
            # Create comprehensive analysis prompt
            comprehensive_prompt = f"""Based on the following {frame_count} frame analyses collected over {duration:.1f} seconds, provide a comprehensive gaming commentary:\n\n{combined_summary}\n\nPlease provide:\n1. **Game Identification**: What game is being played?\n2. **Overall Activity**: What was happening during this time period?\n3. **Key Events**: What were the most important events or changes?\n4. **Player Actions**: What actions was the player taking?\n5. **Gaming Commentary**: Provide engaging commentary as if you're a gaming commentator\n6. **Trends**: Any patterns or trends you noticed\n7. **Recommendations**: What should the player do next?\n\nBe engaging, informative, and provide the kind of commentary you'd hear from a professional gaming stream."""
            
            # Send to CUDA server for comprehensive analysis
            comprehensive_analysis = await self._analyze_with_cuda_comprehensive(comprehensive_prompt)
            
            if comprehensive_analysis and self.analysis_callback and self.start_time:
                self.comprehensive_analyses += 1
                running_time = time.time() - self.start_time
                
                # Generate TTS audio from the comprehensive commentary
                await self._generate_and_broadcast_tts(comprehensive_analysis, frame_count, duration)
                
                # Call analysis callback with comprehensive result (for text channel)
                self.analysis_callback(comprehensive_analysis, frame_count, duration)
                
                self.logger.info(f"[QwenStreaming] Comprehensive analysis completed for {frame_count} frames")
            else:
                self.logger.warning("[QwenStreaming] Failed to generate comprehensive analysis")
                
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Comprehensive analysis error: {e}")
    
    async def _generate_and_broadcast_tts(self, commentary: str, frame_count: int, duration: float):
        """Generate TTS audio from commentary and broadcast through Discord"""
        try:
            # Get TTS service from app context
            tts_service = self.app_context.get_service('tts_service')
            if not tts_service:
                self.logger.error("[QwenStreaming] TTS service not available")
                return
            
            # Generate TTS audio
            self.logger.info(f"[QwenStreaming] Generating TTS audio for {len(commentary)} character commentary")
            tts_audio = tts_service.generate_audio(commentary)
            
            if not tts_audio:
                self.logger.error("[QwenStreaming] Failed to generate TTS audio")
                return
            
            # Get Discord bot instance from app context
            discord_bot = self.app_context.get_service('discord_bot')
            if not discord_bot:
                self.logger.error("[QwenStreaming] Discord bot not available")
                return
            
            # Play TTS audio through Discord voice channel
            self.logger.info(f"[QwenStreaming] Broadcasting TTS commentary through Discord voice")
            await discord_bot._play_tts_audio_with_feedback_prevention(tts_audio)
            
            self.logger.info(f"[QwenStreaming] TTS commentary broadcast completed - {frame_count} frames analyzed over {duration:.1f}s")
            
        except Exception as e:
            self.logger.error(f"[QwenStreaming] TTS broadcast error: {e}")
    
    async def _capture_obs_frame(self) -> Optional[np.ndarray]:
        """Capture high-resolution frame from OBS"""
        try:
            # Initialize OBS service if not already done
            if self.obs_service is None:
                from services.obs_video_service import OBSVideoService
                self.obs_service = OBSVideoService(self.app_context)
                
                # Disable VL model initialization to prevent multiple model loading
                self.obs_service.use_qwen_vl = False
                self.obs_service.use_llamacpp_vl = False
                
                obs_success = await self.obs_service.initialize()
                if not obs_success:
                    self.logger.error("[QwenStreaming] Failed to initialize OBS service")
                    return None
                
                self.logger.info("[QwenStreaming] OBS service initialized (VL models disabled)")
            
            # Capture screenshot (uses service's configured settings)
            frame = self.obs_service.capture_obs_screenshot()
            
            if frame is None:
                self.logger.warning("[QwenStreaming] Failed to capture OBS screenshot")
                return None
            
            return frame
            
        except Exception as e:
            self.logger.error(f"[QwenStreaming] OBS capture error: {e}")
            return None
    
    def _frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Convert OpenCV frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality])
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            return img_base64
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Frame to base64 conversion error: {e}")
            return None
    
    async def _analyze_frame(self, frame: np.ndarray) -> Optional[str]:
        """Analyze a single frame using Qwen2.5-VL CUDA server"""
        try:
            # Convert frame to base64
            frame_base64 = self._frame_to_base64(frame)
            if not frame_base64:
                return None
            
            # Create analysis prompt
            prompt = """Analyze this gaming screenshot and provide a brief, engaging commentary. Focus on:
1. What game is being played
2. What's happening in the scene
3. Any notable actions, events, or gameplay elements
4. Brief commentary that would be interesting for viewers

Keep the response concise and engaging for live streaming commentary."""
            
            # Use Qwen2.5-VL CUDA server directly
            analysis = await self._analyze_with_cuda_comprehensive(prompt, frame_base64)
            
            if analysis:
                self.successful_analyses += 1
                return analysis
            else:
                self.failed_analyses += 1
                return None
                
        except Exception as e:
            self.logger.error(f"[QwenStreaming] Frame analysis error: {e}")
            self.failed_analyses += 1
            return None
    
    async def _analyze_with_cuda_comprehensive(self, prompt: str, image_base64: Optional[str] = None) -> Optional[str]:
        """Analyze with Qwen2.5-VL CUDA server using comprehensive prompt"""
        try:
            # Create the message with or without image
            if image_base64:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ]
            else:
                # Text-only analysis
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            
            payload = {
                "model": "Qwen2.5-VL-7B-Instruct",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.cuda_server_url,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            return content.strip()
                    else:
                        self.logger.error(f"[QwenStreaming] CUDA server error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"[QwenStreaming] CUDA analysis error: {e}")
            return None
    
    def is_active(self) -> bool:
        """Check if streaming is active"""
        return self.is_streaming
    
    def get_status(self) -> Dict[str, Any]:
        """Get streaming status"""
        status = {
            'is_streaming': self.is_streaming,
            'frame_count': self.frame_count,
            'collection_duration': self.collection_duration,
            'frame_interval': self.frame_interval,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'comprehensive_analyses': self.comprehensive_analyses
        }
        
        if self.start_time:
            status['running_time'] = time.time() - self.start_time
        
        if self.analysis_times:
            status['avg_analysis_time'] = sum(self.analysis_times) / len(self.analysis_times)
            status['last_analysis_time'] = self.analysis_times[-1]
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_streaming()
        
        # Cleanup OBS service
        if self.obs_service:
            try:
                self.obs_service.disconnect()
                self.obs_service = None
                self.logger.info("[QwenStreaming] OBS service cleaned up")
            except Exception as e:
                self.logger.error(f"[QwenStreaming] Error cleaning up OBS service: {e}")

