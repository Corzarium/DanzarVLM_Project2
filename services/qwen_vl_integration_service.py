#!/usr/bin/env python3
"""
Qwen2.5-VL Integration Service for DanzarVLM
Handles OBS vision analysis and Discord integration with CUDA acceleration
"""

import asyncio
import base64
import cv2
import json
import logging
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import aiohttp

class QwenVLIntegrationService:
    """Comprehensive Qwen2.5-VL integration for OBS vision and Discord"""
    
    def __init__(self, app_context):
        """Initialize Qwen2.5-VL integration service"""
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Service configuration
        self.cuda_server_url = "http://127.0.0.1:8083/v1/chat/completions"
        self.transformers_fallback = False  # Disabled by default, only enable if CUDA fails
        self.max_analysis_duration = 30.0
        self.commentary_interval = 15.0
        self.min_commentary_interval = 10.0
        self.max_commentary_sentences = 3
        
        # Gaming prompts for different scenarios
        self.gaming_prompts = [
            "Analyze this gaming screenshot. What game is being played, what's happening, and what actions should the player take? Be specific about what you see.",
            "Describe this video game scene in detail. What characters, objects, and environment elements do you see? What type of game is this?",
            "What's happening in this game? Describe the current situation, any threats, and what the player might be trying to accomplish.",
            "Briefly describe what's happening in this gaming screenshot in 1-2 sentences."
        ]
        
        # State management
        self.is_analyzing = False
        self.last_commentary_time = None
        self.analysis_callbacks = []
        self.commentary_history = []
        self.max_history_size = 10
        
        # Performance tracking
        self.analysis_times = []
        self.successful_analyses = 0
        self.failed_analyses = 0
        
        self.logger.info("[QwenVLIntegration] Initialized with CUDA server integration")
    
    async def initialize(self) -> bool:
        """Initialize the integration service"""
        try:
            # Test CUDA server connectivity
            cuda_available = await self._test_cuda_server()
            if cuda_available:
                self.logger.info("[QwenVLIntegration] CUDA server is available and responsive")
                self.transformers_fallback = False  # Disable fallback since CUDA is working
            else:
                self.logger.warning("[QwenVLIntegration] CUDA server not available, enabling transformers fallback")
                self.transformers_fallback = True
            
            # Initialize transformers fallback only if needed
            if self.transformers_fallback:
                try:
                    from services.qwen_vl_service import QwenVLService
                    self.transformers_service = QwenVLService(self.app_context)
                    transformers_success = await self.transformers_service.initialize()
                    if transformers_success:
                        self.logger.info("[QwenVLIntegration] Transformers fallback initialized")
                    else:
                        self.logger.warning("[QwenVLIntegration] Transformers fallback failed")
                except Exception as e:
                    self.logger.error(f"[QwenVLIntegration] Error initializing transformers fallback: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] Initialization failed: {e}")
            return False
    
    async def _test_cuda_server(self) -> bool:
        """Test CUDA server connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:8083/health", timeout=5) as response:
                    return response.status == 200
        except:
            try:
                # Try a simple chat completion test
                test_payload = {
                    "model": "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
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
            except:
                return False
    
    async def analyze_obs_frame(self, frame: np.ndarray, prompt: Optional[str] = None) -> Optional[str]:
        """Analyze OBS frame with Qwen2.5-VL with improved detail retention"""
        if self.is_analyzing:
            self.logger.debug("[QwenVLIntegration] Analysis already in progress, skipping")
            return None
        
        self.is_analyzing = True
        start_time = time.time()
        
        try:
            # Convert frame to base64 with HIGH quality to preserve details
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # Enhanced prompt for better text and detail recognition
            if not prompt:
                prompt = """Analyze this gaming screenshot in detail. Pay special attention to:
1. Character names and text visible on screen - read them carefully
2. UI elements, status bars, and any displayed information
3. What game is being played and what's happening
4. Character appearance, equipment, and actions
5. Environment details and location

Be very specific about any names, text, or numbers you can see. If you see character names, spell them exactly as they appear."""
            
            # Try CUDA server first
            analysis = await self._analyze_with_cuda(img_base64, prompt)
            
            # Fallback to transformers if CUDA fails
            if not analysis and self.transformers_fallback:
                self.logger.info("[QwenVLIntegration] CUDA analysis failed, trying transformers fallback")
                analysis = await self._analyze_with_transformers(frame, prompt)
            
            if analysis:
                duration = time.time() - start_time
                self.analysis_times.append(duration)
                self.successful_analyses += 1
                
                # Keep only recent analysis times
                if len(self.analysis_times) > 20:
                    self.analysis_times.pop(0)
                
                self.logger.info(f"[QwenVLIntegration] Analysis completed in {duration:.2f}s")
                return analysis
            else:
                self.failed_analyses += 1
                self.logger.warning("[QwenVLIntegration] Analysis failed")
                return None
                
        except Exception as e:
            self.failed_analyses += 1
            self.logger.error(f"[QwenVLIntegration] Analysis error: {e}")
            return None
        finally:
            self.is_analyzing = False
    
    async def _analyze_with_cuda(self, img_base64: str, prompt: str) -> Optional[str]:
        """Analyze image using CUDA-enabled Qwen2.5-VL server"""
        try:
            payload = {
                "model": "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 200,
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
                        return result['choices'][0]['message']['content']
                    else:
                        self.logger.error(f"[QwenVLIntegration] CUDA server error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] CUDA analysis error: {e}")
            return None
    
    async def _analyze_with_transformers(self, frame: np.ndarray, prompt: str) -> Optional[str]:
        """Analyze image using transformers Qwen2.5-VL fallback"""
        try:
            if hasattr(self, 'transformers_service'):
                return await self.transformers_service.analyze_image(frame, prompt)
            return None
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] Transformers analysis error: {e}")
            return None
    
    async def generate_gaming_commentary(self, frame: np.ndarray, context: str = "") -> Optional[str]:
        """Generate gaming commentary for Discord"""
        # Check if enough time has passed since last commentary
        if self.last_commentary_time:
            time_since_last = time.time() - self.last_commentary_time
            if time_since_last < self.min_commentary_interval:
                return None
        
        try:
            # Create contextual prompt
            commentary_prompt = f"""Analyze this gaming screenshot and provide engaging commentary for a live stream. 
            Focus on what's happening, any interesting details, and provide helpful insights.
            {context}
            
            Keep the commentary concise and entertaining, suitable for real-time gaming commentary."""
            
            analysis = await self.analyze_obs_frame(frame, commentary_prompt)
            
            if analysis:
                self.last_commentary_time = time.time()
                
                # Add to commentary history
                commentary_entry = {
                    'timestamp': datetime.now(),
                    'commentary': analysis,
                    'context': context
                }
                self.commentary_history.append(commentary_entry)
                
                # Keep history size manageable
                if len(self.commentary_history) > self.max_history_size:
                    self.commentary_history.pop(0)
                
                return analysis
            
            return None
            
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] Commentary generation error: {e}")
            return None
    
    async def process_discord_vision_query(self, frame: np.ndarray, user_query: str, user_name: str) -> Optional[str]:
        """Process vision-based queries from Discord users"""
        try:
            # Create user-specific prompt
            prompt = f"""A Discord user named {user_name} is asking about this gaming screenshot: "{user_query}"
            
            Please provide a helpful and detailed response about what you see in the image, 
            addressing their specific question. Be conversational and informative."""
            
            analysis = await self.analyze_obs_frame(frame, prompt)
            return analysis
            
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] Discord vision query error: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = np.mean(self.analysis_times) if self.analysis_times else 0
        total_analyses = self.successful_analyses + self.failed_analyses
        success_rate = (self.successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
        
        return {
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'total_analyses': total_analyses,
            'success_rate': success_rate,
            'average_time': avg_time,
            'recent_times': self.analysis_times[-5:] if self.analysis_times else [],
            'cuda_available': self._test_cuda_server_sync(),
            'transformers_fallback': self.transformers_fallback
        }
    
    def _test_cuda_server_sync(self) -> bool:
        """Synchronous test of CUDA server"""
        try:
            response = requests.get("http://127.0.0.1:8083/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_commentary_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent commentary history"""
        return self.commentary_history[-limit:] if self.commentary_history else []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'transformers_service'):
                await self.transformers_service.cleanup()
        except Exception as e:
            self.logger.error(f"[QwenVLIntegration] Cleanup error: {e}") 