#!/usr/bin/env python3
"""
Qwen2.5-VL Vision Service - Clean, focused vision analysis
"""

import asyncio
import aiohttp
import base64
import logging
from typing import Optional, Dict, Any
from core.app_context import AppContext

class QwenVisionService:
    """Clean vision service using Qwen2.5-VL for game analysis"""
    
    def __init__(self, app_context: AppContext):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Qwen2.5-VL configuration
        self.qwen_url = "http://localhost:8083"
        self.qwen_timeout = 60  # seconds
        self.qwen_available = False
        
        self.logger.info(f"[QwenVisionService] Initialized - Qwen2.5-VL: {self.qwen_url}")
    
    async def initialize(self) -> bool:
        """Initialize and check service availability"""
        self.logger.info("[QwenVisionService] Checking Qwen2.5-VL availability...")
        
        # Check Qwen2.5-VL availability
        self.qwen_available = await self._check_qwen_health()
        
        self.logger.info(f"[QwenVisionService] Qwen2.5-VL available: {self.qwen_available}")
        
        return self.qwen_available
    
    async def _check_qwen_health(self) -> bool:
        """Check if Qwen2.5-VL server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.qwen_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check for Qwen2.5-VL format {"status":"ok"}
                        return data.get('status') == 'ok'
        except Exception as e:
            self.logger.warning(f"[QwenVisionService] Qwen2.5-VL health check failed: {e}")
        return False
    
    async def analyze_screenshot(self, screenshot_data: str, game_type: str, mode: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Analyze screenshot with Qwen2.5-VL
        
        Args:
            screenshot_data: Base64 encoded screenshot
            game_type: Type of game for context
            mode: Analysis mode (auto, fast, detailed)
        
        Returns:
            Analysis result or None if failed
        """
        if not self.qwen_available:
            self.logger.error("[QwenVisionService] Qwen2.5-VL not available")
            return None
        
        return await self._analyze_with_qwen(screenshot_data, game_type, mode)
    
    async def _analyze_with_qwen(self, screenshot_data: str, game_type: str, mode: str) -> Optional[Dict[str, Any]]:
        """Analyze with Qwen2.5-VL (vision-based analysis)"""
        try:
            # Prepare prompt based on mode
            if mode == "fast":
                prompt = f"""Analyze this {game_type} screenshot quickly. Focus on:
- Main action/activity
- Key UI elements
- Player status/health
- Immediate context
Keep response brief and focused."""
            elif mode == "detailed":
                prompt = f"""Provide detailed analysis of this {game_type} screenshot:
- Describe the scene and atmosphere
- Identify all UI elements and their states
- Analyze player actions and intentions
- Note any important game events or changes
- Provide context-aware commentary
Be thorough but concise."""
            else:  # auto mode
                prompt = f"""Analyze this {game_type} screenshot:
- Describe what's happening in the game
- Identify key UI elements and player status
- Provide relevant commentary based on the scene
- Note any important details or changes
Be informative and engaging."""
            
            # Prepare request payload
            payload = {
                "model": "qwen2.5-vl",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_data}"}}
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.qwen_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.qwen_timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        return {
                            "analysis": content,
                            "service_used": "qwen2.5-vl",
                            "mode": mode,
                            "game_type": game_type,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    else:
                        self.logger.error(f"[QwenVisionService] Qwen2.5-VL API error {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("[QwenVisionService] Qwen2.5-VL request timed out")
            return None
        except Exception as e:
            self.logger.error(f"[QwenVisionService] Qwen2.5-VL analysis failed: {e}")
            return None
    
    async def get_fast_analysis(self, screenshot_data: str, game_type: str) -> Optional[Dict[str, Any]]:
        """Get fast analysis"""
        return await self.analyze_screenshot(screenshot_data, game_type, "fast")
    
    async def get_detailed_analysis(self, screenshot_data: str, game_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis"""
        return await self.analyze_screenshot(screenshot_data, game_type, "detailed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "qwen_vision",
            "qwen_available": self.qwen_available,
            "qwen_url": self.qwen_url,
            "qwen_timeout": self.qwen_timeout
        } 