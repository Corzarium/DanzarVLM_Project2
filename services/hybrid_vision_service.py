#!/usr/bin/env python3
"""
Hybrid Vision Service - Intelligently routes between Phi-4 (fast) and MiniGPT-4 (detailed)
"""

import asyncio
import aiohttp
import base64
import logging
import time
from typing import Optional, Dict, Any, List
from PIL import Image
import io

class HybridVisionService:
    """Hybrid vision service that routes between Phi-4 and MiniGPT-4"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Service endpoints
        self.phi4_url = "http://localhost:8083"
        self.minigpt4_url = "http://localhost:8084"
        
        # Performance thresholds
        self.phi4_timeout = 15  # seconds
        self.minigpt4_timeout = 120  # seconds
        
        # Service availability
        self.phi4_available = False
        self.minigpt4_available = False
        
        self.logger.info(f"[HybridVisionService] Initialized - Phi-4: {self.phi4_url}, MiniGPT-4: {self.minigpt4_url}")
    
    async def initialize(self) -> bool:
        """Initialize and check service availability"""
        try:
            # Check Phi-4 availability
            self.phi4_available = await self._check_phi4_health()
            
            # Check MiniGPT-4 availability
            self.minigpt4_available = await self._check_minigpt4_health()
            
            self.logger.info(f"[HybridVisionService] Phi-4 available: {self.phi4_available}")
            self.logger.info(f"[HybridVisionService] MiniGPT-4 available: {self.minigpt4_available}")
            
            return self.phi4_available or self.minigpt4_available
            
        except Exception as e:
            self.logger.error(f"[HybridVisionService] Initialization failed: {e}")
            return False
    
    async def _check_phi4_health(self) -> bool:
        """Check if Phi-4 server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.phi4_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('is_loaded', False)
                    return False
        except Exception as e:
            self.logger.warning(f"[HybridVisionService] Phi-4 health check failed: {e}")
            return False
    
    async def _check_minigpt4_health(self) -> bool:
        """Check if MiniGPT-4 server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.minigpt4_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('is_loaded', False)
                    return False
        except Exception as e:
            self.logger.warning(f"[HybridVisionService] MiniGPT-4 health check failed: {e}")
            return False
    
    async def analyze_screenshot(self, screenshot_data: str, game_type: str = "everquest", 
                               mode: str = "auto", fast_mode: bool = False) -> Optional[Dict[str, Any]]:
        """
        Analyze screenshot using hybrid approach
        
        Args:
            screenshot_data: Base64 encoded image
            game_type: Type of game (everquest, etc.)
            mode: "auto", "fast", "detailed", "phi4", "minigpt4"
            fast_mode: Whether to use fast mode (for MiniGPT-4)
        """
        try:
            # Determine which service to use
            service_choice = self._choose_service(mode, fast_mode)
            
            if service_choice == "phi4":
                return await self._analyze_with_phi4(screenshot_data, game_type)
            elif service_choice == "minigpt4":
                return await self._analyze_with_minigpt4(screenshot_data, game_type, fast_mode)
            else:
                # Auto mode - try Phi-4 first, fallback to MiniGPT-4
                return await self._analyze_auto(screenshot_data, game_type, fast_mode)
                
        except Exception as e:
            self.logger.error(f"[HybridVisionService] Analysis failed: {e}")
            return None
    
    def _choose_service(self, mode: str, fast_mode: bool) -> str:
        """Choose which service to use based on mode and availability"""
        if mode == "phi4":
            return "phi4" if self.phi4_available else "minigpt4"
        elif mode == "minigpt4":
            return "minigpt4" if self.minigpt4_available else "phi4"
        elif mode == "fast":
            return "phi4" if self.phi4_available else "minigpt4"
        elif mode == "detailed":
            return "minigpt4" if self.minigpt4_available else "phi4"
        else:  # auto mode
            return "auto"
    
    async def _analyze_auto(self, screenshot_data: str, game_type: str, fast_mode: bool) -> Optional[Dict[str, Any]]:
        """Auto mode: try Phi-4 first, fallback to MiniGPT-4"""
        try:
            # Try Phi-4 first (fast)
            if self.phi4_available:
                self.logger.info("[HybridVisionService] Auto mode: trying Phi-4 first")
                result = await self._analyze_with_phi4(screenshot_data, game_type)
                if result:
                    result['service_used'] = 'phi4'
                    result['mode'] = 'auto'
                    return result
            
            # Fallback to MiniGPT-4
            if self.minigpt4_available:
                self.logger.info("[HybridVisionService] Auto mode: falling back to MiniGPT-4")
                result = await self._analyze_with_minigpt4(screenshot_data, game_type, fast_mode)
                if result:
                    result['service_used'] = 'minigpt4'
                    result['mode'] = 'auto'
                    return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"[HybridVisionService] Auto analysis failed: {e}")
            return None
    
    async def _analyze_with_phi4(self, screenshot_data: str, game_type: str) -> Optional[Dict[str, Any]]:
        """Analyze with Phi-4 (text-based analysis)"""
        try:
            # Create a text-based prompt for Phi-4
            if game_type.lower() == "everquest":
                prompt = "Based on an EverQuest game screenshot, describe what you would typically see: the zone, character activities, UI elements, monsters, NPCs, and overall game situation. Be specific about EverQuest gameplay elements."
            else:
                prompt = f"Based on a {game_type} game screenshot, describe what you would typically see in this type of game."
            
            # Prepare request for Phi-4
            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.phi4_url}/chat/completions",
                    json=request_data,
                    timeout=self.phi4_timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_time = time.time() - start_time
                        
                        return {
                            "analysis": result.get("choices", [{}])[0].get("message", {}).get("content", "No analysis available"),
                            "model": "phi-4-unsloth",
                            "timestamp": int(time.time()),
                            "analysis_time": analysis_time,
                            "service_used": "phi4",
                            "method": "text_based"
                        }
                    else:
                        self.logger.error(f"[HybridVisionService] Phi-4 API error {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("[HybridVisionService] Phi-4 request timed out")
            return None
        except Exception as e:
            self.logger.error(f"[HybridVisionService] Phi-4 analysis failed: {e}")
            return None
    
    async def _analyze_with_minigpt4(self, screenshot_data: str, game_type: str, fast_mode: bool) -> Optional[Dict[str, Any]]:
        """Analyze with MiniGPT-4 (vision-based analysis)"""
        try:
            # Prepare prompt based on game type
            if game_type.lower() == "everquest":
                if fast_mode:
                    prompt = "Quick EverQuest game state?"
                else:
                    prompt = "Analyze this EverQuest screenshot. Describe the current zone, character activities, NPCs, monsters, and game situation."
            else:
                if fast_mode:
                    prompt = f"Quick {game_type} game state?"
                else:
                    prompt = f"Analyze this {game_type} screenshot. Describe what's happening in the game."
            
            # Prepare request data
            request_data = {
                "image_data": screenshot_data,
                "prompt": prompt,
                "fast_mode": fast_mode
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.minigpt4_url}/analyze",
                    json=request_data,
                    timeout=self.minigpt4_timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_time = time.time() - start_time
                        
                        return {
                            "analysis": result.get("analysis", "No analysis available"),
                            "model": "minigpt4",
                            "timestamp": int(time.time()),
                            "analysis_time": analysis_time,
                            "service_used": "minigpt4",
                            "method": "vision_based",
                            "fast_mode": fast_mode
                        }
                    else:
                        self.logger.error(f"[HybridVisionService] MiniGPT-4 API error {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("[HybridVisionService] MiniGPT-4 request timed out")
            return None
        except Exception as e:
            self.logger.error(f"[HybridVisionService] MiniGPT-4 analysis failed: {e}")
            return None
    
    async def get_fast_analysis(self, screenshot_data: str, game_type: str = "everquest") -> Optional[Dict[str, Any]]:
        """Get fast analysis (prefers Phi-4)"""
        return await self.analyze_screenshot(screenshot_data, game_type, mode="fast")
    
    async def get_detailed_analysis(self, screenshot_data: str, game_type: str = "everquest") -> Optional[Dict[str, Any]]:
        """Get detailed analysis (prefers MiniGPT-4)"""
        return await self.analyze_screenshot(screenshot_data, game_type, mode="detailed", fast_mode=False)
    
    async def get_auto_analysis(self, screenshot_data: str, game_type: str = "everquest") -> Optional[Dict[str, Any]]:
        """Get auto analysis (smart routing)"""
        return await self.analyze_screenshot(screenshot_data, game_type, mode="auto")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "hybrid-vision-service",
            "phi4_available": self.phi4_available,
            "minigpt4_available": self.minigpt4_available,
            "phi4_url": self.phi4_url,
            "minigpt4_url": self.minigpt4_url,
            "recommended_service": "phi4" if self.phi4_available else "minigpt4" if self.minigpt4_available else "none"
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        # No specific cleanup needed for this service
        pass
    
    async def capture_screenshot(self) -> Optional[str]:
        """Capture screenshot from OBS for analysis"""
        try:
            # Import OBS service for screenshot capture
            from services.obs_video_service import OBSVideoService
            
            # Create temporary OBS service for screenshot
            obs_service = OBSVideoService(self.app_context)
            if await obs_service.connect_obs():
                screenshot_data = await obs_service.capture_screenshot()
                obs_service.disconnect_obs()
                return screenshot_data
            else:
                self.logger.error("[HybridVisionService] Failed to connect to OBS for screenshot")
                return None
                
        except Exception as e:
            self.logger.error(f"[HybridVisionService] Screenshot capture failed: {e}")
            return None 