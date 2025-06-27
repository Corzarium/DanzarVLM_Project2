#!/usr/bin/env python3
"""
Fast Vision Client Service - Client for external Fast Vision Server
Connects to standalone BLIP/CLIP server for image analysis
"""

import asyncio
import base64
import logging
import time
from typing import Dict, List, Optional
import aiohttp

class FastVisionClientService:
    """Client service for external Fast Vision Server"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Server configuration
        video_config = self.config.get('VIDEO_ANALYSIS', {})
        self.server_url = video_config.get('fast_model_endpoint', 'http://localhost:8082')
        self.timeout = video_config.get('timeout', 30)
        
        # Performance stats
        self.stats = {
            "total_analyses": 0,
            "avg_total_time": 0.0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        
        self.logger.info(f"[FastVisionClient] Initialized as client for {self.server_url}")
    
    async def initialize(self) -> bool:
        """Test connection to Fast Vision Server"""
        try:
            self.logger.info(f"[FastVisionClient] Testing connection to {self.server_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/status", timeout=10) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        self.logger.info(f"[FastVisionClient] Connected to server: {status_data}")
                        return True
                    else:
                        self.logger.error(f"[FastVisionClient] Server returned status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"[FastVisionClient] Failed to connect to server: {e}")
            self.logger.info("[FastVisionClient] Make sure Fast Vision Server is running on the configured endpoint")
            return False
    
    async def analyze_screenshot(self, image_data: str) -> Dict:
        """
        Analyze screenshot using external Fast Vision Server
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Dict with caption, scene_type, confidence, and timing
        """
        start_time = time.time()
        
        try:
            # Prepare request
            request_data = {
                "image_data": image_data,
                "prompt": "Describe this gaming screenshot in detail."
            }
            
            # Send request to server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/analyze",
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        total_time = time.time() - start_time
                        
                        # Update stats
                        self.stats["total_analyses"] += 1
                        self.stats["successful_requests"] += 1
                        self.stats["avg_total_time"] = (
                            (self.stats["avg_total_time"] * (self.stats["total_analyses"] - 1) + total_time) 
                            / self.stats["total_analyses"]
                        )
                        
                        self.logger.info(f"[FastVisionClient] Analysis completed in {total_time:.3f}s - {result.get('scene_type', 'unknown')}: {result.get('caption', 'no caption')}")
                        
                        return {
                            "caption": result.get("caption", "No caption generated"),
                            "scene_type": result.get("scene_type", "unknown"),
                            "scene_confidence": result.get("scene_confidence", 0.0),
                            "analysis_time": result.get("analysis_time", total_time),
                            "timestamp": result.get("timestamp", time.time())
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"[FastVisionClient] Server error {response.status}: {error_text}")
                        self.stats["failed_requests"] += 1
                        
                        return {
                            "caption": "Server error occurred",
                            "scene_type": "unknown",
                            "scene_confidence": 0.0,
                            "analysis_time": time.time() - start_time,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            self.logger.error(f"[FastVisionClient] Request timed out after {self.timeout}s")
            self.stats["failed_requests"] += 1
            return {
                "caption": "Request timed out",
                "scene_type": "unknown",
                "scene_confidence": 0.0,
                "analysis_time": time.time() - start_time,
                "error": "Request timeout"
            }
        except Exception as e:
            self.logger.error(f"[FastVisionClient] Analysis failed: {e}")
            self.stats["failed_requests"] += 1
            return {
                "caption": "Unable to analyze screenshot",
                "scene_type": "unknown",
                "scene_confidence": 0.0,
                "analysis_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def batch_analyze(self, image_list: List[str]) -> List[Dict]:
        """
        Analyze multiple images in parallel
        
        Args:
            image_list: List of base64 encoded images
            
        Returns:
            List of analysis results
        """
        tasks = [self.analyze_screenshot(image_data) for image_data in image_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"[FastVisionClient] Batch analysis failed for image {i}: {result}")
                processed_results.append({
                    "caption": "Analysis failed",
                    "scene_type": "unknown",
                    "scene_confidence": 0.0,
                    "analysis_time": 0.0,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "service": "FastVisionClient",
            "server_url": self.server_url,
            "stats": self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_analyses"]) * 100
            ) if self.stats["total_analyses"] > 0 else 0
        }
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.stats["total_analyses"] > 0 and self.stats["failed_requests"] < self.stats["total_analyses"] 