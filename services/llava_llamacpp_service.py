"""
LLaVA LlamaCpp Service for DanzarVLM
Direct integration with llama.cpp for vision analysis using LLaVA models
"""

import asyncio
import subprocess
import tempfile
import json
import logging
import os
import base64
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import cv2
import aiohttp
import requests

class LLaVALlamaCppService:
    """Service for LLaVA vision analysis using llama.cpp"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Configuration - Use existing Qwen2.5-VL model which is LLaVA-compatible
        self.model_path = self.config.get('LLAVA_LLAMACPP', {}).get('model_path', 'models-gguf/Qwen2.5-VL-7B-Instruct-q8_0.gguf')
        self.mmproj_path = self.config.get('LLAVA_LLAMACPP', {}).get('mmproj_path', 'models-gguf/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf')
        self.executable_path = self.config.get('LLAVA_LLAMACPP', {}).get('executable_path', 'llama-cpp-cuda/llama-server.exe')
        self.cli_executable = self.config.get('LLAVA_LLAMACPP', {}).get('cli_executable', 'llama-cpp-cuda/llama-llava-cli.exe')
        
        # Performance settings
        self.gpu_layers = self.config.get('LLAVA_LLAMACPP', {}).get('gpu_layers', 99)  # Offload all layers to GPU
        self.threads = self.config.get('LLAVA_LLAMACPP', {}).get('threads', 8)
        self.context_size = self.config.get('LLAVA_LLAMACPP', {}).get('context_size', 4096)
        self.max_tokens = self.config.get('LLAVA_LLAMACPP', {}).get('max_tokens', 512)
        
        # Server mode settings
        self.server_host = self.config.get('LLAVA_LLAMACPP', {}).get('server_host', 'localhost')
        self.server_port = self.config.get('LLAVA_LLAMACPP', {}).get('server_port', 8084)
        self.use_server_mode = self.config.get('LLAVA_LLAMACPP', {}).get('use_server_mode', True)
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.total_images_processed = 0
        self.average_processing_time = 0.0
        
        # Runtime state
        self.server_process: Optional[subprocess.Popen] = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the LLaVA llama.cpp service"""
        try:
            # Verify paths exist
            if not Path(self.model_path).exists():
                self.logger.error(f"❌ LLaVA model not found: {self.model_path}")
                return False
                
            if not Path(self.mmproj_path).exists():
                self.logger.error(f"❌ Multimodal projection file not found: {self.mmproj_path}")
                return False
                
            if not Path(self.executable_path).exists():
                self.logger.error(f"❌ llama-server executable not found: {self.executable_path}")
                return False
                
            if not Path(self.cli_executable).exists():
                self.logger.warning(f"⚠️ LLaVA CLI executable not found: {self.cli_executable}")
                self.logger.info("Will use regular llama-server for vision analysis")
            
            if self.use_server_mode:
                success = await self._start_server()
                if not success:
                    self.logger.warning("⚠️ Server mode failed, falling back to CLI mode")
                    self.use_server_mode = False
            
            self.is_initialized = True
            self.logger.info(f"✅ LLaVA llama.cpp service initialized (server_mode: {self.use_server_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLaVA llama.cpp service: {e}", exc_info=True)
            return False
    
    async def _start_server(self) -> bool:
        """Start llama.cpp server for API-based interaction"""
        try:
            cmd = [
                str(Path(self.executable_path).resolve()),
                '-m', str(Path(self.model_path).resolve()),
                '--mmproj', str(Path(self.mmproj_path).resolve()),
                '--host', '127.0.0.1',  # Force IPv4 to avoid connection issues
                '--port', str(self.server_port),
                '-ngl', str(self.gpu_layers),
                '-t', str(self.threads),
                '-c', str(self.context_size),
                '--chat-template', 'llava',  # Use LLaVA chat template
                '--log-disable'  # Reduce log noise
            ]
            
            self.logger.info(f"🚀 Starting LLaVA llama.cpp server: {' '.join(cmd[:8])}...")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            # Wait for server to start
            await asyncio.sleep(8)
            
            # Test server connection
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://{self.server_host}:{self.server_port}/health', timeout=5) as response:
                        if response.status == 200:
                            self.logger.info(f"✅ LLaVA llama.cpp server started on {self.server_host}:{self.server_port}")
                            return True
            except:
                pass
            
            # Check if process is still running
            if self.server_process.poll() is None:
                self.logger.info(f"✅ LLaVA llama.cpp server process running on {self.server_host}:{self.server_port}")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                self.logger.error(f"❌ LLaVA server failed to start. Stdout: {stdout[:500]}, Stderr: {stderr[:500]}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start LLaVA server: {e}", exc_info=True)
            return False
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 for model input"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if too large (LLaVA works better with reasonable sizes)
            height, width = frame_rgb.shape[:2]
            if width > 1024 or height > 1024:
                scale = min(1024/width, 1024/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Encode as JPEG with high quality
            success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
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
    
    def save_temp_image(self, frame: np.ndarray) -> str:
        """Save frame as temporary image file"""
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Convert BGR to RGB and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if too large
            height, width = frame_rgb.shape[:2]
            if width > 1024 or height > 1024:
                scale = min(1024/width, 1024/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(temp_path, frame_rgb)
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Error saving temp image: {e}")
            return None
    
    async def analyze_screenshot(self, frame: np.ndarray, prompt: str = "Describe this image in detail. What is happening in the scene?") -> str:
        """
        Analyze a screenshot with LLaVA
        
        Args:
            frame: OpenCV frame
            prompt: Text prompt for analysis
            
        Returns:
            Analysis text
        """
        start_time = time.time()
        
        try:
            if self.use_server_mode:
                result = await self._analyze_screenshot_server(frame, prompt)
            else:
                result = await self._analyze_screenshot_cli(frame, prompt)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_images_processed += 1
            self.average_processing_time = self.total_processing_time / self.total_images_processed
            
            self.logger.info(f"LLaVA analysis completed in {processing_time:.2f}s (avg: {self.average_processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ LLaVA analysis error: {e}")
            return f"Error analyzing image: {str(e)}"
    
    async def _analyze_screenshot_server(self, frame: np.ndarray, prompt: str) -> str:
        """Analyze screenshot using server mode"""
        try:
            # Convert frame to base64
            base64_image = self.frame_to_base64(frame)
            if not base64_image:
                return "Error: Failed to encode image"
            
            # Prepare request payload for llama.cpp server
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
                "stream": False
            }
            
            # Make request to llama.cpp server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://{self.server_host}:{self.server_port}/v1/chat/completions',
                    json=payload,
                    timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        return content.strip() if content else "No analysis generated"
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Server error {response.status}: {error_text}")
                        return f"Server error: {response.status}"
                        
        except Exception as e:
            self.logger.error(f"Server analysis error: {e}")
            return f"Server analysis failed: {str(e)}"
    
    async def _analyze_screenshot_cli(self, frame: np.ndarray, prompt: str) -> str:
        """Analyze screenshot using CLI mode"""
        temp_path = None
        try:
            # Save frame to temporary file
            temp_path = self.save_temp_image(frame)
            if not temp_path:
                return "Error: Failed to save temporary image"
            
            # Prepare CLI command
            if Path(self.cli_executable).exists():
                # Use dedicated LLaVA CLI
                cmd = [
                    str(Path(self.cli_executable).resolve()),
                    '-m', str(Path(self.model_path).resolve()),
                    '--mmproj', str(Path(self.mmproj_path).resolve()),
                    '--image', temp_path,
                    '--prompt', prompt,
                    '-ngl', str(self.gpu_layers),
                    '-t', str(self.threads),
                    '-c', str(self.context_size),
                    '--temp', '0.1',
                    '--top-p', '0.9'
                ]
            else:
                # Use regular llama-server CLI mode
                cmd = [
                    str(Path('llama-cpp-cuda/llama-cli.exe').resolve()),
                    '-m', str(Path(self.model_path).resolve()),
                    '--mmproj', str(Path(self.mmproj_path).resolve()),
                    '--image', temp_path,
                    '-p', f"<image>\n{prompt}",
                    '-ngl', str(self.gpu_layers),
                    '-t', str(self.threads),
                    '-c', str(self.context_size),
                    '--temp', '0.1'
                ]
            
            self.logger.info(f"🚀 Running LLaVA CLI: {Path(cmd[0]).name} with image analysis")
            
            # Run the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            
            if process.returncode == 0:
                result = stdout.decode('utf-8', errors='ignore').strip()
                # Clean up the output (remove prompt echo and extra whitespace)
                lines = result.split('\n')
                # Find where the actual response starts (after the prompt)
                response_lines = []
                found_response = False
                for line in lines:
                    if not found_response and (prompt.lower() in line.lower() or 'describe' in line.lower()):
                        found_response = True
                        continue
                    if found_response and line.strip():
                        response_lines.append(line.strip())
                
                if response_lines:
                    return '\n'.join(response_lines)
                else:
                    return result if result else "No analysis generated"
            else:
                error_msg = stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"CLI analysis failed: {error_msg}")
                return f"CLI analysis failed: {error_msg[:200]}"
                
        except asyncio.TimeoutError:
            self.logger.error("CLI analysis timed out")
            return "Analysis timed out"
        except Exception as e:
            self.logger.error(f"CLI analysis error: {e}")
            return f"CLI analysis failed: {str(e)}"
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    async def analyze_multiple_frames(self, frames: List[np.ndarray], prompt: str = "Describe what's happening in these game scenes.") -> List[str]:
        """Analyze multiple frames"""
        results = []
        for i, frame in enumerate(frames):
            frame_prompt = f"{prompt} (Frame {i+1}/{len(frames)})"
            result = await self.analyze_screenshot(frame, frame_prompt)
            results.append(result)
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_processing_time': self.total_processing_time,
            'total_images_processed': self.total_images_processed,
            'average_processing_time': self.average_processing_time,
            'images_per_minute': (60.0 / self.average_processing_time) if self.average_processing_time > 0 else 0
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.server_process and self.server_process.poll() is None:
                self.logger.info("Stopping LLaVA llama.cpp server...")
                self.server_process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(self.server_process), timeout=10)
                except asyncio.TimeoutError:
                    self.logger.warning("Server didn't stop gracefully, forcing kill...")
                    self.server_process.kill()
                    await self._wait_for_process(self.server_process)
                self.logger.info("✅ LLaVA server stopped")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def _wait_for_process(self, process):
        """Wait for process to finish"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    def is_available(self) -> bool:
        """Check if service is available"""
        if not self.is_initialized:
            return False
            
        if not self.use_server_mode:
            return True
            
        # First check if server is reachable (most reliable)
        try:
            response = requests.get(f'http://{self.server_host}:{self.server_port}/health', timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
            
        # If server is not reachable and we have a process reference, check if it's running
        if self.server_process:
            return self.server_process.poll() is None
            
        # No server reachable and no valid process
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'initialized': self.is_initialized,
            'server_mode': self.use_server_mode,
            'server_running': self.server_process.poll() is None if self.server_process else False,
            'model_path': self.model_path,
            'mmproj_path': self.mmproj_path,
            'performance_metrics': self.get_performance_metrics()
        } 