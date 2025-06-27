"""
LlamaCpp Qwen2.5-VL Service for DanzarVLM
Handles vision-language processing using llama.cpp with GGUF models
"""

import asyncio
import subprocess
import tempfile
import json
import logging
import os
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import cv2
import aiohttp

class LlamaCppQwenVLService:
    """Service for interfacing with llama.cpp GGUF Qwen2.5-VL model"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Configuration
        self.model_path = self.config.get('LLAMACPP_QWEN_VL', {}).get('model_path', 'models-gguf/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf')
        self.mmproj_path = self.config.get('LLAMACPP_QWEN_VL', {}).get('mmproj_path', 'models-gguf/Qwen2.5-VL-7B-Instruct-mmproj.gguf')
        self.executable_path = self.config.get('LLAMACPP_QWEN_VL', {}).get('executable_path', 'llama-cpp-cuda/llama-cli.exe')
        self.server_executable = self.config.get('LLAMACPP_QWEN_VL', {}).get('server_executable', 'llama-cpp-cuda/llama-server.exe')
        
        # Performance settings
        self.gpu_layers = self.config.get('LLAMACPP_QWEN_VL', {}).get('gpu_layers', 99)  # Offload all layers to GPU
        self.threads = self.config.get('LLAMACPP_QWEN_VL', {}).get('threads', 8)
        self.context_size = self.config.get('LLAMACPP_QWEN_VL', {}).get('context_size', 4096)
        self.max_tokens = self.config.get('LLAMACPP_QWEN_VL', {}).get('max_tokens', 512)
        
        # Server mode settings
        self.server_host = self.config.get('LLAMACPP_QWEN_VL', {}).get('server_host', 'localhost')
        self.server_port = self.config.get('LLAMACPP_QWEN_VL', {}).get('server_port', 8083)
        self.use_server_mode = self.config.get('LLAMACPP_QWEN_VL', {}).get('use_server_mode', True)
        
        # Runtime state
        self.server_process: Optional[subprocess.Popen] = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the llama.cpp service"""
        try:
            # Verify paths exist
            if not Path(self.model_path).exists():
                self.logger.error(f"❌ Model not found: {self.model_path}")
                return False
                
            if not Path(self.mmproj_path).exists():
                self.logger.warning(f"⚠️ Multimodal projection file not found: {self.mmproj_path}")
                self.logger.warning("⚠️ Vision capabilities will be limited or unavailable")
                
            if not Path(self.executable_path).exists():
                self.logger.error(f"❌ Executable not found: {self.executable_path}")
                return False
            
            if self.use_server_mode:
                success = await self._start_server()
                if not success:
                    self.logger.warning("⚠️ Server mode failed, falling back to CLI mode")
                    self.use_server_mode = False
            
            self.is_initialized = True
            self.logger.info(f"✅ LlamaCpp Qwen VL service initialized (server_mode: {self.use_server_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LlamaCpp Qwen VL service: {e}", exc_info=True)
            return False
    
    async def _start_server(self) -> bool:
        """Start llama.cpp server for API-based interaction"""
        try:
            cmd = [
                str(Path(self.server_executable).resolve()),
                '-m', str(Path(self.model_path).resolve()),
                '--host', '127.0.0.1',  # Force IPv4 to avoid connection issues
                '--port', str(self.server_port),
                '-ngl', str(self.gpu_layers),
                '-t', str(self.threads),
                '-c', str(self.context_size),
                '--jinja',  # Use embedded chat template
                '--log-disable'  # Reduce log noise
            ]
            
            # Add mmproj if available
            if Path(self.mmproj_path).exists():
                cmd.extend(['--mmproj', str(Path(self.mmproj_path).resolve())])
            
            self.logger.info(f"🚀 Starting llama.cpp server: {' '.join(cmd)}")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            # Wait for server to start and model to load
            self.logger.info("⏳ Waiting for llama.cpp server to start and model to load...")
            
            # Check server startup with exponential backoff
            max_wait_time = 120  # 2 minutes max wait
            wait_time = 2
            total_wait = 0
            
            while total_wait < max_wait_time:
                await asyncio.sleep(wait_time)
                total_wait += wait_time
                
                # Check if process is still running
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    self.logger.error(f"❌ Server process died. Stdout: {stdout}, Stderr: {stderr}")
                    return False
                
                # Try to connect to server
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://127.0.0.1:{self.server_port}/v1/models", 
                                             timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                self.logger.info(f"✅ llama.cpp server started and ready on {self.server_host}:{self.server_port}")
                                self.logger.info(f"⏱️  Total startup time: {total_wait}s")
                                return True
                except Exception as e:
                    if total_wait < 30:  # First 30 seconds, log less frequently
                        if total_wait % 10 == 0:
                            self.logger.info(f"⏳ Still waiting for server... ({total_wait}s)")
                    else:
                        self.logger.info(f"⏳ Server not ready yet... ({total_wait}s)")
                
                # Exponential backoff
                wait_time = min(wait_time * 1.5, 10)
            
            # If we get here, server didn't start in time
            self.logger.error(f"❌ Server failed to start within {max_wait_time}s")
            if self.server_process.poll() is None:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
            return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}", exc_info=True)
            return False
    
    def frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
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
            
            # Add data URL prefix
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            self.logger.error(f"Error converting frame to base64: {e}")
            return None
    
    def save_debug_frame(self, frame: np.ndarray, prefix: str = "debug") -> Optional[str]:
        """Save frame for debugging purposes"""
        try:
            debug_dir = self.config.get('DEBUG_OUTPUT_PATH', './debug_frames')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(debug_dir, filename)
            
            cv2.imwrite(filepath, frame)
            self.logger.debug(f"Saved debug frame to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving debug frame: {e}")
            return None
    
    async def analyze_frame(self, frame: np.ndarray, prompt: str = "What's happening in this game?") -> str:
        """
        Analyze a video frame with Qwen2.5-VL
        
        Args:
            frame: OpenCV frame
            prompt: Text prompt for analysis
            
        Returns:
            Analysis text
        """
        try:
            # Save frame to disk for llama.cpp compatibility
            temp_path = self.save_debug_frame(frame, "analysis")
            if not temp_path:
                return "Error: Failed to save frame for analysis"
            
            # Analyze the image
            return await self.analyze_image(temp_path, prompt)
            
        except Exception as e:
            self.logger.error(f"❌ Frame analysis error: {e}")
            return f"Error analyzing frame: {str(e)}"
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Asynchronous image analysis
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis
            
        Returns:
            Analysis text
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if self.use_server_mode:
                return await self._analyze_image_server(image_path, prompt)
            else:
                return await self._analyze_image_cli(image_path, prompt)
                
        except Exception as e:
            self.logger.error(f"❌ Image analysis failed: {e}", exc_info=True)
            return "I encountered an error analyzing the image. Please try again."
    
    async def _analyze_image_server(self, image_path: str, prompt: str) -> str:
        """Analyze image using server API"""
        try:
            # Check if image is a base64 string or URL
            if image_path.startswith('data:image') or image_path.startswith('http'):
                image_data = image_path
            else:
                # For local file, read and convert to base64
                if not os.path.exists(image_path):
                    return f"Error: Image file not found: {image_path}"
                
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    image_data = f"data:image/jpeg;base64,{image_b64}"
            
            # Create messages with image
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can analyze images."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            # Use 127.0.0.1 instead of localhost to force IPv4
            server_url = f"http://127.0.0.1:{self.server_port}/v1/chat/completions"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    server_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout for image processing
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        self.logger.info(f"✅ Generated response: {content[:100]}...")
                        return content.strip()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Server error: {response.status}, {error_text}")
                        return f"Error: Server returned {response.status}"
                        
        except Exception as e:
            self.logger.error(f"❌ Server image analysis error: {e}", exc_info=True)
            return f"Error analyzing image with server: {str(e)}"
    
    async def _analyze_image_cli(self, image_path: str, prompt: str) -> str:
        """Analyze image using CLI mode"""
        try:
            # For CLI mode, we need the actual file path
            if image_path.startswith('data:image'):
                # Save base64 to temp file
                try:
                    image_data = image_path.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_file.write(image_bytes)
                        image_path = temp_file.name
                except Exception as e:
                    self.logger.error(f"❌ Failed to save base64 image: {e}")
                    return "Error: Failed to process base64 image"
            
            elif image_path.startswith('http'):
                self.logger.error("❌ URL images not supported in CLI mode")
                return "Error: URL images not supported in CLI mode"
            
            # Verify image exists
            if not os.path.exists(image_path):
                self.logger.error(f"❌ Image file not found: {image_path}")
                return f"Error: Image file not found: {image_path}"
            
            # Build command
            cmd = [
                str(Path(self.executable_path).resolve()),
                '-m', str(Path(self.model_path).resolve()),
                '-ngl', str(self.gpu_layers),
                '-t', str(self.threads),
                '-c', str(self.context_size),
                '-n', str(self.max_tokens),
                '--temp', '0.7',
                '--jinja',  # Use embedded chat template
                '--image', image_path,  # Image path
            ]
            
            # Add mmproj if available
            if Path(self.mmproj_path).exists():
                cmd.extend(['--mmproj', str(Path(self.mmproj_path).resolve())])
            
            # Create prompt
            system_prompt = "You are a helpful assistant that can analyze images."
            user_prompt = f"{prompt}"
            
            # Format as JSON for chat mode
            chat_data = json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            cmd.extend(['--prompt-cache', 'none', '--prompt', chat_data])
            
            self.logger.info(f"🚀 Running llama.cpp CLI: {' '.join(cmd[:10])}...")
            
            # Run command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                
                if process.returncode != 0:
                    self.logger.error(f"❌ CLI command failed: {stderr}")
                    return f"Error: CLI command failed with code {process.returncode}"
                
                # Extract response from output
                response = stdout.strip()
                
                # Clean up temporary file if needed
                if image_path.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                
                return response
                
            except asyncio.TimeoutError:
                self.logger.error("❌ CLI command timed out")
                process.kill()
                return "Error: Analysis timed out"
                
        except Exception as e:
            self.logger.error(f"❌ CLI image analysis error: {e}", exc_info=True)
            return f"Error analyzing image with CLI: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.server_process:
                self.logger.info("Stopping llama.cpp server...")
                self.server_process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(self.server_process), timeout=5)
                except asyncio.TimeoutError:
                    self.logger.warning("Server didn't terminate gracefully, forcing...")
                    self.server_process.kill()
                
                self.server_process = None
                
            self.is_initialized = False
            self.logger.info("✅ LlamaCpp Qwen VL service cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    async def _wait_for_process(self, process):
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.is_initialized and (
            (self.use_server_mode and self.server_process and self.server_process.poll() is None) or
            (not self.use_server_mode)
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "LlamaCppQwenVLService",
            "model": os.path.basename(self.model_path),
            "mmproj": os.path.basename(self.mmproj_path) if Path(self.mmproj_path).exists() else "Not found",
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "server_mode": self.use_server_mode,
            "server_port": self.server_port if self.use_server_mode else None
        }