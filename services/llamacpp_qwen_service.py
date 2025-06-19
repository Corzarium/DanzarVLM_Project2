"""
LlamaCpp Qwen2.5-Omni Service for DanzarVLM
Handles audio transcription and text generation using llama.cpp with GGUF models
"""

import asyncio
import subprocess
import tempfile
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

from core.app_context import AppContext


class LlamaCppQwenService:
    """Service for interfacing with llama.cpp GGUF Qwen2.5-Omni model"""
    
    def __init__(self, app_context: AppContext):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Configuration
        self.model_path = self.config.get('LLAMACPP_QWEN', {}).get('model_path', 'models-gguf/Qwen2.5-Omni-7B-Q4_K_M.gguf')
        self.executable_path = self.config.get('LLAMACPP_QWEN', {}).get('executable_path', 'llama-cpp-cuda/llama-cli.exe')
        self.server_executable = self.config.get('LLAMACPP_QWEN', {}).get('server_executable', 'llama-cpp-cuda/llama-server.exe')
        
        # Performance settings
        self.gpu_layers = self.config.get('LLAMACPP_QWEN', {}).get('gpu_layers', 99)  # Offload all layers to GPU
        self.threads = self.config.get('LLAMACPP_QWEN', {}).get('threads', 8)
        self.context_size = self.config.get('LLAMACPP_QWEN', {}).get('context_size', 4096)
        self.max_tokens = self.config.get('LLAMACPP_QWEN', {}).get('max_tokens', 512)
        
        # Server mode settings
        self.server_host = self.config.get('LLAMACPP_QWEN', {}).get('server_host', 'localhost')
        self.server_port = self.config.get('LLAMACPP_QWEN', {}).get('server_port', 8081)
        self.use_server_mode = self.config.get('LLAMACPP_QWEN', {}).get('use_server_mode', True)
        
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
                
            if not Path(self.executable_path).exists():
                self.logger.error(f"❌ Executable not found: {self.executable_path}")
                return False
            
            if self.use_server_mode:
                success = await self._start_server()
                if not success:
                    self.logger.warning("⚠️ Server mode failed, falling back to CLI mode")
                    self.use_server_mode = False
            
            self.is_initialized = True
            self.logger.info(f"✅ LlamaCpp Qwen service initialized (server_mode: {self.use_server_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LlamaCpp Qwen service: {e}", exc_info=True)
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
            
            self.logger.info(f"🚀 Starting llama.cpp server: {' '.join(cmd)}")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            # Wait a bit for server to start
            await asyncio.sleep(5)
            
            # Check if server is running
            if self.server_process.poll() is None:
                self.logger.info(f"✅ llama.cpp server started on {self.server_host}:{self.server_port}")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                self.logger.error(f"❌ Server failed to start. Stdout: {stdout}, Stderr: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}", exc_info=True)
            return False
    
    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Qwen2.5-Omni
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # For now, use CLI mode for audio transcription
            # Server mode audio support needs more investigation
            return await self._transcribe_audio_cli(audio_path)
            
        except Exception as e:
            self.logger.error(f"❌ Audio transcription failed: {e}", exc_info=True)
            return "I encountered an error processing your audio request. Please try again."
    
    async def _transcribe_audio_cli(self, audio_path: str) -> str:
        """Transcribe audio using CLI mode - Currently not supported by GGUF Qwen2.5-Omni"""
        try:
            # The GGUF version of Qwen2.5-Omni doesn't support audio input through llama.cpp CLI
            # This is a limitation of the current GGUF implementation
            self.logger.warning("🎵 Audio transcription not supported by GGUF Qwen2.5-Omni model")
            return "I can hear you speaking, but audio transcription is not available with this model version."
                
        except Exception as e:
            self.logger.error(f"❌ CLI transcription error: {e}", exc_info=True)
            return "I encountered an error during audio transcription."
    
    async def generate(self, prompt: Optional[str] = None, system_prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs) -> str:
        """Generate method for LLM service compatibility with OpenAI-style interface"""
        try:
            self.logger.debug(f"🔧 Generate called with: prompt={bool(prompt)}, messages={bool(messages)}, kwargs={list(kwargs.keys())}")
            
            # Handle both simple prompt and OpenAI-style messages
            if messages:
                # Convert OpenAI-style messages to simple prompt
                system_msg = None
                user_msg = None
                
                for msg in messages:
                    if msg.get("role") == "system":
                        system_msg = msg.get("content", "")
                    elif msg.get("role") == "user":
                        user_msg = msg.get("content", "")
                
                # Use the converted messages
                if user_msg:
                    self.logger.debug(f"🔧 Using messages interface: user='{user_msg[:50]}...', system='{system_msg[:50] if system_msg else None}...'")
                    return await self.generate_response(user_msg, system_msg)
                else:
                    self.logger.warning("⚠️ No user message found in messages array")
                    return "I didn't receive a clear message to respond to."
            
            elif prompt:
                # Use simple prompt interface
                self.logger.debug(f"🔧 Using prompt interface: '{prompt[:50]}...'")
                return await self.generate_response(prompt, system_prompt)
            
            else:
                self.logger.error("❌ No prompt or messages provided to generate method")
                self.logger.error(f"❌ Available kwargs: {kwargs}")
                return "I didn't receive any input to process."
                
        except Exception as e:
            self.logger.error(f"❌ Generate method error: {e}", exc_info=True)
            return "I encountered an error processing your request."
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text response using Qwen2.5-Omni
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if self.use_server_mode:
                return await self._generate_response_server(prompt, system_prompt)
            else:
                return await self._generate_response_cli(prompt, system_prompt)
                
        except Exception as e:
            self.logger.error(f"❌ Response generation failed: {e}")
            
            # Try fallback to CLI mode if server mode fails
            if self.use_server_mode:
                self.logger.warning("⚠️ Server mode failed, trying CLI mode fallback...")
                try:
                    return await self._generate_response_cli(prompt, system_prompt)
                except Exception as cli_error:
                    self.logger.error(f"❌ CLI fallback also failed: {cli_error}")
            
            return "I encountered an error generating a response. Please try again."
    
    async def _generate_response_server(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using server API"""
        try:
            import aiohttp
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
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
                    timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout for model loading
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        self.logger.info(f"✅ Generated response: {content[:100]}...")
                        return content.strip()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Server API error {response.status}: {error_text}")
                        return "I encountered an error generating a response."
                        
        except Exception as e:
            self.logger.error(f"❌ Server response generation error: {e}", exc_info=True)
            return "I encountered an error generating a response."
    
    async def _generate_response_cli(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using CLI mode"""
        try:
            # Build command
            cmd = [
                str(Path(self.executable_path).resolve()),
                '-m', str(Path(self.model_path).resolve()),
                '-n', str(self.max_tokens),
                '-ngl', str(self.gpu_layers),
                '-t', str(self.threads),
                '-c', str(self.context_size),
                '--temp', '0.7',
                '--log-disable'
            ]
            
            if system_prompt:
                cmd.extend(['-sys', system_prompt])
            
            cmd.extend(['-p', prompt])
            
            self.logger.info(f"🤖 Generating response with llama.cpp CLI")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                response = stdout.decode('utf-8', errors='ignore').strip()
                # Extract the actual response
                lines = response.split('\n')
                response_lines = []
                capture = False
                
                for line in lines:
                    if 'assistant' in line.lower() or capture:
                        capture = True
                        if line.strip() and not line.startswith('main:') and not line.startswith('system_info:'):
                            response_lines.append(line.strip())
                
                generated_text = ' '.join(response_lines).strip()
                if generated_text:
                    self.logger.info(f"✅ Generated response: {generated_text[:100]}...")
                    return generated_text
                else:
                    self.logger.warning("⚠️ Empty generation response")
                    return "I'm having trouble generating a response right now."
            else:
                self.logger.error(f"❌ CLI generation failed: {stderr.decode()}")
                return "Response generation failed."
                
        except Exception as e:
            self.logger.error(f"❌ CLI generation error: {e}", exc_info=True)
            return "I encountered an error during response generation."
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.server_process and self.server_process.poll() is None:
                self.logger.info("🛑 Stopping llama.cpp server...")
                self.server_process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(self.server_process), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ Server didn't stop gracefully, killing...")
                    self.server_process.kill()
                
                self.server_process = None
                
            self.is_initialized = False
            self.logger.info("✅ LlamaCpp Qwen service cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}", exc_info=True)
    
    async def _wait_for_process(self, process):
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1) 