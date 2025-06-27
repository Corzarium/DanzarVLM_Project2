import logging
import os
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator

logger = logging.getLogger("DanzarVLM.ModelClient")

class ModelClient:
    def __init__(self, api_base_url: Optional[str] = None, api_key: Optional[str] = None, app_context=None):
        """
        Initialize the ModelClient with API configuration.
        
        Args:
            api_base_url: Base URL for the LLM API (optional, falls back to app_context or default)
            api_key: API key for authentication (optional)
            app_context: The application context containing configuration and logger (optional)
        """
        if app_context:
            self.ctx = app_context
            self.logger = app_context.logger
        else:
            self.ctx = None
            self.logger = logger
        
        # Use the configured endpoint from app_context if available, or provided URL
        if app_context and hasattr(app_context, 'global_settings'):
            llm_config = app_context.global_settings.get('LLM_SERVER', {})
            self.provider = llm_config.get('provider', 'llamacpp')  # Default to llamacpp server
            base_endpoint = llm_config.get('endpoint', api_base_url or os.getenv("LLM_ENDPOINT", "http://localhost:8080"))
            
            # Set endpoint for llama.cpp server (OpenAI-compatible)
            if not base_endpoint.endswith('/chat/completions'):
                base_url = base_endpoint.rstrip('/')
                if base_url.endswith('/v1'):
                    self.endpoint = f"{base_url}/chat/completions"
                else:
                    self.endpoint = f"{base_url}/v1/chat/completions"
            else:
                self.endpoint = base_endpoint
        else:
            self.provider = 'llamacpp'  # Default to llamacpp server
            self.endpoint = api_base_url or os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
        
        self.api_key = api_key

    async def generate(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """
        Generate method using llama.cpp server's OpenAI-compatible API (async)
        
        Args:
            messages: The input messages for generation
            **kwargs: Additional generation parameters like temperature, max_tokens, model
            
        Returns:
            Generated text response
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[ModelClient] Generating response with temp={kwargs.get('temperature', 0.7)}, max_tokens={kwargs.get('max_tokens', 512)} using llama.cpp server (attempt {attempt + 1}/{max_retries})")
                
                # llama.cpp server uses OpenAI-compatible API
                payload = {
                    "model": kwargs.get("model", "qwen2.5-omni-7b"),
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "stream": False
                }
                
                # Use reasonable timeout for llama.cpp server
                timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", 120))  # 2 minutes default timeout
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.endpoint,
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        
                        # Handle OpenAI-compatible response format
                        response_data = await response.json()
                        
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            message = response_data["choices"][0]["message"]
                            content = message.get("content", "")
                            return content
                        else:
                            self.logger.warning(f"[ModelClient] Unexpected response format: {response_data}")
                            return ""
                        
            except asyncio.TimeoutError as e:
                self.logger.warning(f"[ModelClient] Request timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] All {max_retries} attempts failed due to timeout")
                    raise
                    
            except aiohttp.ClientConnectionError as e:
                self.logger.warning(f"[ModelClient] Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] All {max_retries} attempts failed due to connection error")
                    raise
                    
            except aiohttp.ClientResponseError as e:
                self.logger.error(f"[ModelClient] API request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1 and e.status >= 500:  # Retry on server errors
                    self.logger.info(f"[ModelClient] Server error, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"[ModelClient] Generation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise
        
        # This should never be reached, but just in case
        return None

    async def generate_streaming(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate streaming response using llama.cpp server's OpenAI-compatible API
        
        Args:
            messages: The input messages for generation
            **kwargs: Additional generation parameters like temperature, max_tokens, model
            
        Yields:
            Text chunks as they are generated
        """
        try:
            self.logger.info(f"[ModelClient] Starting streaming generation with temp={kwargs.get('temperature', 0.7)}, max_tokens={kwargs.get('max_tokens', 512)}")
            
            # llama.cpp server uses OpenAI-compatible streaming API
            payload = {
                "model": kwargs.get("model", "qwen2.5-omni-7b"),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
                "stream": True  # Enable streaming
            }
            
            # Use reasonable timeout for llama.cpp server
            timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", 300))  # 5 minutes default timeout for streaming
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.endpoint,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # Process streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Handle SSE format: "data: {...}"
                        if line.startswith('data: '):
                            data = line[6:]  # Remove "data: " prefix
                            
                            # Check for end of stream
                            if data == '[DONE]':
                                self.logger.debug("[ModelClient] Streaming completed")
                                break
                            
                            try:
                                # Parse JSON data
                                import json
                                chunk_data = json.loads(data)
                                
                                # Extract content from chunk
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    choice = chunk_data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            yield content
                                            
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"[ModelClient] Failed to parse streaming chunk: {e}")
                                continue
                            except Exception as e:
                                self.logger.error(f"[ModelClient] Error processing streaming chunk: {e}")
                                continue
                    
        except asyncio.TimeoutError as e:
            self.logger.error(f"[ModelClient] Streaming request timeout: {e}")
            yield f"Sorry, the request timed out. Please try again."
        except aiohttp.ClientConnectionError as e:
            self.logger.error(f"[ModelClient] Streaming connection error: {e}")
            yield f"Sorry, I couldn't connect to the language model. Please check if the server is running."
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ModelClient] Streaming API error: {e}")
            yield f"Sorry, there was an error with the language model server."
        except Exception as e:
            self.logger.error(f"[ModelClient] Streaming generation error: {e}")
            yield f"Sorry, I encountered an unexpected error while generating the response."