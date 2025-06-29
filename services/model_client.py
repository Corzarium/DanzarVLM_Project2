import logging
import os
import aiohttp
import asyncio
import json
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

    def _process_multimodal_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Process messages to handle multimodal content (images) for VLM.
        Converts text with embedded base64 images to proper multimodal format.
        """
        processed_messages = []
        
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "user")
            
            # Check if content contains image data
            if isinstance(content, str) and "<image>" in content and "</image>" in content:
                self.logger.info(f"[ModelClient] 🔍 Found image data in {role} message")
                self.logger.info(f"[ModelClient] 📊 Content length: {len(content)} chars")
                
                # Extract image data and text
                parts = content.split("<image>")
                if len(parts) == 2:
                    text_before = parts[0].strip()
                    image_part = parts[1].split("</image>")
                    if len(image_part) == 2:
                        image_data = image_part[0].strip()
                        text_after = image_part[1].strip()
                        
                        self.logger.info(f"[ModelClient] 📝 Text before image: {len(text_before)} chars")
                        self.logger.info(f"[ModelClient] 🖼️ Image data length: {len(image_data)} chars")
                        self.logger.info(f"[ModelClient] 📝 Text after image: {len(text_after)} chars")
                        
                        # Create multimodal content
                        multimodal_content = []
                        
                        # Add text before image
                        if text_before:
                            multimodal_content.append({"type": "text", "text": text_before})
                        
                        # Add image if it's valid base64
                        if image_data and image_data != "No screenshot available" and len(image_data) > 100:
                            try:
                                # Validate base64
                                import base64
                                decoded_data = base64.b64decode(image_data)
                                self.logger.info(f"[ModelClient] ✅ Valid base64 image data: {len(decoded_data)} bytes")
                                
                                multimodal_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                })
                                self.logger.info(f"[ModelClient] ✅ Added image to multimodal content ({len(image_data)} chars)")
                                self.logger.info(f"[ModelClient] 📊 Multimodal content now has {len(multimodal_content)} parts")
                            except Exception as e:
                                self.logger.warning(f"[ModelClient] ❌ Invalid base64 image data: {e}")
                                self.logger.warning(f"[ModelClient] 📊 Image data preview: {image_data[:100]}...")
                        else:
                            self.logger.warning(f"[ModelClient] ⚠️ Skipping image data: length={len(image_data)}, valid={image_data != 'No screenshot available'}")
                        
                        # Add text after image
                        if text_after:
                            multimodal_content.append({"type": "text", "text": text_after})
                        
                        # Create new message with multimodal content
                        processed_message = {
                            "role": role,
                            "content": multimodal_content
                        }
                        processed_messages.append(processed_message)
                        self.logger.info(f"[ModelClient] ✅ Created multimodal message with {len(multimodal_content)} content parts")
                        continue
                    else:
                        self.logger.warning(f"[ModelClient] ❌ Malformed image tag structure")
                else:
                    self.logger.warning(f"[ModelClient] ❌ Could not split content on <image> tag")
            
            # If no image data, keep original message
            processed_messages.append(message)
        
        self.logger.info(f"[ModelClient] 📊 Processed {len(messages)} messages into {len(processed_messages)} messages")
        return processed_messages

    async def generate(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs) -> Optional[str]:
        """
        Generate method using llama.cpp server's OpenAI-compatible API (async)
        
        Args:
            messages: The input messages for generation
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional generation parameters like temperature, max_tokens, model
            
        Returns:
            Generated text response
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[ModelClient] Generating response with temp={kwargs.get('temperature', 0.7)}, max_tokens={kwargs.get('max_tokens', 512)} using llama.cpp server (attempt {attempt + 1}/{max_retries})")
                
                # Process messages to handle multimodal content (images)
                processed_messages = self._process_multimodal_messages(messages)
                
                # Log the processed messages structure
                for i, msg in enumerate(processed_messages):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        self.logger.info(f"[ModelClient] 📝 Message {i} ({msg.get('role', 'unknown')}): {len(content)} content parts")
                        for j, part in enumerate(content):
                            part_type = part.get("type", "unknown")
                            if part_type == "image_url":
                                self.logger.info(f"[ModelClient] 🖼️ Part {j}: {part_type} (image data included)")
                            else:
                                text_len = len(part.get("text", ""))
                                self.logger.info(f"[ModelClient] 📝 Part {j}: {part_type} ({text_len} chars)")
                    else:
                        text_len = len(str(content))
                        self.logger.info(f"[ModelClient] 📝 Message {i} ({msg.get('role', 'unknown')}): text only ({text_len} chars)")
                
                # llama.cpp server uses OpenAI-compatible API
                payload = {
                    "model": kwargs.get("model", "Qwen2.5-VL-7B-Instruct"),
                    "messages": processed_messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "stream": False
                }
                
                # Add tools if provided (for function calling)
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"  # Let the model decide when to use tools
                    self.logger.info(f"[ModelClient] 🛠️ Including {len(tools)} tools for function calling")
                    for tool in tools:
                        self.logger.info(f"[ModelClient] 🛠️ Tool: {tool.get('function', {}).get('name', 'unknown')}")
                
                self.logger.info(f"[ModelClient] 🌐 Sending request to: {self.endpoint}")
                self.logger.info(f"[ModelClient] 📦 Payload size: {len(str(payload))} chars")
                
                # Use reasonable timeout for llama.cpp server
                timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", 120))  # 2 minutes default timeout
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.endpoint,
                        json=payload
                    ) as response:
                        self.logger.info(f"[ModelClient] 📡 Response status: {response.status}")
                        
                        response.raise_for_status()
                        
                        # Handle OpenAI-compatible response format
                        response_data = await response.json()
                        
                        self.logger.info(f"[ModelClient] 📥 Response received: {len(str(response_data))} chars")
                        
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            choice = response_data["choices"][0]
                            message = choice["message"]
                            
                            # Check if the model wants to call a tool
                            if "tool_calls" in message and message["tool_calls"]:
                                self.logger.info(f"[ModelClient] 🛠️ Model requested tool calls: {len(message['tool_calls'])}")
                                for tool_call in message["tool_calls"]:
                                    self.logger.info(f"[ModelClient] 🛠️ Tool call: {tool_call.get('function', {}).get('name', 'unknown')}")
                                # Return the tool calls for processing
                                return {
                                    "type": "tool_calls",
                                    "tool_calls": message["tool_calls"]
                                }
                            
                            content = message.get("content", "")
                            self.logger.info(f"[ModelClient] ✅ Generated response: {len(content)} chars")
                            self.logger.info(f"[ModelClient] 📝 Response preview: {content[:100]}...")
                            return content
                        else:
                            self.logger.warning(f"[ModelClient] ❌ Unexpected response format: {response_data}")
                            return ""
                        
            except asyncio.TimeoutError as e:
                self.logger.warning(f"[ModelClient] ⏰ Request timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] 🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] ❌ All {max_retries} attempts failed due to timeout")
                    raise
                    
            except aiohttp.ClientConnectionError as e:
                self.logger.warning(f"[ModelClient] 🔌 Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] 🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] ❌ All {max_retries} attempts failed due to connection error")
                    raise
                    
            except aiohttp.ClientResponseError as e:
                self.logger.error(f"[ModelClient] ❌ API request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1 and e.status >= 500:  # Retry on server errors
                    self.logger.info(f"[ModelClient] 🔄 Server error, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"[ModelClient] ❌ Generation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] 🔄 Retrying in {retry_delay} seconds...")
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
                "model": kwargs.get("model", "Qwen2.5-VL-7B-Instruct"),
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