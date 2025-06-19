import logging
import os
import requests
from typing import Optional, Dict, Any, List

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
            self.provider = llm_config.get('provider', 'openai')  # Default to openai (LM Studio)
            base_endpoint = llm_config.get('endpoint', api_base_url or os.getenv("LLM_ENDPOINT", "http://localhost:1234"))
            
            # Set endpoint based on provider
            if self.provider == 'openai':
                # OpenAI-compatible API (LM Studio)
                if not base_endpoint.endswith('/chat/completions'):
                    base_url = base_endpoint.rstrip('/')
                    if base_url.endswith('/v1'):
                        self.endpoint = f"{base_url}/chat/completions"
                    else:
                        self.endpoint = f"{base_url}/v1/chat/completions"
                else:
                    self.endpoint = base_endpoint
            else:
                # Ollama API
                if base_endpoint.endswith('/api/chat'):
                    base_endpoint = base_endpoint.replace('/api/chat', '/api/generate')
                elif not base_endpoint.endswith('/api/generate'):
                    base_url = base_endpoint.rstrip('/')
                    if not base_url.endswith('/api'):
                        base_endpoint = f"{base_url}/api/generate"
                self.endpoint = base_endpoint
        else:
            self.provider = 'openai'  # Default to openai (LM Studio)
            self.endpoint = api_base_url or os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        
        self.api_key = api_key

    def generate(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """
        Unified generate method supporting both OpenAI and Ollama APIs
        
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
                self.logger.info(f"[ModelClient] Generating response with temp={kwargs.get('temperature', 0.7)}, max_tokens={kwargs.get('max_tokens', 4096)} using {self.provider} (attempt {attempt + 1}/{max_retries})")
                
                if self.provider == 'openai':
                    # OpenAI-compatible API (LM Studio)
                    payload = {
                        "model": kwargs.get("model", "mimo-vl-7b-rl"),
                        "messages": messages,
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 4096),  # Increased default for reasoning models
                        "stream": False
                    }
                    
                    # Use longer timeout for potentially slow responses
                    timeout = kwargs.get("timeout", 120)  # 2 minutes default timeout
                    
                    response = requests.post(
                        self.endpoint,
                        json=payload,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    
                    # Handle OpenAI response format
                    response_data = response.json()
                    
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        message = response_data["choices"][0]["message"]
                        content = message.get("content", "")
                        
                        # Handle MiMo-VL-7B-RL model which puts response in reasoning_content when content is empty
                        if not content and "reasoning_content" in message:
                            reasoning_content = message.get("reasoning_content", "")
                            if reasoning_content:
                                self.logger.info(f"[ModelClient] Using reasoning_content as response (content was empty)")
                                return reasoning_content
                        
                        return content
                    else:
                        self.logger.warning(f"[ModelClient] Unexpected OpenAI response format: {response_data}")
                        return ""
                        
                else:
                    # Ollama API
                    prompt = self._messages_to_prompt(messages)
                    
                    payload = {
                        "model": kwargs.get("model", "qwen2.5:3b"),
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", 0.7),
                            "num_predict": kwargs.get("max_tokens", 4096)  # Increased default for reasoning models
                        }
                    }
                    
                    # Use longer timeout for potentially slow responses
                    timeout = kwargs.get("timeout", 120)  # 2 minutes default timeout
                    
                    response = requests.post(
                        self.endpoint,
                        json=payload,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    
                    # Handle Ollama response format
                    response_data = response.json()
                    
                    if "response" in response_data:
                        return response_data["response"]
                    else:
                        self.logger.warning(f"[ModelClient] Unexpected Ollama response format: {response_data}")
                        return ""
                        
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"[ModelClient] Request timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] All {max_retries} attempts failed due to timeout")
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"[ModelClient] Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"[ModelClient] All {max_retries} attempts failed due to connection error")
                    raise
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"[ModelClient] API request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1 and "500" in str(e):  # Retry on server errors
                    self.logger.info(f"[ModelClient] Server error, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"[ModelClient] Generation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"[ModelClient] Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise
        
        # This should never be reached, but just in case
        return None
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to a single prompt for Ollama"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        return "\n".join(prompt_parts)