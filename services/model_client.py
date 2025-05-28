import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger("DanzarVLM.ModelClient")

class ModelClient:
    def __init__(self, api_base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ModelClient with API configuration.
        
        Args:
            api_base_url: Base URL for the LLM API
            api_key: Optional API key for authentication
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.logger = logger

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256, model: str = "mistral", endpoint: str = "chat/completions") -> str:
        """
        Generate a response from the LLM model.
        
        Args:
            prompt: The input prompt for generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            self.logger.info(f"[ModelClient] Generating response with temp={temperature}, max_tokens={max_tokens}")
            
            # Prepare the API request
            url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the API call
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise ValueError("Invalid response format from LLM API")
                
            # Extract the generated text
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[ModelClient] API request failed: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"[ModelClient] Generation failed: {e}", exc_info=True)
            raise 