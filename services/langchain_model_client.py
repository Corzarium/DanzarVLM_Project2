"""
LangChain Model Client Wrapper
==============================

This module provides a LangChain-compatible wrapper for the existing Ollama model client,
enabling integration with LangChain agents and tools.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class LangChainOllamaWrapper(BaseChatModel):
    """LangChain wrapper for Ollama model client."""
    
    def __init__(self, ollama_client):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama_wrapper"
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model."""
        try:
            # Convert LangChain messages to Ollama format
            ollama_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    ollama_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    ollama_messages.append({
                        "role": "assistant", 
                        "content": message.content
                    })
                elif isinstance(message, SystemMessage):
                    ollama_messages.append({
                        "role": "system",
                        "content": message.content
                    })
            
            # Call Ollama client
            response = await self.ollama_client.chat_completion(
                messages=ollama_messages,
                **kwargs
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                generation = ChatGeneration(message=AIMessage(content=content))
                return ChatResult(generations=[generation])
            else:
                raise ValueError("No response from Ollama client")
                
        except Exception as e:
            self.logger.error(f"Error in LangChain Ollama wrapper: {e}")
            raise
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate method."""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs)) 