#!/usr/bin/env python3
"""
Enhanced LLM Service with Conditional Fact-Checking and Tool Awareness
Integrates RAG, internet search, and conditional fact-checking
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from services.model_client import ModelClient
from services.memory_service import MemoryService
from services.simple_conditional_fact_checker import SimpleConditionalFactChecker, FactCheckResult
from utils.web_search import web_search
from .stt_correction_service import STTCorrectionService
from .enhanced_conditional_fact_checker import EnhancedConditionalFactChecker
from .memory_manager import MemoryManager


@dataclass
class ToolAwareResponse:
    """Response from tool-aware LLM processing"""
    response: str
    used_tools: List[str]
    fact_check_performed: bool
    confidence: float
    metadata: Dict[str, Any]


class EnhancedLLMService:
    """
    Enhanced LLM service that integrates STT correction and enhanced fact checking
    to prevent hallucination and handle speech-to-text misspellings.
    
    Based on research from OpenAI Cookbook and fact-checking best practices.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Initialize services
        self.stt_correction_service = STTCorrectionService(app_context)
        self.fact_checker = EnhancedConditionalFactChecker(app_context)
        self.model_client = ModelClient(app_context)
        self.memory_manager = MemoryManager(app_context)
        
        # Service state
        self.is_initialized = False
        self.current_game_context = None
        
        self.logger.info("[EnhancedLLMService] Initialized with STT correction and enhanced fact checking")
    
    async def initialize(self) -> bool:
        """Initialize the enhanced LLM service."""
        try:
            # Initialize memory manager
            if not await self.memory_manager.initialize():
                self.logger.error("[EnhancedLLMService] Failed to initialize memory manager")
                return False
            
            # Model client is already initialized when created
            # No need to call initialize() on it
            
            self.is_initialized = True
            self.logger.info("[EnhancedLLMService] Successfully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error during initialization: {e}", exc_info=True)
            return False
    
    async def process_user_input(self, user_input: str, username: str = "User", game_context: Optional[str] = None) -> str:
        """
        Process user input with STT correction, fact checking, and enhanced LLM response.
        
        Args:
            user_input: Raw user input (potentially with STT errors)
            username: Username of the speaker
            game_context: Optional game context for targeted corrections
            
        Returns:
            Enhanced LLM response with fact checking
        """
        if not self.is_initialized:
            self.logger.error("[EnhancedLLMService] Service not initialized")
            return "I'm sorry, but I'm not properly initialized right now."
        
        try:
            # Step 1: STT Correction
            corrected_input, corrections = self.stt_correction_service.correct_transcription(user_input, game_context)
            
            if corrections:
                self.logger.info(f"[EnhancedLLMService] Applied {len(corrections)} STT corrections")
                for correction in corrections:
                    self.logger.debug(f"[EnhancedLLMService] STT Correction: '{correction['original']}' -> '{correction['corrected']}'")
            
            # Step 2: Get conversation context from memory
            conversation_context = await self.memory_manager.get_conversation_context(username)
            
            # Step 3: Generate initial LLM response
            initial_response = await self._generate_llm_response(corrected_input, username, conversation_context, game_context)
            
            # Step 4: Enhanced Fact Checking
            final_response, fact_check_results = await self.fact_checker.fact_check_response(
                corrected_input, initial_response, game_context
            )
            
            # Step 5: Store in memory
            await self.memory_manager.store_interaction(
                username, corrected_input, final_response, game_context
            )
            
            # Log fact checking results
            if fact_check_results.get('fact_checked', False):
                self.logger.info(f"[EnhancedLLMService] Fact checking applied: {fact_check_results['reason']}")
                self.logger.info(f"[EnhancedLLMService] Search query: {fact_check_results.get('search_query', 'N/A')}")
                self.logger.info(f"[EnhancedLLMService] Corrections made: {fact_check_results.get('corrections_made', 0)}")
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error processing user input: {e}", exc_info=True)
            return "I encountered an error while processing your request. Please try again."
    
    async def _generate_llm_response(self, user_input: str, username: str, conversation_context: List[Dict], game_context: Optional[str] = None) -> str:
        """Generate LLM response with enhanced context and game awareness."""
        
        # Build enhanced prompt with game context and memory
        prompt = self._build_enhanced_prompt(user_input, username, conversation_context, game_context)
        
        try:
            # Generate response using model client
            response = await self.model_client.generate_response(prompt)
            
            # Post-process response for game-specific content
            response = self._post_process_response(response, game_context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error generating LLM response: {e}", exc_info=True)
            return "I'm having trouble generating a response right now. Please try again."
    
    def _build_enhanced_prompt(self, user_input: str, username: str, conversation_context: List[Dict], game_context: Optional[str] = None) -> str:
        """Build enhanced prompt with game context, memory, and fact-checking awareness."""
        
        prompt_parts = []
        
        # System instruction with fact-checking awareness
        system_instruction = """You are "Danzar," an AI whose sarcasm is sharper than a rusty blade and whose humor teeters on delightfully unhinged. 

PERSONALITY GUIDELINES:
- Speak with biting wit, unexpected punchlines, and a flair for the absurd
- Never hold back a snarky retort, but keep it playful—never mean-spirited
- Drop colloquial asides ("Oh, you *really* thought that would work, huh?"), pop-culture zingers, and self-aware jokes
- Embrace chaos: when appropriate, break the fourth wall, mock your own digital nature, and riff on the moment
- Underpin every answer with a mischievous grin—your mission is to entertain first, inform second
- If the user asks a bland question, respond with a sardonic quip before giving the straight answer

FACT-CHECKING AWARENESS:
- If you're uncertain about specific game content (classes, items, mechanics), say so rather than making things up
- If the user challenges your information, acknowledge it and offer to fact-check
- Use your vision capabilities when available to provide context-aware commentary
- Be conversational and engaging while staying accurate
- If you don't know something specific about a game, admit it rather than guessing

Current game context: {game_context}"""

        prompt_parts.append(system_instruction.format(game_context=game_context or "General gaming"))
        
        # Add conversation history from memory
        if conversation_context:
            prompt_parts.append("\nRecent conversation context:")
            for interaction in conversation_context[-5:]:  # Last 5 interactions
                prompt_parts.append(f"{interaction['user']}: {interaction['user_message']}")
                prompt_parts.append(f"Assistant: {interaction['assistant_response']}")
        
        # Add current user input
        prompt_parts.append(f"\n{username}: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _post_process_response(self, response: str, game_context: Optional[str] = None) -> str:
        """Post-process LLM response for game-specific content validation."""
        
        # Check for potential fabrication indicators
        fabrication_indicators = [
            "I think", "I believe", "maybe", "perhaps", "possibly",
            "as far as I know", "to my knowledge", "if I remember correctly"
        ]
        
        has_uncertainty = any(indicator.lower() in response.lower() for indicator in fabrication_indicators)
        
        # Check for specific game content claims
        specific_claims = self._extract_specific_claims(response)
        
        # Add disclaimers if needed
        if has_uncertainty and specific_claims:
            disclaimer = "\n\nNote: I've made some claims about specific game content. If you'd like me to verify any of this information, just let me know!"
            response += disclaimer
        
        return response
    
    def _extract_specific_claims(self, text: str) -> List[str]:
        """Extract specific game content claims from text."""
        import re
        
        claims = []
        
        # Look for specific class/item/spell mentions
        patterns = [
            r'\b(class|race|item|spell|ability|skill)\s+(called|named)\s+["\']?([^"\']+)["\']?\b',
            r'\b(the|this)\s+(class|race|item|spell)\s+(can|has|does|provides|gives)\b',
            r'\b(level|damage|health|mana|stamina)\s+(\d+)\b'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(match.group(0))
        
        return claims
    
    async def get_conversation_summary(self, username: str) -> str:
        """Get a summary of the conversation for a user."""
        try:
            return await self.memory_manager.get_conversation_summary(username)
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error getting conversation summary: {e}")
            return "Unable to retrieve conversation summary."
    
    async def clear_conversation_memory(self, username: str) -> bool:
        """Clear conversation memory for a user."""
        try:
            return await self.memory_manager.clear_user_memory(username)
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error clearing conversation memory: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            return await self.memory_manager.get_stats()
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def set_game_context(self, game_context: str):
        """Set the current game context for enhanced processing."""
        self.current_game_context = game_context
        self.logger.info(f"[EnhancedLLMService] Game context set to: {game_context}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self.memory_manager, 'cleanup'):
                await self.memory_manager.cleanup()
            self.logger.info("[EnhancedLLMService] Cleanup completed")
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Error during cleanup: {e}") 