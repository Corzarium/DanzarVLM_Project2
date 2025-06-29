#!/usr/bin/env python3
"""
Real-Time Streaming LLM Service
===============================

Implements real-time streaming of LLM responses with concurrent TTS generation.
Based on professional streaming patterns from LiteLLM and async TTS implementations.

Key Features:
- Real-time LLM response streaming (word-by-word or sentence-by-sentence)
- Concurrent TTS generation as text streams in
- Immediate user feedback with progressive response display
- Async/await patterns for non-blocking operation
- Proper error handling and fallback mechanisms
"""

import asyncio
import time
import logging
import re
from typing import List, Optional, Callable, AsyncGenerator, Dict, Any
from dataclasses import dataclass
import threading

@dataclass
class StreamingChunk:
    """A chunk of streaming response"""
    text: str
    chunk_type: str  # 'word', 'sentence', 'phrase'
    is_final: bool
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class RealTimeStreamingLLMService:
    """
    Real-time streaming LLM service with concurrent TTS generation.
    
    Features:
    - Streams LLM responses word-by-word or sentence-by-sentence
    - Generates TTS audio concurrently as text streams in
    - Provides immediate user feedback
    - Handles Azure TTS and other TTS services
    - Non-blocking async operation
    """
    
    def __init__(self, app_context, model_client=None, tts_service=None):
        self.app_context = app_context
        self.logger = getattr(app_context, 'logger', None)
        self.config = app_context.global_settings
        
        # Services
        self.model_client = model_client
        self.tts_service = tts_service
        
        # Streaming configuration
        self.streaming_config = self.config.get('STREAMING_RESPONSE', {})
        self.enable_streaming = self.streaming_config.get('enabled', True)
        self.stream_mode = 'sentence'  # Changed from 'word' to 'sentence' for better TTS
        self.min_chunk_length = 1
        self.chunk_delay_ms = 0  # No delay for sentence streaming
        
        # TTS streaming configuration
        self.enable_tts_streaming = self.streaming_config.get('enable_tts_streaming', True)
        self.tts_concurrent_limit = self.streaming_config.get('tts_concurrent_limit', 3)
        
        # Active streaming sessions
        self.active_streams = {}
        self.stream_counter = 0
        
        # TTS queue for ordered playback
        self.tts_queue = asyncio.Queue(maxsize=10)
        self.tts_processing = False
        self.tts_task = None
        
        if self.logger:
            self.logger.info(f"[RealTimeStreamingLLM] Initialized (enabled: {self.enable_streaming}, mode: {self.stream_mode})")
    
    async def initialize(self) -> bool:
        """Initialize the streaming service."""
        try:
            if self.logger:
                self.logger.info("[RealTimeStreamingLLM] Service initializing...")
            if not self.enable_streaming:
                if self.logger:
                    self.logger.info("[RealTimeStreamingLLM] Streaming disabled in configuration")
                return True
            
            # Start TTS queue processor
            if self.enable_tts_streaming:
                asyncio.create_task(self._tts_queue_processor())
                if self.logger:
                    self.logger.info("[RealTimeStreamingLLM] TTS queue processor started")
            
            if self.logger:
                self.logger.info("[RealTimeStreamingLLM] Service initialized successfully")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Initialization error: {e}")
            return False
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text forms a complete sentence for TTS."""
        text = text.strip()
        if not text:
            return False
        
        # Check for sentence-ending punctuation
        if text.endswith(('.', '!', '?')):
            return True
        
        # Check for natural sentence breaks (comma + space + capital letter)
        if re.search(r',\s+[A-Z]', text):
            return True
        
        # Check for list items (number + period)
        if re.search(r'\d+\.\s+', text):
            return True
        
        return False
    
    def _fix_compound_words(self, text: str) -> str:
        """Fix broken compound words in text."""
        # Common compound words and game terms to preserve
        compound_words = [
            'necromancer', 'everquest', 'shapeshifter', 'spellcaster', 'marksman',
            'sorcerer', 'druid', 'cleric', 'bard', 'rogue', 'mage', 'hunter',
            'warlock', 'paladin', 'warrior', 'monk', 'ranger', 'wizard'
        ]
        
        # Fix broken compound words
        fixed_text = text
        
        for compound in compound_words:
            # Look for broken versions of the compound word with various patterns
            broken_patterns = []
            
            # Pattern 1: "Nec rom ancer" -> "necromancer"
            for i in range(1, len(compound)):
                broken = f"{compound[:i]} {compound[i:]}"
                broken_patterns.append(broken)
            
            # Pattern 2: "Nec rom an cer" -> "necromancer" (multiple breaks)
            for i in range(1, len(compound)-1):
                for j in range(i+1, len(compound)):
                    broken = f"{compound[:i]} {compound[i:j]} {compound[j:]}"
                    broken_patterns.append(broken)
            
            # Pattern 3: "Nec rom an cer" -> "necromancer" (three breaks)
            for i in range(1, len(compound)-2):
                for j in range(i+1, len(compound)-1):
                    for k in range(j+1, len(compound)):
                        broken = f"{compound[:i]} {compound[i:j]} {compound[j:k]} {compound[k:]}"
                        broken_patterns.append(broken)
            
            # Replace all broken patterns with the correct compound word
            for broken in broken_patterns:
                # Use case-insensitive replacement
                import re
                pattern = re.compile(re.escape(broken), re.IGNORECASE)
                fixed_text = pattern.sub(compound, fixed_text)
        
        return fixed_text
    
    def _has_new_complete_content(self, accumulated_text: str, last_processed_length: int) -> bool:
        """Check if we have new complete content to process."""
        new_content = accumulated_text[last_processed_length:]
        
        # For sentence mode, prioritize sentence completion
        if self.stream_mode == 'sentence':
            # Look for complete sentences (ending with .!?)
            return (new_content.endswith('.') or 
                   new_content.endswith('!') or 
                   new_content.endswith('?') or
                   # Also check for natural sentence breaks
                   bool(re.search(r'[.!?]\s+[A-Z]', new_content)))
        elif self.stream_mode == 'word':
            # Check if we have complete words (ending with space or punctuation)
            return (new_content.endswith(' ') or 
                   new_content.endswith('.') or 
                   new_content.endswith('!') or 
                   new_content.endswith('?') or
                   new_content.endswith(',') or
                   new_content.endswith(';') or
                   new_content.endswith(':'))
        else:  # phrase mode
            # Look for complete phrases
            return (new_content.endswith(',') or 
                   new_content.endswith(';') or 
                   new_content.endswith(':'))
    
    async def handle_user_text_query_streaming(
        self, 
        user_text: str, 
        user_name: str,
        text_callback: Optional[Callable[[str], None]] = None,
        tts_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[str, bool], None]] = None
    ) -> str:
        """
        Handle user query with real-time streaming response.
        
        Args:
            user_text: User's input text
            user_name: Username for context
            text_callback: Called with text chunks for Discord display
            tts_callback: Called with text chunks for TTS generation
            progress_callback: Called with progress updates
            
        Returns:
            Complete response text
        """
        if not self.enable_streaming:
            # Fallback to non-streaming
            return await self._handle_non_streaming_query(user_text, user_name)
        
        try:
            if self.logger:
                self.logger.info(f"[RealTimeStreamingLLM] Starting streaming response for '{user_name}'")
            
            # Create streaming session
            stream_id = self._create_stream_session(user_text, user_name)
            
            # Start TTS queue processor if needed
            if self.enable_tts_streaming and not self.tts_processing:
                asyncio.create_task(self._tts_queue_processor())
            
            # Generate streaming response
            full_response = ""
            accumulated_text = ""  # For TTS sentence accumulation
            
            async for chunk in self._stream_llm_response(user_text, user_name):
                full_response += chunk.text
                
                # Call text callback for Discord display
                if text_callback and chunk.text.strip():
                    try:
                        if asyncio.iscoroutinefunction(text_callback):
                            # Create a task for async callbacks to avoid blocking
                            asyncio.create_task(text_callback(chunk.text))
                        else:
                            text_callback(chunk.text)
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"[RealTimeStreamingLLM] Text callback error: {e}")
                
                # Handle TTS generation - accumulate words into sentences for TTS
                if tts_callback and chunk.text.strip():
                    if chunk.chunk_type == 'sentence':
                        # Send complete sentence to TTS
                        try:
                            if asyncio.iscoroutinefunction(tts_callback):
                                # Create a task for async callbacks to avoid blocking
                                asyncio.create_task(tts_callback(chunk.text))
                            else:
                                tts_callback(chunk.text)
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"[RealTimeStreamingLLM] TTS callback error: {e}")
                    elif chunk.chunk_type == 'word':
                        # Accumulate words for TTS
                        accumulated_text += chunk.text
                        # Check if we have a complete sentence for TTS
                        if self._is_complete_sentence(accumulated_text):
                            try:
                                if asyncio.iscoroutinefunction(tts_callback):
                                    # Create a task for async callbacks to avoid blocking
                                    asyncio.create_task(tts_callback(accumulated_text.strip()))
                                else:
                                    tts_callback(accumulated_text.strip())
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"[RealTimeStreamingLLM] TTS callback error: {e}")
                            accumulated_text = ""  # Reset for next sentence
                
                # Call progress callback
                if progress_callback:
                    progress_callback(chunk.text, chunk.is_final)
                
                # Small delay for natural streaming
                if not chunk.is_final:
                    await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            
            # Send any remaining accumulated text to TTS
            if tts_callback and accumulated_text.strip():
                try:
                    if asyncio.iscoroutinefunction(tts_callback):
                        # Create a task for async callbacks to avoid blocking
                        asyncio.create_task(tts_callback(accumulated_text.strip()))
                    else:
                        tts_callback(accumulated_text.strip())
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[RealTimeStreamingLLM] Final TTS callback error: {e}")
            
            if self.logger:
                self.logger.info(f"[RealTimeStreamingLLM] Streaming completed for '{user_name}'")
            return full_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Streaming error: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    async def _stream_llm_response(self, user_text: str, user_name: str) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream LLM response in real-time.
        
        Yields:
            StreamingChunk objects with text and metadata
        """
        try:
            # Prepare the messages in the format expected by generate_streaming
            messages = self._prepare_streaming_messages(user_text, user_name)
            
            # Call LLM with streaming
            if self.model_client and hasattr(self.model_client, 'generate_streaming'):
                # Use model client's streaming capability
                accumulated_text = ""
                last_processed_length = 0
                
                async for response_chunk in self.model_client.generate_streaming(messages):
                    # Accumulate text to prevent word breaking
                    accumulated_text += response_chunk
                    
                    # Process accumulated text when we have new complete sentences
                    if self._has_new_complete_content(accumulated_text, last_processed_length):
                        # Get the new content since last processing
                        new_content = accumulated_text[last_processed_length:]
                        # Process only the new content
                        for chunk in self._process_accumulated_text(new_content):
                            yield chunk
                        # Update last processed length
                        last_processed_length = len(accumulated_text)
                
                # Process any remaining text at the end
                if accumulated_text[last_processed_length:].strip():
                    remaining_content = accumulated_text[last_processed_length:]
                    for chunk in self._process_accumulated_text(remaining_content):
                        chunk.is_final = True
                        yield chunk
            else:
                # Fallback: generate full response and stream it
                full_response = await self._generate_full_response(user_text, user_name)
                for chunk in self._stream_text_response(full_response):
                    yield chunk
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] LLM streaming error: {e}")
            yield StreamingChunk(
                text=f"Sorry, I encountered an error: {str(e)}",
                chunk_type='sentence',
                is_final=True,
                timestamp=time.time()
            )
    
    def _smart_split_words(self, text: str) -> List[str]:
        """
        Smart word splitting that preserves contractions, punctuation, and compound words.
        
        Examples:
        - "Necromancer's" -> ["Necromancer's"]
        - "Cleric: The" -> ["Cleric:", "The"]
        - "shapeshifter" -> ["shapeshifter"]
        - "don't" -> ["don't"]
        """
        import re
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Common compound words and game terms to preserve
        compound_words = [
            'necromancer', 'everquest', 'shapeshifter', 'spellcaster', 'marksman',
            'sorcerer', 'druid', 'cleric', 'bard', 'rogue', 'mage', 'hunter',
            'warlock', 'paladin', 'warrior', 'monk', 'ranger', 'wizard'
        ]
        
        # Create a pattern that includes compound words as single units
        compound_pattern = '|'.join(r'\b' + re.escape(word) + r'\b' for word in compound_words)
        
        # Pattern to match words with contractions and punctuation
        word_pattern = r"""
        (?:
            # Compound words (preserved as single units)
            """ + compound_pattern + r"""
            |
            # Words with contractions/possessives
            \b\w+(?:'[a-z]+)?\b
            |
            # Punctuation marks (including periods)
            [,;:!?.]
            |
            # Numbers
            \b\d+\b
            |
            # Special characters that should be separate
            [@#$%^&*()\[\]{}|\\/"'`~]
        )
        """
        
        # Find all matches
        matches = re.findall(word_pattern, text, re.VERBOSE | re.IGNORECASE)
        
        # Filter out empty matches and clean up
        words = []
        for match in matches:
            match = match.strip()
            if match:
                # Add space after words that aren't punctuation
                if not re.match(r'^[,;:!?.]$', match):
                    words.append(match + ' ')
                else:
                    words.append(match)
        
        return words

    def _process_accumulated_text(self, text: str) -> List[StreamingChunk]:
        """Process accumulated text with smart splitting to prevent word breaking."""
        chunks = []
        
        if self.stream_mode == 'sentence':
            # Split into sentences and preserve compound words
            sentences = self._split_into_sentences(text)
            for sentence in sentences:
                if len(sentence.strip()) >= self.min_chunk_length:
                    # Apply smart word splitting to preserve compound words within sentences
                    processed_sentence = self._fix_compound_words(sentence)
                    chunks.append(StreamingChunk(
                        text=processed_sentence,
                        chunk_type='sentence',
                        is_final=False,
                        timestamp=time.time()
                    ))
        elif self.stream_mode == 'word':
            # Use smart splitting on accumulated text
            words = self._smart_split_words(text)
            for word in words:
                chunks.append(StreamingChunk(
                    text=word,
                    chunk_type='word',
                    is_final=False,
                    timestamp=time.time()
                ))
        else:  # phrase mode
            phrases = self._split_into_phrases(text)
            for phrase in phrases:
                if len(phrase.strip()) >= self.min_chunk_length:
                    chunks.append(StreamingChunk(
                        text=phrase,
                        chunk_type='phrase',
                        is_final=False,
                        timestamp=time.time()
                    ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentence endings
        sentence_pattern = r'[.!?]+(?:\s+|$|"|\')'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences and add punctuation back
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= self.min_chunk_length:
                # Add appropriate punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                result.append(sentence)
        
        return result
    
    def _split_into_phrases(self, text: str) -> List[str]:
        """Split text into phrases for streaming."""
        # Split by commas, semicolons, and other natural breaks
        phrase_pattern = r'[,;:]\s+'
        phrases = re.split(phrase_pattern, text)
        
        result = []
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase and len(phrase) >= self.min_chunk_length:
                result.append(phrase)
        
        return result
    
    def _stream_text_response(self, full_response: str) -> List[StreamingChunk]:
        """Stream a complete text response in chunks."""
        chunks = []
        
        if self.stream_mode == 'word':
            words = self._smart_split_words(full_response)
            for i, word in enumerate(words):
                chunks.append(StreamingChunk(
                    text=word,
                    chunk_type='word',
                    is_final=(i == len(words) - 1),
                    timestamp=time.time()
                ))
        elif self.stream_mode == 'sentence':
            sentences = self._split_into_sentences(full_response)
            for i, sentence in enumerate(sentences):
                chunks.append(StreamingChunk(
                    text=sentence,
                    chunk_type='sentence',
                    is_final=(i == len(sentences) - 1),
                    timestamp=time.time()
                ))
        else:  # phrase mode
            phrases = self._split_into_phrases(full_response)
            for i, phrase in enumerate(phrases):
                chunks.append(StreamingChunk(
                    text=phrase,
                    chunk_type='phrase',
                    is_final=(i == len(phrases) - 1),
                    timestamp=time.time()
                ))
        
        return chunks
    
    async def _generate_full_response(self, user_text: str, user_name: str) -> str:
        """Generate full response as fallback when streaming is not available."""
        try:
            if self.model_client and hasattr(self.model_client, 'generate'):
                messages = self._prepare_streaming_messages(user_text, user_name)
                return await self.model_client.generate(messages)
            else:
                # Fallback to basic response
                return f"I understand you said: '{user_text}'. Let me think about that..."
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Full response generation error: {e}")
            return f"Sorry, I encountered an error while processing your request."
    
    def _prepare_streaming_messages(self, user_text: str, user_name: str) -> List[Dict[str, str]]:
        """Prepare messages in the format expected by generate_streaming."""
        
        # Get recent conversation context from STM
        conversation_context = ""
        if hasattr(self.app_context, 'short_term_memory_service') and self.app_context.short_term_memory_service:
            try:
                recent_entries = self.app_context.short_term_memory_service.get_recent_context(user_name, max_entries=5)
                if recent_entries:
                    context_parts = []
                    for entry in recent_entries:
                        if entry.entry_type == 'user_input':
                            context_parts.append(f"User: {entry.content}")
                        elif entry.entry_type == 'bot_response':
                            context_parts.append(f"Assistant: {entry.content}")
                    
                    if context_parts:
                        conversation_context = "\n".join(context_parts[-8:])  # Last 4 exchanges
                        if self.logger:
                            self.logger.debug(f"[RealTimeStreamingLLM] Using conversation context: {len(conversation_context)} chars")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[RealTimeStreamingLLM] Failed to get STM context: {e}")
        
        # Use profile-based system prompt instead of hardcoded one
        system_message = self.app_context.active_profile.system_prompt_commentary
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add conversation context if available
        if conversation_context:
            messages.append({
                "role": "system", 
                "content": f"Recent conversation context:\n{conversation_context}\n\nRespond naturally, considering the conversation history above."
            })
        
        messages.append({"role": "user", "content": f"{user_text}"})
        
        return messages
    
    async def _tts_queue_processor(self):
        """Process TTS queue for ordered audio playback."""
        if self.tts_processing:
            return
        
        self.tts_processing = True
        if self.logger:
            self.logger.info("[RealTimeStreamingLLM] TTS queue processor started")
        
        try:
            while True:
                try:
                    # Get TTS task from queue - use proper async pattern for asyncio.Queue
                    tts_task = await asyncio.wait_for(
                        self.tts_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process TTS task
                    await self._process_tts_task(tts_task)
                    
                except asyncio.TimeoutError:
                    # Check if we should continue
                    if self.tts_queue.empty() and not self.active_streams:
                        break
                    continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[RealTimeStreamingLLM] TTS queue processing error: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] TTS queue processor error: {e}")
        finally:
            self.tts_processing = False
            if self.logger:
                self.logger.info("[RealTimeStreamingLLM] TTS queue processor stopped")
    
    async def _process_tts_task(self, tts_task: Dict[str, Any]):
        """Process a single TTS task."""
        try:
            text = tts_task.get('text', '')
            callback = tts_task.get('callback')
            
            if not text.strip():
                return
            
            # Generate TTS audio
            if self.tts_service:
                # Check if it's Azure TTS
                try:
                    from services.tts_service_azure import AzureTTSService
                    if isinstance(self.tts_service, AzureTTSService):
                        tts_audio = await self.tts_service.synthesize_speech(text)
                    else:
                        tts_audio = self.tts_service.generate_audio(text)
                except ImportError:
                    tts_audio = self.tts_service.generate_audio(text)
                
                if tts_audio and callback:
                    # Call the callback with the audio - handle both async and sync callbacks
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tts_audio)
                    else:
                        callback(tts_audio)
                    if self.logger:
                        self.logger.debug(f"[RealTimeStreamingLLM] TTS generated for: '{text[:30]}...'")
                else:
                    if self.logger:
                        self.logger.warning(f"[RealTimeStreamingLLM] TTS generation failed for: '{text[:30]}...'")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] TTS task processing error: {e}")
    
    def _create_stream_session(self, user_text: str, user_name: str) -> str:
        """Create a new streaming session."""
        self.stream_counter += 1
        stream_id = f"stream_{self.stream_counter}_{int(time.time())}"
        
        self.active_streams[stream_id] = {
            'user_text': user_text,
            'user_name': user_name,
            'created_at': time.time(),
            'chunks_processed': 0
        }
        
        return stream_id
    
    async def _handle_non_streaming_query(self, user_text: str, user_name: str) -> str:
        """Handle query without streaming (fallback)."""
        try:
            if self.model_client and hasattr(self.model_client, 'generate'):
                messages = self._prepare_streaming_messages(user_text, user_name)
                return await self.model_client.generate(messages)
            else:
                return f"I understand you said: '{user_text}'. Let me think about that..."
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Non-streaming query error: {e}")
            return f"Sorry, I encountered an error while processing your request."
    
    async def add_tts_task(self, text: str, callback: Callable[[bytes], None]):
        """Add a TTS task to the queue."""
        try:
            if not self.enable_tts_streaming:
                return
            
            task = {
                'text': text,
                'callback': callback,
                'timestamp': time.time(),
                'id': f"tts_{int(time.time() * 1000)}"
            }
            
            await self.tts_queue.put(task)
            
            if self.logger:
                self.logger.debug(f"[RealTimeStreamingLLM] Added TTS task: {text[:50]}...")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Error adding TTS task: {e}")

    async def update_visual_context(self, visual_context: str):
        """Update the visual context for the VLM with CLIP insights."""
        try:
            # Store visual context in app context for the VLM to access
            if hasattr(self.app_context, 'current_visual_context'):
                self.app_context.current_visual_context = visual_context
            else:
                self.app_context.current_visual_context = visual_context
            
            # Also store in a timestamped history
            if not hasattr(self.app_context, 'visual_context_history'):
                self.app_context.visual_context_history = []
            
            self.app_context.visual_context_history.append({
                'timestamp': time.time(),
                'context': visual_context
            })
            
            # Keep only last 10 visual contexts
            if len(self.app_context.visual_context_history) > 10:
                self.app_context.visual_context_history = self.app_context.visual_context_history[-10:]
            
            if self.logger:
                self.logger.info(f"[RealTimeStreamingLLM] Updated visual context: {visual_context}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RealTimeStreamingLLM] Error updating visual context: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'enabled': self.enable_streaming,
            'stream_mode': self.stream_mode,
            'active_streams': len(self.active_streams),
            'tts_queue_size': self.tts_queue.qsize(),
            'tts_processing': self.tts_processing
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.active_streams.clear()
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except:
                break 