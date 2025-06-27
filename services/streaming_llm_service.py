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
from queue import Queue
import threading

@dataclass
class StreamingChunk:
    """A chunk of streaming response"""
    text: str
    chunk_type: str  # 'word', 'sentence', 'phrase'
    is_final: bool
    timestamp: float
    metadata: Dict[str, Any] = None

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
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Services
        self.model_client = model_client
        self.tts_service = tts_service
        
        # Streaming configuration
        self.streaming_config = self.config.get('STREAMING_RESPONSE', {})
        self.enable_streaming = self.streaming_config.get('enabled', True)
        self.stream_mode = self.streaming_config.get('mode', 'sentence')  # 'word', 'sentence', 'phrase'
        self.min_chunk_length = self.streaming_config.get('min_chunk_length', 10)
        self.chunk_delay_ms = self.streaming_config.get('chunk_delay_ms', 100)
        
        # TTS streaming configuration
        self.enable_tts_streaming = self.streaming_config.get('enable_tts_streaming', True)
        self.tts_concurrent_limit = self.streaming_config.get('tts_concurrent_limit', 3)
        
        # Active streaming sessions
        self.active_streams = {}
        self.stream_counter = 0
        
        # TTS queue for ordered playback
        self.tts_queue = Queue()
        self.tts_processing = False
        
        self.logger.info(f"[RealTimeStreamingLLM] Initialized (enabled: {self.enable_streaming}, mode: {self.stream_mode})")
    
    async def initialize(self) -> bool:
        """Initialize the streaming service."""
        try:
            if not self.enable_streaming:
                self.logger.info("[RealTimeStreamingLLM] Streaming disabled in configuration")
                return True
            
            # Start TTS queue processor
            if self.enable_tts_streaming:
                asyncio.create_task(self._tts_queue_processor())
                self.logger.info("[RealTimeStreamingLLM] TTS queue processor started")
            
            self.logger.info("[RealTimeStreamingLLM] Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] Initialization failed: {e}")
            return False
    
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
            self.logger.info(f"[RealTimeStreamingLLM] Starting streaming response for '{user_name}'")
            
            # Create streaming session
            stream_id = self._create_stream_session(user_text, user_name)
            
            # Start TTS queue processor if needed
            if self.enable_tts_streaming and not self.tts_processing:
                asyncio.create_task(self._tts_queue_processor())
            
            # Generate streaming response
            full_response = ""
            async for chunk in self._stream_llm_response(user_text, user_name):
                full_response += chunk.text
                
                # Call text callback for Discord display
                if text_callback and chunk.text.strip():
                    text_callback(chunk.text)
                
                # Call TTS callback for audio generation
                if tts_callback and chunk.text.strip() and chunk.chunk_type == 'sentence':
                    tts_callback(chunk.text)
                
                # Call progress callback
                if progress_callback:
                    progress_callback(chunk.text, chunk.is_final)
                
                # Small delay for natural streaming
                if not chunk.is_final:
                    await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            
            self.logger.info(f"[RealTimeStreamingLLM] Streaming completed for '{user_name}'")
            return full_response
            
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] Streaming error: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    async def _stream_llm_response(self, user_text: str, user_name: str) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream LLM response in real-time.
        
        Yields:
            StreamingChunk objects with text and metadata
        """
        try:
            # Prepare the prompt
            prompt = self._prepare_streaming_prompt(user_text, user_name)
            
            # Call LLM with streaming
            if self.model_client and hasattr(self.model_client, 'generate_streaming'):
                # Use model client's streaming capability
                async for response_chunk in self.model_client.generate_streaming(prompt):
                    for chunk in self._process_response_chunk(response_chunk):
                        yield chunk
            else:
                # Fallback: generate full response and stream it
                full_response = await self._generate_full_response(user_text, user_name)
                for chunk in self._stream_text_response(full_response):
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] LLM streaming error: {e}")
            yield StreamingChunk(
                text=f"Sorry, I encountered an error: {str(e)}",
                chunk_type='sentence',
                is_final=True,
                timestamp=time.time()
            )
    
    def _process_response_chunk(self, response_chunk: str) -> List[StreamingChunk]:
        """Process a chunk of LLM response into streaming chunks."""
        chunks = []
        
        if self.stream_mode == 'word':
            # Stream word by word
            words = response_chunk.split()
            for word in words:
                chunks.append(StreamingChunk(
                    text=word + ' ',
                    chunk_type='word',
                    is_final=False,
                    timestamp=time.time()
                ))
        elif self.stream_mode == 'sentence':
            # Stream sentence by sentence
            sentences = self._split_into_sentences(response_chunk)
            for sentence in sentences:
                if len(sentence.strip()) >= self.min_chunk_length:
                    chunks.append(StreamingChunk(
                        text=sentence,
                        chunk_type='sentence',
                        is_final=False,
                        timestamp=time.time()
                    ))
        else:  # phrase mode
            # Stream phrase by phrase
            phrases = self._split_into_phrases(response_chunk)
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
            words = full_response.split()
            for i, word in enumerate(words):
                chunks.append(StreamingChunk(
                    text=word + ' ',
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
                return await self.model_client.generate(user_text)
            else:
                # Fallback to basic response
                return f"I understand you said: '{user_text}'. Let me think about that..."
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] Full response generation error: {e}")
            return f"Sorry, I encountered an error while processing your request."
    
    def _prepare_streaming_prompt(self, user_text: str, user_name: str) -> str:
        """Prepare prompt for streaming LLM generation."""
        # Add streaming-specific instructions
        streaming_instructions = """
        Please respond naturally and conversationally. 
        Your response will be streamed in real-time, so keep it engaging and clear.
        """
        
        return f"{streaming_instructions}\n\nUser ({user_name}): {user_text}\nAssistant:"
    
    async def _tts_queue_processor(self):
        """Process TTS queue for ordered audio playback."""
        if self.tts_processing:
            return
        
        self.tts_processing = True
        self.logger.info("[RealTimeStreamingLLM] TTS queue processor started")
        
        try:
            while True:
                try:
                    # Get TTS task from queue
                    tts_task = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.tts_queue.get, True
                        ),
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
                    self.logger.error(f"[RealTimeStreamingLLM] TTS queue processing error: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] TTS queue processor error: {e}")
        finally:
            self.tts_processing = False
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
                    # Call the callback with the audio
                    await callback(tts_audio)
                    self.logger.debug(f"[RealTimeStreamingLLM] TTS generated for: '{text[:30]}...'")
                else:
                    self.logger.warning(f"[RealTimeStreamingLLM] TTS generation failed for: '{text[:30]}...'")
            
        except Exception as e:
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
                return await self.model_client.generate(user_text)
            else:
                return f"I understand you said: '{user_text}'. Let me think about that..."
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] Non-streaming query error: {e}")
            return f"Sorry, I encountered an error while processing your request."
    
    def add_tts_task(self, text: str, callback: Callable[[bytes], None]):
        """Add a TTS task to the queue."""
        if not self.enable_tts_streaming:
            return
        
        try:
            self.tts_queue.put_nowait({
                'text': text,
                'callback': callback,
                'timestamp': time.time()
            })
        except Exception as e:
            self.logger.error(f"[RealTimeStreamingLLM] Failed to add TTS task: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'enabled': self.enable_streaming,
            'mode': self.stream_mode,
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