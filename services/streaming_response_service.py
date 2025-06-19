#!/usr/bin/env python3
# services/streaming_response_service.py

import re
import time
import asyncio
import logging
import threading
from typing import List, Optional, Callable, Any, Dict
from dataclasses import dataclass
from queue import Queue, Empty
import queue

@dataclass
class StreamChunk:
    """A chunk of streamed response"""
    text: str
    chunk_type: str  # 'sentence', 'phrase', 'word'
    is_final: bool
    timestamp: float
    metadata: dict = None

class SentenceStreamingService:
    """
    Service for streaming responses sentence-by-sentence to improve TTS processing and user experience.
    Allows for immediate TTS processing of complete sentences while the rest of the response is being generated.
    """
    
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.SentenceStreaming")
        
        # Configuration
        self.enable_streaming = self.ctx.global_settings.get("STREAMING_RESPONSE", {}).get("enabled", True)
        self.min_sentence_length = self.ctx.global_settings.get("STREAMING_RESPONSE", {}).get("min_sentence_length", 10)
        self.sentence_delay_ms = self.ctx.global_settings.get("STREAMING_RESPONSE", {}).get("sentence_delay_ms", 250)
        self.chunk_timeout_s = self.ctx.global_settings.get("STREAMING_RESPONSE", {}).get("chunk_timeout_s", 5.0)
        
        # Sentence boundaries - comprehensive list
        self.sentence_endings = r'[.!?]+(?:\s+|$|"|\')'
        self.sentence_pattern = re.compile(self.sentence_endings)
        
        # Common abbreviations that shouldn't end sentences
        self.abbreviations = {
            'dr', 'mr', 'ms', 'mrs', 'prof', 'vs', 'etc', 'inc', 'ltd', 'corp',
            'i.e', 'e.g', 'ex', 'no', 'st', 'ave', 'blvd', 'rd', 'sq', 'ft', 'in',
            'lb', 'oz', 'kg', 'cm', 'mm', 'km', 'hr', 'min', 'sec', 'am', 'pm',
            'lvl', 'hp', 'mp', 'dps', 'exp', 'dmg', 'def', 'str', 'dex', 'int'  # Gaming terms
        }
        
        # Active streaming sessions
        self.active_streams = {}
        self.stream_counter = 0
        
        self.logger.info(f"[SentenceStreaming] Initialized (enabled: {self.enable_streaming})")

    def start_response_stream(self, user_text: str, user_name: str, 
                            tts_callback: Optional[Callable[[str], None]] = None,
                            text_callback: Optional[Callable[[str], None]] = None,
                            discord_bot = None) -> str:
        """
        Start a new response stream.
        
        Args:
            user_text: The original user query
            user_name: Username of the requester
            tts_callback: Function to call with TTS-ready sentences
            text_callback: Function to call with text chunks for Discord
            discord_bot: Discord bot instance for queue-based processing
            
        Returns:
            Stream ID for managing this stream
        """
        if not self.enable_streaming:
            return None
        
        self.stream_counter += 1
        stream_id = f"stream_{self.stream_counter}_{int(time.time())}"
        
        stream_data = {
            'stream_id': stream_id,
            'user_text': user_text,
            'user_name': user_name,
            'tts_callback': tts_callback,
            'text_callback': text_callback,
            'discord_bot': discord_bot,  # Store discord bot instance
            'buffer': '',
            'sent_sentences': [],
            'chunks_queue': Queue(),
            'is_complete': False,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        self.active_streams[stream_id] = stream_data
        self.logger.info(f"[SentenceStreaming] Started stream {stream_id} for user '{user_name}'")
        
        return stream_id

    def add_text_chunk(self, stream_id: str, text_chunk: str, is_final: bool = False) -> bool:
        """
        Add a chunk of text to an active stream.
        
        Args:
            stream_id: The stream identifier
            text_chunk: New text to add
            is_final: Whether this is the final chunk
            
        Returns:
            True if processed successfully
        """
        if not self.enable_streaming or stream_id not in self.active_streams:
            return False
        
        stream_data = self.active_streams[stream_id]
        stream_data['last_activity'] = time.time()
        
        # Add chunk to buffer
        stream_data['buffer'] += text_chunk
        
        # Process complete sentences from buffer
        sentences = self._extract_complete_sentences(stream_data['buffer'])
        
        for sentence in sentences:
            if sentence and len(sentence.strip()) >= self.min_sentence_length:
                self._process_sentence(stream_data, sentence)
                # Remove processed sentence from buffer
                stream_data['buffer'] = stream_data['buffer'][len(sentence):].lstrip()
        
        # If final chunk, process remaining buffer
        if is_final:
            remaining = stream_data['buffer'].strip()
            if remaining:
                self._process_sentence(stream_data, remaining)
            
            stream_data['is_complete'] = True
            self.finalize_stream(stream_id)
        
        return True

    def stream_complete_response(self, stream_id: str, complete_response: str) -> bool:
        """Stream a complete response as sentences with proper synchronization"""
        if stream_id not in self.active_streams:
            self.logger.warning(f"[SentenceStreaming] Cannot stream - stream {stream_id} not found")
            return False
        
        stream_data = self.active_streams[stream_id]
        sentences = self._split_into_sentences(complete_response)
        
        self.logger.info(f"[SentenceStreaming] Streaming {len(sentences)} sentences for {stream_id}")
        
        # Store sentences in stream data for finalization
        stream_data['sent_sentences'] = sentences
        
        # Mark stream as complete and finalize (this will call the new queue-based system)
        stream_data['is_complete'] = True
        self.finalize_stream(stream_id)
        
        return True

    def _extract_complete_sentences(self, text: str) -> List[str]:
        """Extract complete sentences from text buffer"""
        sentences = []
        
        # Find sentence boundaries
        for match in self.sentence_pattern.finditer(text):
            start = 0 if not sentences else len(''.join(sentences))
            end = match.end()
            
            potential_sentence = text[start:end].strip()
            
            # Check if it's a real sentence end (not an abbreviation)
            if self._is_real_sentence_end(potential_sentence):
                sentences.append(potential_sentence)
        
        return sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split complete text into sentences"""
        # Clean up text first
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = []
        current_pos = 0
        
        # Find all sentence boundaries
        for match in self.sentence_pattern.finditer(text):
            # Extract potential sentence
            sentence_end = match.end()
            potential_sentence = text[current_pos:sentence_end].strip()
            
            # Check if it's a real sentence end (not an abbreviation)
            if self._is_real_sentence_end(potential_sentence):
                sentences.append(potential_sentence)
                current_pos = sentence_end
        
        # Add remaining text as final sentence if any
        remaining = text[current_pos:].strip()
        if remaining and len(remaining) >= self.min_sentence_length:
            sentences.append(remaining)
        
        return sentences

    def _is_real_sentence_end(self, sentence: str) -> bool:
        """Check if this is a real sentence end, not an abbreviation"""
        sentence = sentence.strip()
        
        if len(sentence) < self.min_sentence_length:
            return False
        
        # Check for abbreviations
        words = sentence.lower().split()
        if words:
            last_word = words[-1].rstrip('.!?')
            if last_word in self.abbreviations:
                return False
        
        # Check for common gaming abbreviations
        gaming_abbrevs = ['lvl', 'hp', 'mp', 'exp', 'dmg', 'dps', 'str', 'dex', 'int', 'wis']
        if any(abbrev in sentence.lower() for abbrev in gaming_abbrevs):
            # Be more lenient with gaming content
            return len(sentence) > 15
        
        return True

    def _process_sentence(self, stream_data: dict, sentence: str):
        """Process a complete sentence for TTS and text output"""
        sentence = sentence.strip()
        if not sentence:
            return
        
        stream_data['sent_sentences'].append(sentence)
        
        self.logger.debug(f"[SentenceStreaming] Processing sentence: '{sentence[:50]}...'")
        
        # Send to TTS callback (for immediate audio generation)
        if stream_data['tts_callback']:
            try:
                stream_data['tts_callback'](sentence)
            except Exception as e:
                self.logger.error(f"[SentenceStreaming] TTS callback error: {e}")
        
        # Send to text callback (for Discord text display)
        if stream_data['text_callback']:
            try:
                stream_data['text_callback'](sentence)
            except Exception as e:
                self.logger.error(f"[SentenceStreaming] Text callback error: {e}")

    def finalize_stream(self, stream_id: str):
        """Finalize a stream and process all collected sentences"""
        try:
            if stream_id not in self.active_streams:
                self.logger.warning(f"[SentenceStreaming] Stream {stream_id} not found for finalization")
                return
            
            stream_data = self.active_streams[stream_id]
            sentences = stream_data['sent_sentences']
            
            # Log completion
            duration = time.time() - stream_data['created_at']
            self.logger.info(f"[SentenceStreaming] Finalized stream {stream_id}: {len(sentences)} sentences in {duration:.2f}s")
            
            # Process sentences using pipeline method
            self._queue_sentences_for_discord(stream_data, sentences)
            
            # Clean up
            del self.active_streams[stream_id]
            
        except Exception as e:
            self.logger.error(f"[SentenceStreaming] Error finalizing stream {stream_id}: {e}", exc_info=True)

    def get_full_response(self, stream_id: str) -> Optional[str]:
        """Get the complete response text for a stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream_data = self.active_streams[stream_id]
        return ' '.join(stream_data['sent_sentences'])

    def is_stream_complete(self, stream_id: str) -> bool:
        """Check if a stream is complete"""
        if stream_id not in self.active_streams:
            return True
        
        return self.active_streams[stream_id]['is_complete']

    def get_stream_stats(self, stream_id: str) -> Optional[dict]:
        """Get statistics for a stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream_data = self.active_streams[stream_id]
        return {
            'stream_id': stream_id,
            'user_name': stream_data['user_name'],
            'sentences_sent': len(stream_data['sent_sentences']),
            'is_complete': stream_data['is_complete'],
            'duration': time.time() - stream_data['created_at'],
            'buffer_length': len(stream_data['buffer'])
        }

    def cleanup_old_streams(self, max_age_seconds: int = 300):
        """Clean up streams older than max_age_seconds"""
        current_time = time.time()
        to_remove = []
        
        for stream_id, stream_data in self.active_streams.items():
            if current_time - stream_data['last_activity'] > max_age_seconds:
                to_remove.append(stream_id)
        
        for stream_id in to_remove:
            self.finalize_stream(stream_id)
        
        if to_remove:
            self.logger.info(f"[SentenceStreaming] Cleaned up {len(to_remove)} old streams")

    def _queue_sentences_for_discord(self, stream_data: Dict, sentences: List[str]):
        """Queue sentences for Discord processing using pipeline method"""
        try:
            if not sentences:
                return
            
            # Use the new pipeline processing method
            discord_bot = stream_data.get('discord_bot')
            
            # Debug logging
            self.logger.info(f"[SentenceStreaming] Discord bot debug: {type(discord_bot)} - {discord_bot is not None}")
            if discord_bot:
                self.logger.info(f"[SentenceStreaming] Discord bot has stream_sentences_to_discord: {hasattr(discord_bot, 'stream_sentences_to_discord')}")
            
            if discord_bot and hasattr(discord_bot, 'stream_sentences_to_discord'):
                stream_id = stream_data.get('stream_id', 'unknown')
                self.logger.info(f"[SentenceStreaming] Queued {len(sentences)} sentences for sequential processing")
                
                # Call the new pipeline processing method
                discord_bot.stream_sentences_to_discord(sentences, stream_id)
            else:
                self.logger.warning(f"[SentenceStreaming] Discord bot not available for pipeline processing")
                
        except Exception as e:
            self.logger.error(f"[SentenceStreaming] Error queuing sentences for Discord: {e}", exc_info=True)

# Utility functions for easy integration

def create_tts_callback(app_context) -> Callable[[str], None]:
    """Create a TTS callback function for sentence streaming with Chatterbox synchronization"""
    def tts_callback(sentence: str):
        try:
            if app_context.audio_service_instance and app_context.global_settings.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
                streaming_config = app_context.global_settings.get("STREAMING_RESPONSE", {})
                max_queue_size = streaming_config.get("max_sentence_queue_size", 3)
                
                # Initialize streaming queue with appropriate size for Chatterbox
                if not hasattr(app_context, 'streaming_tts_queue'):
                    app_context.streaming_tts_queue = Queue(maxsize=max_queue_size)
                
                # Check if queue is getting full (important for slow Chatterbox TTS)
                current_queue_size = app_context.streaming_tts_queue.qsize()
                if current_queue_size >= max_queue_size - 1:
                    logger = logging.getLogger("DanzarVLM.SentenceStreaming")
                    logger.warning(f"Streaming TTS queue nearly full ({current_queue_size}/{max_queue_size}), may need to slow down sentence generation")
                
                tts_audio = app_context.tts_service_instance.generate_audio(sentence) if app_context.tts_service_instance else None
                if tts_audio:
                    # Add to streaming queue with timeout for Chatterbox
                    try:
                        # For Chatterbox, use a timeout since it's slow
                        timeout = streaming_config.get("sentence_queue_timeout_s", 120)
                        if timeout > 0:
                            app_context.streaming_tts_queue.put(tts_audio, timeout=min(timeout, 30))
                        else:
                            app_context.streaming_tts_queue.put_nowait(tts_audio)
                        
                        logger = logging.getLogger("DanzarVLM.SentenceStreaming")
                        logger.debug(f"Queued sentence for TTS: '{sentence[:30]}...' (queue: {app_context.streaming_tts_queue.qsize()}/{max_queue_size})")
                        
                    except queue.Full:
                        # If streaming queue is full, drop oldest from main queue and add there
                        logger = logging.getLogger("DanzarVLM.SentenceStreaming")
                        logger.warning("Streaming TTS queue full, falling back to main TTS queue")
                        
                        # Clear space in main queue
                        try:
                            while app_context.tts_queue.qsize() >= 2:
                                app_context.tts_queue.get_nowait()
                                app_context.tts_queue.task_done()
                        except queue.Empty:
                            pass
                        
                        app_context.tts_queue.put_nowait(tts_audio)
        except Exception as e:
            logging.getLogger("DanzarVLM.SentenceStreaming").error(f"TTS callback error: {e}")
    
    return tts_callback

def create_text_callback(app_context) -> Callable[[str], None]:
    """Create a text callback function for sentence streaming"""
    def text_callback(sentence: str):
        try:
            # Add sentence to text message queue for Discord
            app_context.text_message_queue.put_nowait(f"📝 {sentence}")
        except Exception as e:
            logging.getLogger("DanzarVLM.SentenceStreaming").error(f"Text callback error: {e}")
    
    return text_callback 