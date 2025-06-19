# discord_integration/bot_client.py
import discord
from discord.ext import commands
import asyncio
import time
import queue # For queue.Empty
import io    # For io.BytesIO
import threading
import os    # Moved to top with other imports
import tempfile
import concurrent.futures
from typing import Optional, TYPE_CHECKING, List, Dict, Any, cast
# import gtts # Removed - using Chatterbox TTS instead
import re
import numpy as np
# Using py-cord's built-in voice recording instead of custom audio sink
import whisper  # For STT processing
from services.tts_service import TTSService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from core.config_loader import load_global_settings
from utils.general_utils import setup_logger

# Discord.py will handle Opus loading automatically
print("[DiscordBot] Using py-cord voice capabilities with recording")

if TYPE_CHECKING: # To avoid circular import for type hinting
    from ..DanzarVLM import AppContext # Adjust path if DanzarVLM.py is not in parent
    # from ..services.audio_service import AudioService # Not directly used here

class DiscordBot(commands.Bot):
    def __init__(self, 
                 command_prefix: str,
                 intents: discord.Intents,
                 stt_service,  # STTService type
                 tts_service: TTSService,
                 llm_service: LLMService,
                 memory_service: MemoryService):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.logger = setup_logger(__name__)
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.voice_client: Optional[discord.VoiceClient] = None
        self.is_processing = False
        self.current_text_channel: Optional[discord.TextChannel] = None
        self.settings = load_global_settings() or {}
        self.tts_is_playing = asyncio.Event()
        self.auto_leave_if_alone_timeout_s = float(self.settings.get("DISCORD_AUTO_LEAVE_TIMEOUT_S", 60.0))
        
        # Voice recording state
        self.connections = {}  # Guild ID -> VoiceClient mapping for recording
        self.whisper_model = None  # Will be loaded in setup_hook
        
        self.logger.info("[DiscordBot] Bot initialized with voice capabilities")

        # Debug: Log all methods that have the commands.command decorator
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__commands_is_command__'):
                self.logger.info(f"[DiscordBot] Found command method: {attr_name}")

    async def setup_hook(self):
        """Setup hook called after login but before connecting to gateway."""
        self.logger.info("🔧 Loading Whisper model in setup_hook...")
        try:
            # Load Whisper model asynchronously to avoid blocking
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None, 
                whisper.load_model, 
                "tiny"  # Use tiny model for faster processing
            )
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            self.whisper_model = None

    async def on_ready(self):
        """Called when bot is ready."""
        self.logger.info(f"🎤 {self.__class__.__name__} ready as {self.user}")
        self.logger.info(f"🎤 Available commands: {', '.join([f'!{cmd.name}' for cmd in self.commands])}")

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM responses for Discord"""
        if not text:
            return text
        
        # Log the original response with think tags for debugging
        if '<think>' in text.lower():
            think_content = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
            if think_content:
                self.logger.info(f"[DiscordBot] Model thinking process: {think_content.group(1).strip()[:100]}...")
        
        # Remove think tags and their content, keeping only what comes after
        # Use DOTALL flag to handle multi-line thinking content
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        # If response is empty after stripping think tags, provide fallback
        if not clean_text and text.strip():
            clean_text = "I'm thinking about that... let me get back to you."
            self.logger.warning("[DiscordBot] Response was empty after stripping think tags, using fallback")
        
        return clean_text

    # Voice Recording Commands - Following Pycord Guide Pattern
    @commands.command(name='join')
    async def join_command(self, ctx):
        """Join voice channel and start recording."""
        self.logger.info(f"📞 !join used by {ctx.author.name}")
        
        voice = ctx.author.voice
        if not voice:
            await ctx.respond("❌ You aren't in a voice channel!")
            return
        
        self.logger.info(f"🎯 Target channel: {voice.channel.name}")
        self.logger.info(f"🔗 Attempting connection to {voice.channel.name}")
        
        try:
            # Connect to voice channel with proper parameters
            vc = await voice.channel.connect(timeout=10.0, reconnect=True)
            self.connections[ctx.guild.id] = vc
            self.current_text_channel = ctx.channel
            
            self.logger.info(f"✅ Successfully connected to {voice.channel.name}")
            
            # Start recording using py-cord's built-in recording
            vc.start_recording(
                discord.sinks.WaveSink(),  # Use WaveSink for recording
                self.recording_finished,   # Callback when recording stops
                ctx.channel              # Pass channel for responses
            )
            
            self.logger.info(f"🎙️ Recording started in {voice.channel.name}")
            await ctx.respond(f"🎙️ **Joined {voice.channel.name}** - Recording started!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to join voice channel: {e}")
            await ctx.respond(f"❌ Failed to join voice channel: {str(e)}")

    @commands.command(name='leave')
    async def leave_command(self, ctx):
        """Stop recording and leave voice channel."""
        self.logger.info(f"🛑 !leave used by {ctx.author.name}")
        
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            self.logger.info("🛑 Recording stopped")
            
            # Stop recording (this will trigger the callback)
            vc.stop_recording()
            
            # Clean up connection
            del self.connections[ctx.guild.id]
            await ctx.message.delete()  # Clean up command message
            
        else:
            await ctx.respond("❌ I am currently not recording here.")

    @commands.command(name='test')
    async def test_command(self, ctx):
        """Test command to verify bot is working."""
        self.logger.info(f"🧪 !test used by {ctx.author.name}")
        await ctx.respond("🧪 **DanzarAI Voice Bot is working!** Use `!join` to start voice recording.")

    async def recording_finished(self, sink: discord.sinks.WaveSink, channel: discord.TextChannel, *args):
        """Callback when recording is finished - processes all recorded audio."""
        self.logger.info("🎵 Processing recorded audio...")
        
        try:
            # Disconnect from voice
            await sink.vc.disconnect()
            self.logger.info("✅ Successfully disconnected from voice")
            
            # Process each user's audio
            if sink.audio_data:
                for user_id, audio in sink.audio_data.items():
                    try:
                        # Get user info
                        user = self.get_user(user_id)
                        user_name = user.display_name if user else f"User_{user_id}"
                        
                        # Check if audio file has content
                        audio_file = audio.file
                        if os.path.getsize(audio_file.name) > 1024:  # At least 1KB
                            self.logger.info(f"🎤 Processing audio from {user_name} ({os.path.getsize(audio_file.name)} bytes)")
                            
                            # Process audio with Whisper
                            transcription = await self.process_audio_with_whisper(audio_file.name)
                            
                            if transcription and transcription.strip():
                                self.logger.info(f"📝 Transcription from {user_name}: {transcription}")
                                
                                # Send transcription to channel
                                await channel.send(f"🎤 **{user_name}**: {transcription}")
                                
                                # Process with LLM if available
                                if self.llm_service:
                                    await self.handle_llm_query_and_respond(transcription, user_name, channel)
                            else:
                                self.logger.info(f"🔇 No speech detected from {user_name}")
                        else:
                            self.logger.info(f"🔇 Audio file too small from {user_name}, skipping")
                            
                        # Clean up temporary file
                        try:
                            os.unlink(audio_file.name)
                        except:
                            pass
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error processing audio for user {user_id}: {e}")
            else:
                self.logger.info("🔇 No audio data recorded")
                
        except Exception as e:
            self.logger.error(f"❌ Error in recording_finished callback: {e}")

    async def process_audio_with_whisper(self, audio_file_path: str) -> Optional[str]:
        """Process audio file with Whisper STT."""
        if not self.whisper_model:
            self.logger.error("❌ Whisper model not loaded")
            return None
            
        try:
            # Run Whisper transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.whisper_model.transcribe,
                audio_file_path
            )
            
            if result and "text" in result:
                text = result["text"].strip()
                return text if text else None
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Whisper transcription error: {e}")
            return None

    async def handle_llm_query_and_respond(self, user_query: str, user_name: str, channel):
        """Handle LLM query and send response back to Discord channel with sentence streaming"""
        try:
            if self.llm_service:
                response = await self.llm_service.handle_user_text_query(user_query, user_name)
                if response is not None and isinstance(response, str) and response.strip():
                    # Strip think tags before sending to Discord or TTS
                    clean_response = self._strip_think_tags(response)
                    
                    if clean_response.strip():
                        # Check if sentence streaming is enabled for Discord text
                        streaming_config = self.memory_service.global_settings.get("STREAMING_RESPONSE", {})
                        enable_text_streaming = streaming_config.get("enable_text_streaming", False)
                        
                        if enable_text_streaming and self.memory_service.streaming_response_instance:
                            # Use sentence streaming for Discord text
                            await self._send_streamed_discord_response(clean_response, user_query, user_name, channel)
                        else:
                            # Traditional single message (with chunking if needed)
                            await self._send_traditional_discord_response(clean_response, channel)
                    else:
                        await channel.send("I'm processing your request... please wait a moment.")
                elif response is None:
                    # Response already handled inside LLM service (text + TTS), nothing to do.
                    pass
                else:
                    await channel.send("Sorry, I couldn't process that request right now.")
            else:
                await channel.send("Sorry, my LLM service is not available right now.")
        except Exception as e:
            self.logger.error(f"Error handling LLM query: {e}", exc_info=True)
            # Provide more specific error messages based on the exception
            if "content" in str(e).lower() and "length" in str(e).lower():
                await channel.send("Sorry, my response was too long. Could you ask a more specific question?")
            else:
                await channel.send("Sorry, I encountered an error processing your request.")

    async def _send_streamed_discord_response(self, response: str, user_query: str, user_name: str, channel):
        """Send response using NEW sequential processing system (OLD streaming method disabled)"""
        try:
            # NEW: Use sequential processing instead of old streaming system
            streaming_service = self.memory_service.streaming_response_instance
            if not streaming_service:
                await self._send_traditional_discord_response(response, channel)
                return
            
            # Start a streaming session that will queue sentences for sequential processing
            stream_id = streaming_service.start_response_stream(
                user_text=user_query,
                user_name=user_name,
                tts_callback=None,  # Sequential system handles TTS
                text_callback=None  # Sequential system handles Discord text
            )
            
            if stream_id:
                # This will queue sentences in the sequential processing queue
                streaming_service.stream_complete_response(stream_id, response)
                self.logger.info(f"[DiscordBot] Queued response for sequential processing via stream {stream_id}")
            else:
                # Fallback to traditional response
                self.logger.warning("[DiscordBot] Failed to start streaming session, using traditional response")
                await self._send_traditional_discord_response(response, channel)
                
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in sequential Discord response: {e}")
            await self._send_traditional_discord_response(response, channel)

    async def _send_traditional_discord_response(self, response: str, channel):
        """Send response to Discord as traditional single/chunked messages *with* optional TTS playback"""
        try:
            # Ensure we're connected to the target voice channel (for TTS)
            if self.memory_service.global_settings.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
                try:
                    await self._join_target_voice_channel()
                except Exception as vc_err:
                    self.logger.warning(f"[DiscordBot] Could not join voice before TTS playback: {vc_err}")

            # Discord has a 2000 character limit for messages
            if len(response) > 1900:  # Leave some buffer
                # Split into chunks for very long responses
                chunks = []
                remaining = response.strip()
                while remaining:
                    if len(remaining) <= 1900:
                        chunks.append(remaining)
                        break

                    # Find a reasonable break-point (prefer sentence boundary)
                    break_point = 1900
                    for i in range(1800, min(1900, len(remaining))):
                        if remaining[i] in '.!?':
                            break_point = i + 1
                            break
                    chunks.append(remaining[:break_point])
                    remaining = remaining[break_point:].strip()
            else:
                chunks = [response]

            for idx, chunk in enumerate(chunks):
                msg_to_send = chunk if len(chunks) == 1 else f"**[Part {idx+1}/{len(chunks)}]**\n{chunk}"
                await channel.send(msg_to_send)

                # TTS playback for each chunk (optional)
                if self.memory_service.global_settings.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
                    try:
                        # Use correct TTS service and method
                        if not self.memory_service.tts_service_instance:
                            self.logger.warning("[DiscordBot] TTS service not available for chat reply")
                            continue
                        tts_audio = self.memory_service.tts_service_instance.generate_audio(chunk)
                        if tts_audio:
                            # Ensure voice connection only when we actually have audio
                            try:
                                await self._join_target_voice_channel()
                            except Exception as vc_err:
                                self.logger.warning(f"[DiscordBot] Could not join voice before TTS playback: {vc_err}")

                            # Run blocking playback in executor so we don't block Discord event loop
                            await self.loop.run_in_executor(
                                None,
                                self._play_tts_audio_sync,
                                tts_audio,
                                f"trad_{int(time.time())}_{idx}"
                            )
                        else:
                            self.logger.warning("[DiscordBot] No TTS audio generated for traditional chunk; skipping voice playback")
                    except Exception as tts_err:
                        self.logger.error(f"[DiscordBot] TTS playback failed for chunk: {tts_err}")

                # Small delay between chunks (text sent already) to avoid rate-limits
                if idx < len(chunks) - 1:
                    await asyncio.sleep(0.5)

        except Exception as e:
            self.logger.error(f"[DiscordBot] Error sending traditional Discord response: {e}")
            try:
                await channel.send("Sorry, I encountered an error sending my response.")
            except Exception:
                pass  # suppress secondary errors

    async def _join_target_voice_channel(self) -> Optional[discord.VoiceClient]:
        guild_id = self.settings.get("DISCORD_GUILD_ID")
        voice_channel_id = self.settings.get("DISCORD_VOICE_CHANNEL_ID")
        
        if not guild_id or not voice_channel_id:
            self.logger.error("Missing guild ID or voice channel ID in settings")
            return None
            
        guild = self.get_guild(int(guild_id))
        if not guild:
            self.logger.error(f"Target guild {guild_id} not found")
            return None
            
        voice_channel = guild.get_channel(int(voice_channel_id))
        if not voice_channel or not isinstance(voice_channel, discord.VoiceChannel):
            self.logger.error(f"Target voice channel {voice_channel_id} not found or invalid")
            return None

        current_vc = self.voice_client
        
        # Check if we have a valid connection to the target channel
        if current_vc and current_vc.is_connected():
            if current_vc.channel.id == voice_channel.id:
                self.logger.info(f"Already connected to target channel '{voice_channel.name}'")
                return current_vc
            else:
                self.logger.info(f"Connected to different channel. Disconnecting from '{current_vc.channel.name}'")
                try:
                    await current_vc.disconnect(force=True)
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during voice cleanup: {cleanup_error}")
                finally:
                    self.voice_client = None
            
            # Wait a bit more to ensure Discord has processed the disconnect
            await asyncio.sleep(0.5)
        
        try:
            self.logger.info(f"Connecting to voice channel '{voice_channel.name}'...")
            new_vc = await voice_channel.connect(
                timeout=10.0,
                reconnect=True,
                self_deaf=True,  # Bot doesn't need to hear itself
                self_mute=False  # Bot should be able to speak
            )
            
            self.voice_client = new_vc
            self.logger.info(f"Successfully connected to '{voice_channel.name}' (session: {new_vc.session_id})")
            
            return new_vc
            
        except Exception as e:
            self.logger.error(f"Failed to connect to voice channel: {e}")
            # Cleanup on failure
            self.voice_client = None
            return None

    def _start_recording(self, voice_client):
        """
        Start recording with discord.py compatibility layer (no actual voice input)
        """
        try:
            if self.is_starting_recording:
                self.logger.debug("[DiscordBot] Already starting recording, skipping.")
                return
            self.is_starting_recording = True
            
            # DISABLED: Audio recording not needed for TTS-only bot
            self.logger.info(f"[DiscordBot] Voice recording disabled for TTS-only operation in '{voice_client.channel.name}'.")
                
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error starting voice recording: {e}", exc_info=True)
        finally:
            self.is_starting_recording = False

    def _ensure_recording_is_active(self, voice_client):
        if voice_client and voice_client.is_connected():
            # DISABLED: Audio recording not needed for TTS-only bot
            self.logger.debug(f"[DiscordBot] Voice recording disabled for TTS-only operation in '{voice_client.channel.name}'.")

    async def _auto_leave_if_alone(self, voice_client, channel):
        try:
            await asyncio.sleep(self.auto_leave_if_alone_timeout_s)
            # Re-check conditions before leaving
            if voice_client.is_connected() and voice_client.channel and voice_client.channel.id == channel.id:
                if sum(1 for m in channel.members if not m.bot) == 0:
                    self.logger.info(f"[DiscordBot] Auto-leave timer expired. Still alone in '{channel.name}'. Disconnecting.")
                    await voice_client.disconnect(force=False) 
                    self.voice_client = None
                else:
                    self.logger.info(f"[DiscordBot] Auto-leave timer expired, but users are now in '{channel.name}'. Staying connected.")
            else:
                self.logger.debug("[DiscordBot] Auto-leave timer expired, but bot's VC state changed or not in the original channel anymore.")
        except asyncio.CancelledError:
            self.logger.info(f"[DiscordBot] Auto-leave timer for '{channel.name}' was cancelled.")
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in _auto_leave_if_alone for '{channel.name}': {e}", exc_info=True)
        finally:
            if self.auto_leave_timer_task is asyncio.current_task(): # Clear if this task instance is the one stored
                self.auto_leave_timer_task = None

    def _after_tts_playback(self, error):
        if error: 
            self.logger.error(f"[DiscordBotPlayback] Error during TTS playback: {error}")
        else: 
            self.logger.debug("[DiscordBotPlayback] TTS playback finished.")
        
        # Clear the playing flag to allow next sentence
        self.memory_service.tts_is_playing.clear()
        
        # Add a small delay before next sentence for Chatterbox streaming
        streaming_config = self.memory_service.global_settings.get("STREAMING_RESPONSE", {})
        if streaming_config.get("wait_for_sentence_completion", True):
            sentence_delay = streaming_config.get("sentence_delay_ms", 500)
            if sentence_delay > 0:
                # Use a timer to add the delay without blocking
                import threading
                threading.Timer(sentence_delay / 1000.0, lambda: None).start()
    
    def _after_tts_playback_with_cleanup(self, error, temp_file_path):
        """TTS playback callback that also cleans up temporary files"""
        self._after_tts_playback(error)
        
        # Check if voice connection is still healthy after playback
        voice_client = self.voice_client
        if voice_client and not voice_client.is_connected():
            self.logger.warning("[DiscordBotPlayback] Voice connection lost during TTS playback")
            # Clear the reference to the disconnected client
            self.voice_client = None
        
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                self.logger.debug(f"[DiscordBotPlayback] Cleaned up temp file: {temp_file_path}")
        except Exception as cleanup_error:
            self.logger.warning(f"[DiscordBotPlayback] Failed to cleanup temp file {temp_file_path}: {cleanup_error}")

    def run_playback_loop(self):
        """Disabled - now using queue-based sentence pairing system"""
        self.logger.info("[DiscordBot] Old pipeline processing loop disabled - using new queue-based system.")
        
        # Keep the thread alive but do nothing - new system handles everything
        while not self.memory_service.shutdown_event.is_set():
            time.sleep(1)
        
        self.logger.info("[DiscordBot] Pipeline processing loop stopped.")

    def run_text_message_loop(self):
        """Handle text messages for backward compatibility"""
        self.logger.info("[DiscordBot] Text Message Loop (minimal for backward compatibility).")
        
        while not self.memory_service.shutdown_event.is_set():
            try:
                if not self.memory_service.text_message_queue.empty():
                    try:
                        text_msg = self.memory_service.text_message_queue.get_nowait()
                        self._send_discord_text_message(text_msg)
                        self.memory_service.text_message_queue.task_done()
                    except queue.Empty:
                        pass
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"[DiscordBot] Error in text message loop: {e}", exc_info=True)
                time.sleep(1)
        
        self.logger.info("[DiscordBot] Text Message Loop stopped.")

    def _process_sentences_pipeline(self, sentences, stream_id):
        """Process sentences in pairs with proper queue system"""
        try:
            batch_size = 2  # Always process 2 sentences at a time
            
            # Split sentences into pairs
            sentence_pairs = []
            for i in range(0, len(sentences), batch_size):
                pair = sentences[i:i + batch_size]
                sentence_pairs.append(pair)
            
            self.logger.info(f"[DiscordBot] Processing {len(sentences)} sentences in {len(sentence_pairs)} pairs")
            
            # Queue to store completed sentence pairs with their TTS audio
            completed_pairs_queue = queue.Queue()
            
            # Start processing pairs
            for pair_idx, sentence_pair in enumerate(sentence_pairs):
                pair_id = f"pair_{int(time.time())}_{pair_idx}"
                
                # Start TTS generation for this pair in background
                tts_thread = threading.Thread(
                    target=self._generate_tts_for_pair,
                    args=(pair_id, sentence_pair, completed_pairs_queue),
                    daemon=True
                )
                tts_thread.start()
                
                # If this is not the first pair, play the previous completed pair
                if pair_idx > 0:
                    self._play_completed_pair_from_queue(completed_pairs_queue)
            
            # Play any remaining pairs in the queue
            while not completed_pairs_queue.empty() or any(t.is_alive() for t in threading.enumerate() if t.name.startswith('Thread')):
                self._play_completed_pair_from_queue(completed_pairs_queue)
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            self.logger.info(f"[DiscordBot] Completed processing all sentence pairs")
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in pipeline processing: {e}", exc_info=True)

    def _generate_tts_for_pair(self, pair_id, sentence_pair, completed_queue):
        """Generate TTS for a sentence pair and add to completed queue"""
        try:
            self.logger.info(f"[DiscordBot] Generating TTS for {pair_id} with {len(sentence_pair)} sentences")
            
            # Generate TTS for each sentence in the pair
            pair_data = {
                'pair_id': pair_id,
                'sentences': [],
                'tts_audio': [],
                'generation_time': time.time()
            }
            
            for sentence_idx, sentence in enumerate(sentence_pair):
                start_time = time.time()
                clean_sentence = sentence.strip()
                
                # Generate TTS audio
                if not self.memory_service.tts_service_instance:
                    self.logger.warning(f"[DiscordBot] TTS service not available for {pair_id}_sentence_{sentence_idx}")
                    tts_audio = None
                else:
                    tts_audio = self.memory_service.tts_service_instance.generate_audio(clean_sentence)
                generation_time = time.time() - start_time
                
                if tts_audio:
                    self.logger.info(f"[DiscordBot] TTS generated for {pair_id}_sentence_{sentence_idx} in {generation_time:.1f}s ({len(tts_audio)} bytes)")
                    pair_data['sentences'].append(clean_sentence)
                    pair_data['tts_audio'].append(tts_audio)
                else:
                    self.logger.warning(f"[DiscordBot] TTS generation failed for {pair_id}_sentence_{sentence_idx}")
                    # Still add the sentence but with None audio to maintain pairing
                    pair_data['sentences'].append(clean_sentence)
                    pair_data['tts_audio'].append(None)
            
            # Add completed pair to queue
            total_time = time.time() - pair_data['generation_time']
            self.logger.info(f"[DiscordBot] Completed TTS generation for {pair_id} in {total_time:.1f}s")
            completed_queue.put(pair_data)
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error generating TTS for {pair_id}: {e}")
            # Add failed pair to queue to maintain order
            completed_queue.put({
                'pair_id': pair_id,
                'sentences': sentence_pair,
                'tts_audio': [None] * len(sentence_pair),
                'generation_time': time.time()
            })

    def _play_completed_pair_from_queue(self, completed_queue, timeout=60):
        """Play a completed sentence pair from the queue"""
        try:
            # Wait for a completed pair with timeout
            try:
                pair_data = completed_queue.get(timeout=timeout)
            except queue.Empty:
                self.logger.warning(f"[DiscordBot] Timeout waiting for completed pair")
                return
            
            pair_id = pair_data['pair_id']
            sentences = pair_data['sentences']
            tts_audios = pair_data['tts_audio']
            
            self.logger.info(f"[DiscordBot] Playing pair {pair_id} with {len(sentences)} sentences")
            
            # Play each sentence in the pair sequentially
            for sentence_idx, (sentence, tts_audio) in enumerate(zip(sentences, tts_audios)):
                self.logger.info(f"[DiscordBot] Playing sentence {sentence_idx + 1}/{len(sentences)} from {pair_id}: '{sentence[:50]}...'")
                
                # Send Discord message first
                start_time = time.time()
                self._send_discord_text_message(f"💬 {sentence}")
                discord_time = time.time() - start_time
                self.logger.info(f"[DiscordBot] Discord message sent in {discord_time:.1f}s")
                
                # Play TTS audio if available
                if tts_audio:
                    self._play_tts_audio_sync(tts_audio, f"{pair_id}_sentence_{sentence_idx}")
                else:
                    self.logger.warning(f"[DiscordBot] No TTS audio for {pair_id}_sentence_{sentence_idx}")
                
                self.logger.info(f"[DiscordBot] Completed sentence {sentence_idx + 1}/{len(sentences)} from {pair_id}")
            
            self.logger.info(f"[DiscordBot] Completed playing pair {pair_id}")
            completed_queue.task_done()
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error playing completed pair: {e}")

    def _play_tts_audio_sync(self, tts_audio, sentence_key):
        """Play TTS audio synchronously with proper error handling and timeout safeguards"""
        voice_client = None
        try:
            # --- SAFETY: Skip tiny/empty audio blobs that would keep the bot 'speaking' forever ---
            min_audio_size = 1024  # bytes – empirically, <1 KB means header-only / silent
            if not tts_audio or len(tts_audio) < min_audio_size:
                self.logger.warning(
                    f"[DiscordBot] Skipping playback for {sentence_key}: audio blob too small (size={len(tts_audio) if tts_audio else 0} bytes)"
                )
                return
            
            # Check voice connection and attempt to reconnect if needed
            voice_client = self.voice_client
            if not voice_client or not voice_client.is_connected():
                self.logger.info(f"[DiscordBot] Voice client not connected, attempting to join for TTS playback")
                # Try to reconnect for TTS
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    voice_client = loop.run_until_complete(self._join_target_voice_channel())
                    if not voice_client:
                        self.logger.warning(f"[DiscordBot] Failed to reconnect for TTS playback")
                        return
                except Exception as reconnect_error:
                    self.logger.error(f"[DiscordBot] Error reconnecting for TTS: {reconnect_error}")
                    return
            
            # Double-check voice connection health before playing
            if not voice_client.is_connected():
                self.logger.warning(f"[DiscordBot] Voice client lost connection during TTS setup")
                return

            # Create temporary file for audio
            temp_file = f"temp_tts_{sentence_key.replace('/', '_')}.wav"
            try:
                with open(temp_file, 'wb') as f:
                    f.write(tts_audio)

                # Calculate expected duration with more generous timeout for long TTS
                import wave, contextlib
                duration_sec = 0.0
                try:
                    with contextlib.closing(wave.open(temp_file, 'rb')) as wf:
                        frames = wf.getnframes(); rate = wf.getframerate()
                        duration_sec = frames / float(rate) if rate else 0.0
                except Exception:
                    pass
                # More generous timeout: actual duration + 10 seconds buffer, max 120 seconds
                max_wait = max(10.0, min(duration_sec + 10.0, 120.0))

                start_time = time.time()
                self.logger.info(
                    f"[DiscordBot] Playing TTS for {sentence_key} (size={len(tts_audio)} bytes, est_dur={duration_sec:.2f}s, timeout={max_wait:.1f}s)"
                )

                # Set speaking state flag and cancel any auto-leave timer
                self.memory_service.tts_is_playing.set()
                
                # Cancel auto-leave timer during TTS playback
                if self.auto_leave_timer_task and not self.auto_leave_timer_task.done():
                    self.auto_leave_timer_task.cancel()
                    self.auto_leave_timer_task = None
                    self.logger.info(f"[DiscordBot] Cancelled auto-leave timer for TTS playback")

                # Play audio using FFmpeg
                audio_source = discord.FFmpegPCMAudio(temp_file)
                voice_client.play(audio_source, after=lambda e: self._after_tts_playback_with_cleanup(e, temp_file))

                # Wait for playback to complete with timeout, but yield control frequently
                # to prevent blocking Discord's voice heartbeat
                while voice_client.is_playing() and (time.time() - start_time) < max_wait:
                    time.sleep(0.05)  # Shorter sleep to yield control more frequently

                if voice_client.is_playing():
                    self.logger.warning(
                        f"[DiscordBot] Playback timeout reached for {sentence_key} after {max_wait:.1f}s – forcing stop()"
                    )
                    voice_client.stop()
                    # Wait a bit for the stop to take effect
                    time.sleep(0.5)

                # Ensure speaking state is cleared
                self.memory_service.tts_is_playing.clear()
                
                # Force speaking state to false and stop any lingering audio
                if hasattr(voice_client, '_player'):
                    voice_client._player = None
                voice_client.stop()
                
                playback_time = time.time() - start_time
                self.logger.info(
                    f"[DiscordBot] TTS playback completed for {sentence_key} in {playback_time:.1f}s"
                )

            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass  # Ignore cleanup errors

                # Final cleanup of voice state
                if voice_client and voice_client.is_connected():
                    voice_client.stop()
                    if hasattr(voice_client, '_player'):
                        voice_client._player = None
                self.memory_service.tts_is_playing.clear()

        except Exception as e:
            self.logger.error(f"[DiscordBot] Error playing TTS audio for {sentence_key}: {e}")
            # Ensure cleanup on error
            self.memory_service.tts_is_playing.clear()
            if voice_client and voice_client.is_connected():
                voice_client.stop()

    def _send_discord_text_message(self, message_text: str):
        """Send a text message to Discord channel synchronously"""
        try:
            text_channel_id = self.memory_service.global_settings.get("DISCORD_TEXT_CHANNEL_ID")
            if text_channel_id and self:
                channel = self.get_channel(text_channel_id)
                if channel and isinstance(channel, discord.TextChannel):
                    # Use asyncio to send message synchronously with longer timeout
                    future = asyncio.run_coroutine_threadsafe(
                        channel.send(message_text),
                        self.loop
                    )
                    try:
                        future.result(timeout=30)  # Increased timeout to 30 seconds
                        self.logger.debug(f"[DiscordBot] Text message sent to channel {text_channel_id}")
                    except concurrent.futures.TimeoutError:
                        self.logger.warning(f"[DiscordBot] Discord text message timeout after 30s - continuing anyway")
                        # Don't raise the error, just continue processing
                else:
                    self.logger.warning(f"[DiscordBot] Text channel {text_channel_id} not found.")
            else:
                self.logger.warning("[DiscordBot] DISCORD_TEXT_CHANNEL_ID not set or bot not ready.")
        except Exception as e:
            self.logger.warning(f"[DiscordBot] Error sending Discord text message (continuing): {e}")
            # Don't raise the error to prevent breaking the TTS pipeline

    def stream_sentences_to_discord(self, sentences, stream_id):
        """SIMPLIFIED: Use traditional response instead of complex pipeline"""
        try:
            if not sentences:
                self.logger.warning(f"[DiscordBot] No sentences to stream for {stream_id}")
                return
            
            # SIMPLIFIED: Just send as one traditional response instead of complex pipeline
            full_response = " ".join(sentences)
            self.logger.info(f"[DiscordBot] Sending traditional response instead of pipeline processing")
            
            # Send to Discord text channel
            self._send_discord_text_message(f"💬 {full_response}")
            
            # Generate and play TTS for full response
            if self.memory_service.tts_service_instance:
                try:
                    tts_audio = self.memory_service.tts_service_instance.generate_audio(full_response)
                    if tts_audio:
                        self._play_tts_audio_sync(tts_audio, f"traditional_{stream_id}")
                    else:
                        self.logger.warning(f"[DiscordBot] No TTS audio generated for response")
                except Exception as e:
                    self.logger.error(f"[DiscordBot] TTS generation failed: {e}")
            
            self.logger.info(f"[DiscordBot] Completed traditional response processing")
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in traditional response: {e}", exc_info=True)

    def run(self):
        token = self.settings.get("DISCORD_BOT_TOKEN")
        if not token: 
            self.logger.critical("[DiscordBot] DISCORD_BOT_TOKEN missing. Bot cannot run.")
            return
        try:
            self.logger.info("[DiscordBot] Starting Discord bot client...")
            
            # Use discord.py's built-in run method which handles event loop management
            self.run(token, reconnect=True)
            
        except discord.LoginFailure:
            self.logger.critical("[DiscordBot] Login failed: Invalid Discord token provided.")
        except Exception as e:
            self.logger.critical(f"[DiscordBot] Critical error running bot: {e}", exc_info=True)
        finally:
            self.logger.info("[DiscordBot] Bot thread exiting.")

    def on_voice_activity(self, is_speaking: bool, speech_ended: bool) -> None:
        """Handle voice activity detection events."""
        try:
            self.logger.info(f"[DiscordBot] Voice activity detected: is_speaking={is_speaking}, speech_ended={speech_ended}")
            
            if is_speaking and not self.is_processing:
                self.logger.info("[DiscordBot] Starting to process speech")
                self.is_processing = True
                asyncio.run_coroutine_threadsafe(self.handle_voice_command(), self.loop)
            elif speech_ended and self.is_processing:
                self.logger.info("[DiscordBot] Speech ended, finishing processing")
                self.is_processing = False
                
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in voice activity handler: {e}", exc_info=True)
            self.is_processing = False

    async def handle_voice_command(self) -> Optional[str]:
        """Process voice command audio and return command text."""
        try:
            self.logger.info("[DiscordBot] Converting audio for STT...")
            # Get audio data from sink
            if self.audio_sink:
                self.logger.info("[DiscordBot] Got audio sink, retrieving audio data...")
                audio_data = self.audio_sink.get_audio_data()
                if audio_data is not None:
                    self.logger.info(f"[DiscordBot] Got audio data, shape: {audio_data.shape}")
                    # Get transcription
                    self.logger.info("[DiscordBot] Sending audio to STT service...")
                    transcription = await self.stt_service.transcribe_audio(audio_data)
                    
                    if transcription and transcription.strip():
                        self.logger.info(f"[DiscordBot] Transcription: {transcription}")
                        return transcription
                    else:
                        self.logger.info("[DiscordBot] No transcription returned")
                        return None
                else:
                    self.logger.warning("[DiscordBot] No audio data available")
                    return None
            else:
                self.logger.warning("[DiscordBot] No audio sink available")
                return None
                
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error processing voice command: {e}", exc_info=True)
            return None

    async def join_voice_channel(self, channel: discord.VoiceChannel) -> None:
        """Join a voice channel and set up audio processing."""
        try:
            self.logger.info(f"[DiscordBot] Attempting to join voice channel: {channel.name}")
            
            if self.voice_client and self.voice_client.is_connected():
                self.logger.info("[DiscordBot] Disconnecting from current voice client")
                await self.voice_client.disconnect()
            
            self.logger.info("[DiscordBot] Connecting to new voice channel...")
            self.voice_client = await channel.connect()
            self.logger.info(f"[DiscordBot] Successfully connected to voice channel: {channel.name}")
            
            # Set up audio sink with debug logging
            self.logger.info("[DiscordBot] Initializing audio sink...")
            self.audio_sink = VoiceAudioSink(
                vad_callback=self.on_voice_activity,
                sample_rate=48000,  # Discord's sample rate
                channels=2,  # Discord's channel count
                buffer_size=960,  # 20ms at 48kHz
                vad_threshold=0.3,  # Lower threshold for better sensitivity
                vad_trigger_level=2,  # Lower trigger level
                vad_hold_time=0.3  # Shorter hold time
            )
            
            # Start recording immediately
            self.logger.info("[DiscordBot] Starting voice recording...")
            self.audio_sink.start_recording()
            
            # Set up voice receive with explicit logging
            self.logger.info("[DiscordBot] Setting up voice receive...")
            self.voice_client.start_recording(
                self.audio_sink,
                self.on_voice_data,
                self.on_voice_error
            )
            
            # Enable voice receive
            self.voice_client.decoder.enabled = True
            self.voice_client.recording = True
            
            # Log voice client state
            self.logger.info(f"[DiscordBot] Voice client state: connected={self.voice_client.is_connected()}, "
                           f"recording={self.voice_client.recording}, "
                           f"speaking={self.voice_client.speaking}, "
                           f"decoder_enabled={self.voice_client.decoder.enabled}")
            
            # Store references
            self.current_text_channel = channel.guild.text_channels[0]  # Use first text channel for now
            
            self.logger.info("[DiscordBot] Voice setup complete")
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error joining voice channel: {e}", exc_info=True)
            raise

    async def on_voice_data(self, data: bytes) -> None:
        """Handle incoming voice data."""
        try:
            if not self.audio_sink:
                self.logger.warning("[DiscordBot] Received voice data but audio sink is not initialized")
                return
                
            # Log audio data details
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.logger.info(f"[DiscordBot] Received voice data: size={len(data)}, "
                           f"shape={audio_data.shape}, "
                           f"min={audio_data.min()}, "
                           f"max={audio_data.max()}, "
                           f"mean={audio_data.mean():.2f}")
            
            # Process the audio data
            self.audio_sink.write(data)
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error processing voice data: {e}", exc_info=True)

    def on_voice_error(self, error):
        """Handle voice error events."""
        try:
            self.logger.error(f"[DiscordBot] Voice error detected: {error}")
            
        except Exception as e:
            self.logger.error(f"[DiscordBot] Error in voice error handler: {e}", exc_info=True)



    @commands.command(name='leave')
    async def leave_command(self, ctx):
        """Leave the current voice channel."""
        try:
            if ctx.voice_client:
                channel_name = ctx.voice_client.channel.name
                await ctx.voice_client.disconnect()
                self.voice_client = None
                self.audio_sink = None
                await ctx.send(f"✅ Left voice channel: **{channel_name}**")
                self.logger.info(f"[DiscordBot] Left voice channel: {channel_name}")
            else:
                await ctx.send("ℹ️ I'm not in any voice channel!")
        except Exception as e:
            self.logger.error(f"Error leaving voice channel: {e}", exc_info=True)
            await ctx.send("❌ Failed to leave voice channel!")

    @commands.command(name='status')
    async def status_command(self, ctx):
        """Show bot status and current voice channel."""
        try:
            if ctx.voice_client and ctx.voice_client.channel:
                channel_name = ctx.voice_client.channel.name
                member_count = len(ctx.voice_client.channel.members)
                await ctx.send(f"🎤 **Voice Status:**\n"
                             f"• Channel: **{channel_name}**\n"
                             f"• Members: **{member_count}**\n"
                             f"• Voice processing: **{'Active' if self.audio_sink else 'Inactive'}**")
            else:
                await ctx.send("🔇 **Voice Status:** Not connected to any voice channel")
        except Exception as e:
            self.logger.error(f"Error getting status: {e}", exc_info=True)
            await ctx.send("❌ Failed to get status!")

    async def _setup_voice_processing(self, voice_client: discord.VoiceClient):
        """Setup voice activity detection and audio processing."""
        try:
            self.logger.info("[DiscordBot] Setting up voice processing...")
            
            # Create audio sink for voice activity detection
            self.audio_sink = VoiceAudioSink(
                sample_rate=48000,
                channels=2,
                buffer_size=960,
                vad_threshold=0.3,
                vad_trigger_level=2,
                vad_hold_time=0.3
            )
            
            # Set up callbacks
            self.audio_sink.on_voice_activity = self.on_voice_activity
            self.audio_sink.on_voice_data = self.on_voice_data
            self.audio_sink.on_voice_error = self.on_voice_error
            
            # Start recording
            voice_client.start_recording(
                self.audio_sink,
                self._after_recording,
                self._recording_error
            )
            
            self.logger.info("[DiscordBot] Voice processing setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up voice processing: {e}", exc_info=True)

    def _after_recording(self, sink, channel, *args):
        """Called when recording stops."""
        self.logger.info("[DiscordBot] Recording stopped")

    def _recording_error(self, error):
        """Called when recording encounters an error."""
        self.logger.error(f"[DiscordBot] Recording error: {error}")