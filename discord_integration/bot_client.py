# discord_integration/bot_client.py
import discord
import asyncio
import time
import queue # For queue.Empty
import io    # For io.BytesIO
import threading
from typing import Optional, TYPE_CHECKING, List # Added List

if TYPE_CHECKING: # To avoid circular import for type hinting
    from ..DanzarVLM import AppContext # Adjust path if DanzarVLM.py is not in parent
    # from ..services.audio_service import AudioService # Not directly used here

class DiscordBotRunner:
    def __init__(self, app_context: 'AppContext'):
        self.app_context = app_context
        self.logger = app_context.logger
        
        intents = discord.Intents.default()
        intents.voice_states = True 
        intents.message_content = True 
        
        self.bot = discord.Client(intents=intents)
        self.app_context.discord_bot_instance = self.bot

        self.target_guild_id: Optional[int] = None
        self.target_voice_channel_id: Optional[int] = None
        self.auto_leave_timer_task: Optional[asyncio.Task] = None
        self.auto_leave_if_alone_timeout_s: float = 60.0
        self.bot_auto_rejoin_enabled: bool = True
        self.is_starting_recording = False

        self._setup_bot_events()

    def _setup_bot_events(self):
        @self.bot.event
        async def on_ready():
            self.logger.info(f"[DiscordBot] Logged in as {self.bot.user.name} (ID: {self.bot.user.id})")
            self.app_context.discord_bot_async_loop = self.bot.loop

            self.target_guild_id = self.app_context.global_settings.get("DISCORD_GUILD_ID")
            self.target_voice_channel_id = self.app_context.global_settings.get("DISCORD_VOICE_CHANNEL_ID")
            self.auto_leave_if_alone_timeout_s = float(self.app_context.global_settings.get("DISCORD_AUTO_LEAVE_TIMEOUT_S", 60.0))
            self.bot_auto_rejoin_enabled = self.app_context.global_settings.get("DISCORD_BOT_AUTO_REJOIN_ENABLED", True)

            if self.target_guild_id and self.target_voice_channel_id:
                await self._join_target_voice_channel()
            else:
                self.logger.warning("[DiscordBot] Target guild/voice channel ID not set for auto-join.")

        @self.bot.event
        async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
            if not self.bot.user or member.id == self.bot.user.id: return
            
            bot_vc = self.app_context.discord_voice_client
            
            # Scenario 1: Bot is connected
            if bot_vc and bot_vc.is_connected():
                current_bot_channel = bot_vc.channel
                # Check if update is relevant
                if (before.channel == current_bot_channel and after.channel != current_bot_channel) or \
                   (before.channel != current_bot_channel and after.channel == current_bot_channel) or \
                   (before.channel == current_bot_channel and after.channel == current_bot_channel and before.channel is not None):
                    
                    num_human_members = sum(1 for m in current_bot_channel.members if not m.bot)
                    self.logger.debug(f"Voice update in bot's channel '{current_bot_channel.name}'. Humans: {num_human_members}")
                    
                    if num_human_members == 0:
                        if not self.auto_leave_timer_task or self.auto_leave_timer_task.done():
                            self.logger.info(f"Bot alone in '{current_bot_channel.name}'. Starting auto-leave timer ({self.auto_leave_if_alone_timeout_s}s).")
                            self.auto_leave_timer_task = self.bot.loop.create_task(self._auto_leave_if_alone(bot_vc, current_bot_channel))
                    else: # Someone is with the bot
                        if self.auto_leave_timer_task and not self.auto_leave_timer_task.done():
                            self.logger.info(f"User in '{current_bot_channel.name}'. Cancelling auto-leave timer.")
                            self.auto_leave_timer_task.cancel()
                            self.auto_leave_timer_task = None # Clear the task reference
            
            # Scenario 2: Bot NOT connected, user joins TARGET channel
            elif (not bot_vc or not bot_vc.is_connected()) and self.bot_auto_rejoin_enabled: # Added check for bot_vc.is_connected() just in case
                if member.bot: return

                if after.channel and after.channel.id == self.target_voice_channel_id:
                    # Check if anyone (human) is in the target channel now
                    if sum(1 for m in after.channel.members if not m.bot) > 0:
                        self.logger.info(f"User '{member.display_name}' joined target VC '{after.channel.name}'. Bot attempting auto-rejoin.")
                        await self._join_target_voice_channel()

        @self.bot.event
        async def on_message(message: discord.Message):
            if message.author == self.bot.user:
                return

            command_prefix = self.app_context.global_settings.get("DISCORD_COMMAND_PREFIX", "!")
            
            # Check if the message starts with the base command prefix + "danzar"
            base_command_trigger = f"{command_prefix}danzar"
            
            if not message.content.lower().startswith(base_command_trigger.lower()):
                # Not a command for Danzar, or not using the main trigger word
                # If you have other bot commands with different triggers, handle them here or let discord.ext.commands process
                # await self.bot.process_commands(message) # If using commands.Bot
                return

            # Extract arguments/subcommands after "!danzar"
            # Add a space to base_command_trigger for splitting, or handle no-space case
            if len(message.content) > len(base_command_trigger) and message.content[len(base_command_trigger)] == ' ':
                args_part = message.content[len(base_command_trigger) + 1:].strip()
            elif len(message.content) == len(base_command_trigger): # Just "!danzar"
                args_part = ""
            else: # e.g. "!danzarcommentary" - treat as not a valid subcommand start for this logic
                # Or you could decide to parse this differently. For now, assume space separation for subcommands.
                # If this was meant to be a query, it will be caught by the general query logic later IF it has a space.
                # This path currently means only exact "!danzar" or "!danzar <something>" is handled.
                # We'll refine the general query catcher too.
                args_part = message.content[len(base_command_trigger):].strip() # Capture what's after !danzar

            args_part_lower = args_part.lower()

            # --- Specific Internal Subcommands ---
            if args_part_lower == "commentary on":
                self.logger.info(f"[DiscordBot] Internal Cmd: commentary on from {message.author.name}")
                self.app_context.ndi_commentary_enabled.set()
                await message.channel.send("🎙️ NDI Gameplay Commentary: **ON**")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                    async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                    if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                return # Processed

            if args_part_lower == "commentary off":
                self.logger.info(f"[DiscordBot] Internal Cmd: commentary off from {message.author.name}")
                self.app_context.ndi_commentary_enabled.clear()
                await message.channel.send("🎙️ NDI Gameplay Commentary: **OFF**")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                    async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                    if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                return # Processed

            # Add other specific subcommands here, e.g., "profile <name>"
            # if args_part_lower.startswith("profile "):
            #     profile_name = args_part[len("profile "):].strip()
            #     # ... handle profile change ...
            #     return

            # --- If it wasn't an internal subcommand, it might be a query or just "!danzar" ---
            if args_part: # If there was *something* after "!danzar "
                user_query = args_part # Use the case-preserved args_part as the query
                self.logger.info(f"[DiscordBot] Forwarding Danzar query to LLM from {message.author.name}: '{user_query}'")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                
                if self.app_context.llm_service_instance:
                    threading.Thread(
                        target=self.app_context.llm_service_instance.handle_user_text_query,
                        args=(user_query, message.author.name),
                        daemon=True
                    ).start()
                else:
                    await message.channel.send("Sorry, my LLM service is not available right now.")
                    if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                        self.app_context.is_in_conversation.clear()
                return # Query handled

            elif message.content.lower() == base_command_trigger.lower(): # Exactly "!danzar" with no arguments
                await message.channel.send(f"Hi {message.author.mention}! Try `{command_prefix}danzar commentary on/off` or `{command_prefix}danzar <your question>`.")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                     async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                     if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                return
            
            # --- Potentially other specific commands with arguments, e.g., !danzar profile <name> ---
            # cmd_profile_base = f"{command_prefix}danzar profile "
            # if msg_content_lower.startswith(cmd_profile_base):
            #     profile_name = msg_content[len(cmd_profile_base):].strip()
            #     self.logger.info(f"[DiscordBot] Internal Cmd: change profile to '{profile_name}' from {message.author.name}")
            #     # ... (logic to change profile) ...
            #     await message.channel.send(f"Attempting to switch to profile: {profile_name}")
            #     # ... (handle convo flags) ...
            #     return # Processed, stop here


            # --- General !danzar <query> to LLM ---
            # This will only be reached if none of the specific commands above matched.
            # We check if it starts with "!danzar " (note the space) to ensure it's a query.
            general_danzar_query_prefix = f"{command_prefix}danzar " # Must have a space for a query
            
            if msg_content.lower().startswith(general_danzar_query_prefix.lower()):
                user_query = msg_content[len(general_danzar_query_prefix):].strip()

                if not user_query: # User typed "!danzar " but nothing after
                    await message.channel.send(f"Hi {message.author.mention}, you used the Danzar command but didn't ask a question. What can I help you with?")
                    self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                    if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                         async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                         if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                    return

                # It's a valid query for the LLM
                self.logger.info(f"[DiscordBot] Forwarding Danzar query to LLM from {message.author.name}: '{user_query}'")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                
                if self.app_context.llm_service_instance:
                    threading.Thread(
                        target=self.app_context.llm_service_instance.handle_user_text_query,
                        args=(user_query, message.author.name),
                        daemon=True
                    ).start()
                else:
                    await message.channel.send("Sorry, my LLM service is not available right now.")
                    if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                        self.app_context.is_in_conversation.clear()
                return # Query handled (sent to LLM or error message given)
            
            # If the message was just "!danzar" (no space, no subcommand)
            elif msg_content.lower() == f"{command_prefix}danzar":
                await message.channel.send(f"Hi {message.author.mention}! Did you mean to ask something? Try `{command_prefix}danzar <your question>` or `{command_prefix}danzar commentary on/off`.")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                     async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                     if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                return

            # General !danzar <query>
            general_danzar_command_base = f"{command_prefix}danzar " # Note the space
            if message.content.lower().startswith(general_danzar_command_base.lower()):
                user_query = message.content[len(general_danzar_command_base):].strip()

                if not user_query: # Just "!danzar "
                    await message.channel.send(f"Hi {message.author.mention}, what can I help you with? (Try `{command_prefix}danzar <your question>`)")
                    self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                    if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                         async def clear_convo_flag(): await asyncio.sleep(1); self.app_context.is_in_conversation.clear()
                         if self.bot.loop: self.bot.loop.create_task(clear_convo_flag())
                    return

                self.logger.info(f"[DiscordBot] Danzar query from {message.author.name}: '{user_query}'")
                self.app_context.is_in_conversation.set(); self.app_context.last_interaction_time = time.time()
                
                if self.app_context.llm_service_instance:
                    threading.Thread(
                        target=self.app_context.llm_service_instance.handle_user_text_query,
                        args=(user_query, message.author.name),
                        daemon=True
                    ).start()
                else:
                    await message.channel.send("Sorry, my LLM service is unavailable.")
                    # Clear convo flag if LLM is down
                    if self.app_context.global_settings.get("CLEAR_CONVERSATION_FLAG_IMMEDIATELY_FOR_INTERNAL_CMDS", True):
                         self.app_context.is_in_conversation.clear()
                return

    async def _join_target_voice_channel(self):
        if not self.target_guild_id or not self.target_voice_channel_id: return None
        guild = self.bot.get_guild(self.target_guild_id)
        if not guild: self.logger.error(f"Target guild {self.target_guild_id} not found."); return None
        voice_channel = guild.get_channel(self.target_voice_channel_id)
        if not voice_channel or not isinstance(voice_channel, discord.VoiceChannel):
            self.logger.error(f"Target VC {self.target_voice_channel_id} not found/not VC."); return None

        current_vc = self.app_context.discord_voice_client
        if current_vc and current_vc.channel.id == voice_channel.id and current_vc.is_connected():
            self.logger.info(f"Already in target VC: '{voice_channel.name}'. Ensuring recording.")
            self._ensure_recording_is_active(current_vc)
            return current_vc

        try:
            if current_vc and current_vc.is_connected(): # If connected to a different channel
                self.logger.info(f"Disconnecting from current VC '{current_vc.channel.name}' to join target.")
                await current_vc.disconnect(force=True) # Use force=True for quicker disconnect if needed
                self.app_context.discord_voice_client = None # Clear it after successful disconnect
            
            self.logger.info(f"Attempting to connect to VC: '{voice_channel.name}'")
            new_vc = await voice_channel.connect(timeout=10.0, reconnect=True) # Added timeout and reconnect
            self.app_context.discord_voice_client = new_vc
            self.logger.info(f"Successfully connected to '{voice_channel.name}'.")
            self._start_recording(new_vc)

            if self.auto_leave_timer_task and not self.auto_leave_timer_task.done():
                self.auto_leave_timer_task.cancel() # Cancel any pending leave timer
                self.auto_leave_timer_task = None
            return new_vc
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout trying to connect to VC '{voice_channel.name}'.")
        except discord.errors.ClientException as e:
             self.logger.error(f"Discord ClientException while connecting to '{voice_channel.name}': {e}")
        except Exception as e:
            self.logger.error(f"Generic error connecting to VC '{voice_channel.name}': {e}", exc_info=True)
        
        # Cleanup if connection failed partially
        if self.app_context.discord_voice_client and not self.app_context.discord_voice_client.is_connected():
            self.app_context.discord_voice_client = None
        elif self.app_context.discord_voice_client and self.app_context.discord_voice_client.channel.id != voice_channel.id:
            # This case should ideally be handled by disconnecting above, but as a safeguard:
            try: await self.app_context.discord_voice_client.disconnect(force=True)
            except: pass
            self.app_context.discord_voice_client = None
        return None


    def _start_recording(self, voice_client: discord.VoiceClient):
        if not self.app_context.audio_service_instance: 
            self.logger.error("[DiscordBot] AudioService not available. Cannot start recording.")
            return
        if self.is_starting_recording:
            self.logger.debug("[DiscordBot] Recording start sequence already in progress.")
            return
        # Check if sink is already there and listening
        if hasattr(voice_client, 'main_sink') and voice_client.main_sink and voice_client.main_sink.is_listening():
             self.logger.info(f"[DiscordBot] Recording already active in '{voice_client.channel.name}'.")
             return

        self.is_starting_recording = True
        self.logger.info(f"[DiscordBot] Attempting to start recording in '{voice_client.channel.name}'.")
        try:
            from .audio_sink import DanzarAudioSink # Ensure this path is correct
            sink = DanzarAudioSink(self.app_context, self.logger) # <--- PASS self.app_context
            
            voice_client.start_recording(sink, None)
            voice_client.main_sink = sink 
            self.logger.info(f"[DiscordBot] Recording started in '{voice_client.channel.name}'.")
        except Exception as e: 
            self.logger.error(f"[DiscordBot] Error starting recording: {e}", exc_info=True)
        finally: 
            self.is_starting_recording = False
            
    def _ensure_recording_is_active(self, voice_client: discord.VoiceClient):
        if voice_client and voice_client.is_connected():
            # Check if sink attribute exists and if it's listening
            if not hasattr(voice_client, 'main_sink') or \
               not voice_client.main_sink or \
               not getattr(voice_client.main_sink, 'is_listening', lambda: False)(): # Safely check is_listening
                self.logger.info(f"[DiscordBot] Recording in '{voice_client.channel.name}' appears inactive. Attempting to (re)start.")
                self._start_recording(voice_client)
            else: 
                self.logger.debug(f"[DiscordBot] Recording in '{voice_client.channel.name}' confirmed active.")


    async def _auto_leave_if_alone(self, voice_client: discord.VoiceClient, channel: discord.VoiceChannel):
        try:
            await asyncio.sleep(self.auto_leave_if_alone_timeout_s)
            # Re-check conditions before leaving
            if voice_client.is_connected() and voice_client.channel and voice_client.channel.id == channel.id:
                if sum(1 for m in channel.members if not m.bot) == 0:
                    self.logger.info(f"[DiscordBot] Auto-leave timer expired. Still alone in '{channel.name}'. Disconnecting.")
                    await voice_client.disconnect(force=False) 
                    self.app_context.discord_voice_client = None
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
        self.app_context.tts_is_playing.clear()


    def run_playback_loop(self):
        self.logger.info("[DiscordBot] Starting TTS/Text Playback Loop.")
        while not self.app_context.shutdown_event.is_set():
            try:
                # Process Text Messages
                if not self.app_context.text_message_queue.empty():
                    text_msg = self.app_context.text_message_queue.get_nowait()
                    if self.bot.is_ready() and self.app_context.discord_bot_async_loop: # Check bot.loop too
                        text_channel_id = self.app_context.global_settings.get("DISCORD_TEXT_CHANNEL_ID")
                        if text_channel_id:
                            channel = self.bot.get_channel(text_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                asyncio.run_coroutine_threadsafe(channel.send(text_msg), self.bot.loop)
                            else: self.logger.warning(f"[DiscordBotPlayback] Text channel {text_channel_id} not found.")
                        else: self.logger.warning("[DiscordBotPlayback] DISCORD_TEXT_CHANNEL_ID not set for text output.")
                    self.app_context.text_message_queue.task_done()
                
                # Process TTS Audio
                if not self.app_context.tts_is_playing.is_set() and \
                   not self.app_context.tts_queue.empty():
                    
                    audio_data_bytes = self.app_context.tts_queue.get_nowait()
                    
                    vc = self.app_context.discord_voice_client
                    if vc and vc.is_connected() and self.bot.is_ready() and self.app_context.discord_bot_async_loop:
                        
                        self.logger.debug("[DiscordBotPlayback] Attempting to play TTS audio.")
                        self.app_context.tts_is_playing.set()
                        
                        audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_data_bytes), pipe=True)
                        
                        vc.play(audio_source, after=self._after_tts_playback)
                        # self.logger.debug("[DiscordBotPlayback] TTS audio passed to voice_client.play()") # Already logged by play()
                    else:
                        if not (vc and vc.is_connected()):
                             self.logger.warning("[DiscordBotPlayback] Voice client not connected. TTS audio dropped.")
                        else: # Bot not ready or loop not available
                             self.logger.warning("[DiscordBotPlayback] Bot not ready/loop NA for TTS. Audio dropped.")
                        self.app_context.tts_is_playing.clear() # Ensure flag is cleared if play didn't happen
                    
                    self.app_context.tts_queue.task_done()
                
                time.sleep(0.05) 

            except queue.Empty:
                time.sleep(0.1) 
            except Exception as e:
                self.logger.error(f"[DiscordBotPlayback] Error in playback loop: {e}", exc_info=True)
                if self.app_context.tts_is_playing.is_set(): # Critical to clear flag on error
                    self.app_context.tts_is_playing.clear()
                time.sleep(1)
        self.logger.info("[DiscordBot] TTS/Text Playback Loop stopped.")

    def run(self):
        token = self.app_context.global_settings.get("DISCORD_BOT_TOKEN")
        if not token: 
            self.logger.critical("[DiscordBot] DISCORD_BOT_TOKEN missing. Bot cannot run.")
            return
        try:
            self.logger.info("[DiscordBot] Starting Discord bot client...")
            self.bot.run(token)
        except discord.LoginFailure:
            self.logger.critical("[DiscordBot] Login failed: Invalid Discord token provided.")
        except Exception as e:
            self.logger.critical(f"[DiscordBot] Critical error running bot: {e}", exc_info=True)