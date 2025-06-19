"""
Enhanced Discord Voice Bot for DanzarAI
Integrates voice chat functionality with existing DanzarAI services
"""

import discord
from discord.ext import commands, voice_recv
import asyncio
import logging
from typing import Optional, Dict, Any
import os

from services.voice_chat_service import VoiceChatService
from services.tts_service import TTSService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from core.config_loader import load_global_settings
from utils.general_utils import setup_logger


class EnhancedVoiceBot(commands.Bot):
    """
    Enhanced Discord Bot with Voice Chat Capabilities
    
    Features:
    - Voice recognition and STT processing
    - LLM conversation with context memory
    - TTS response playback
    - Turn-taking coordination
    - Integration with existing DanzarAI services
    """
    
    def __init__(self):
        """Initialize the enhanced voice bot."""
        # Load configuration
        self.config = load_global_settings() or {}
        
        # Setup Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        # Initialize bot
        command_prefix = self.config.get('DISCORD_COMMAND_PREFIX', '!')
        super().__init__(command_prefix=command_prefix, intents=intents)
        
        # Setup logging
        self.logger = setup_logger(__name__)
        
        # Service instances
        self.tts_service: Optional[TTSService] = None
        self.llm_service: Optional[LLMService] = None
        self.memory_service: Optional[MemoryService] = None
        self.voice_chat_service: Optional[VoiceChatService] = None
        
        # Discord state
        self.target_guild_id = self.config.get('DISCORD_GUILD_ID')
        self.target_text_channel_id = self.config.get('DISCORD_TEXT_CHANNEL_ID')
        self.target_voice_channel_id = self.config.get('DISCORD_VOICE_CHANNEL_ID')
        
        self.logger.info("[EnhancedVoiceBot] Bot initialized")
    
    async def setup_services(self) -> bool:
        """Initialize DanzarAI services."""
        try:
            # Initialize services in dependency order
            self.logger.info("[EnhancedVoiceBot] Initializing DanzarAI services...")
            
            # Initialize TTS service
            self.tts_service = TTSService(self.config)
            if not await self.tts_service.initialize():
                self.logger.error("[EnhancedVoiceBot] Failed to initialize TTS service")
                return False
            
            # Initialize Memory service
            self.memory_service = MemoryService(self.config)
            if not await self.memory_service.initialize():
                self.logger.error("[EnhancedVoiceBot] Failed to initialize Memory service")
                return False
            
            # Initialize LLM service
            self.llm_service = LLMService(self.config, self.memory_service)
            if not await self.llm_service.initialize():
                self.logger.error("[EnhancedVoiceBot] Failed to initialize LLM service")
                return False
            
            # Initialize Voice Chat service
            self.voice_chat_service = VoiceChatService(
                tts_service=self.tts_service,
                llm_service=self.llm_service,
                memory_service=self.memory_service
            )
            if not await self.voice_chat_service.initialize():
                self.logger.error("[EnhancedVoiceBot] Failed to initialize Voice Chat service")
                return False
            
            self.logger.info("[EnhancedVoiceBot] All services initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Service initialization error: {e}", exc_info=True)
            return False
    
    async def on_ready(self):
        """Called when bot is ready and connected."""
        self.logger.info(f'[EnhancedVoiceBot] Bot is ready! Logged in as {self.user}')
        self.logger.info(f'[EnhancedVoiceBot] Commands loaded: {[cmd.name for cmd in self.commands]}')
        
        # Initialize services
        if not await self.setup_services():
            self.logger.error("[EnhancedVoiceBot] Failed to setup services - bot may not function properly")
        
        # Auto-join voice channel if configured
        if self.config.get('DISCORD_BOT_AUTO_REJOIN_ENABLED', False):
            await self._auto_join_voice_channel()
    
    async def _auto_join_voice_channel(self):
        """Automatically join the configured voice channel."""
        try:
            if not self.target_voice_channel_id:
                self.logger.warning("[EnhancedVoiceBot] No target voice channel configured for auto-join")
                return
            
            voice_channel = self.get_channel(self.target_voice_channel_id)
            if not voice_channel:
                self.logger.error(f"[EnhancedVoiceBot] Voice channel {self.target_voice_channel_id} not found")
                return
            
            # Join voice channel with voice receive capability
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
            
            # Start voice chat service
            if self.voice_chat_service:
                await self.voice_chat_service.start_listening(voice_client)
                self.logger.info(f"[EnhancedVoiceBot] Auto-joined voice channel: {voice_channel.name}")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Auto-join failed: {e}", exc_info=True)
    
    @commands.command(name='join')
    async def join_command(self, ctx):
        """Join the voice channel that the user is currently in."""
        try:
            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.send("❌ You need to be in a voice channel first!")
                return
            
            channel = ctx.author.voice.channel
            
            # Check if already connected
            if ctx.voice_client:
                if ctx.voice_client.channel == channel:
                    await ctx.send(f"ℹ️ Already in voice channel: **{channel.name}**")
                    return
                else:
                    # Move to new channel
                    await ctx.voice_client.move_to(channel)
                    await ctx.send(f"✅ Moved to voice channel: **{channel.name}**")
            else:
                # Connect with voice receive capability
                voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
                await ctx.send(f"✅ Joined voice channel: **{channel.name}**")
            
            # Start voice chat service
            if self.voice_chat_service and ctx.voice_client:
                await self.voice_chat_service.start_listening(ctx.voice_client)
                await ctx.send("🎤 **DanzarAI Voice Chat is now active!** Speak to me and I'll respond.")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Join command error: {e}", exc_info=True)
            await ctx.send("❌ Failed to join voice channel. Check logs for details.")
    
    @commands.command(name='leave')
    async def leave_command(self, ctx):
        """Leave the current voice channel."""
        try:
            if not ctx.voice_client:
                await ctx.send("ℹ️ I'm not in any voice channel!")
                return
            
            channel_name = ctx.voice_client.channel.name
            
            # Stop voice chat service
            if self.voice_chat_service:
                self.voice_chat_service.stop_listening()
            
            # Disconnect from voice
            await ctx.voice_client.disconnect()
            await ctx.send(f"✅ Left voice channel: **{channel_name}**")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Leave command error: {e}", exc_info=True)
            await ctx.send("❌ Error leaving voice channel.")
    
    @commands.command(name='voice_status')
    async def voice_status_command(self, ctx):
        """Show voice chat service status."""
        try:
            if not self.voice_chat_service:
                await ctx.send("❌ Voice chat service not initialized.")
                return
            
            status = self.voice_chat_service.get_context_summary()
            
            embed = discord.Embed(title="🎤 Voice Chat Status", color=0x00ff00)
            embed.add_field(name="Listening", value="✅" if status['is_listening'] else "❌", inline=True)
            embed.add_field(name="Processing", value="🔄" if status['is_processing'] else "⏸️", inline=True)
            embed.add_field(name="Speaking", value="🗣️" if status['is_speaking'] else "🤐", inline=True)
            embed.add_field(name="Context Length", value=f"{status['context_length']} messages", inline=False)
            
            if status['recent_messages']:
                recent_text = "\n".join([
                    f"**{msg['role']}**: {msg['content'][:50]}..." 
                    for msg in status['recent_messages']
                ])
                embed.add_field(name="Recent Messages", value=recent_text, inline=False)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Voice status error: {e}", exc_info=True)
            await ctx.send("❌ Error getting voice status.")
    
    @commands.command(name='clear_context')
    async def clear_context_command(self, ctx):
        """Clear the voice conversation context."""
        try:
            if not self.voice_chat_service:
                await ctx.send("❌ Voice chat service not available.")
                return
            
            self.voice_chat_service.clear_context()
            await ctx.send("✅ Voice conversation context cleared.")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Clear context error: {e}", exc_info=True)
            await ctx.send("❌ Error clearing context.")
    
    @commands.command(name='test_voice')
    async def test_voice_command(self, ctx):
        """Test voice chat functionality."""
        try:
            if not ctx.voice_client:
                await ctx.send("❌ I need to be in a voice channel first! Use `!join`")
                return
            
            if not self.voice_chat_service:
                await ctx.send("❌ Voice chat service not available.")
                return
            
            # Test TTS
            test_message = "Hello! This is a test of the DanzarAI voice chat system. I can hear you and respond with voice!"
            
            # Generate and play TTS
            await self.voice_chat_service._play_tts_response(test_message)
            await ctx.send("🔊 Voice test completed! Did you hear me?")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Voice test error: {e}", exc_info=True)
            await ctx.send("❌ Voice test failed. Check logs for details.")
    
    @commands.command(name='chat')
    async def chat_command(self, ctx, *, message: str):
        """Text-based chat with DanzarAI (also works in voice channels)."""
        try:
            if not self.llm_service:
                await ctx.send("❌ LLM service not available.")
                return
            
            # Show typing indicator
            async with ctx.typing():
                # Get LLM response
                response = await self.llm_service.handle_user_text_query(message, ctx.author.display_name)
                
                if not response:
                    await ctx.send("🤔 I couldn't generate a response. Please try again.")
                    return
                
                # Clean response
                clean_response = self._strip_think_tags(response)
                
                # Send text response
                if len(clean_response) > 1900:
                    # Split long responses
                    chunks = [clean_response[i:i+1900] for i in range(0, len(clean_response), 1900)]
                    for i, chunk in enumerate(chunks):
                        await ctx.send(f"**[Part {i+1}/{len(chunks)}]**\n{chunk}")
                else:
                    await ctx.send(clean_response)
                
                # If in voice channel, also play TTS
                if ctx.voice_client and self.voice_chat_service:
                    try:
                        await self.voice_chat_service._play_tts_response(clean_response)
                    except Exception as tts_error:
                        self.logger.warning(f"[EnhancedVoiceBot] TTS playback failed: {tts_error}")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Chat command error: {e}", exc_info=True)
            await ctx.send("❌ Error processing your message.")
    
    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM responses."""
        if not text:
            return text
        
        import re
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        if not clean_text and text.strip():
            clean_text = "I'm thinking about that... let me get back to you."
        
        return clean_text
    
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state changes for auto-management."""
        try:
            # Auto-leave if alone in voice channel
            if (self.voice_client and 
                len(self.voice_client.channel.members) == 1 and  # Only bot left
                self.config.get('DISCORD_AUTO_LEAVE_TIMEOUT_S', 0) > 0):
                
                timeout = self.config.get('DISCORD_AUTO_LEAVE_TIMEOUT_S', 600)
                await asyncio.sleep(timeout)
                
                # Check again after timeout
                if (self.voice_client and 
                    len(self.voice_client.channel.members) == 1):
                    
                    channel_name = self.voice_client.channel.name
                    if self.voice_chat_service:
                        self.voice_chat_service.stop_listening()
                    
                    await self.voice_client.disconnect()
                    self.logger.info(f"[EnhancedVoiceBot] Auto-left voice channel: {channel_name} (alone for {timeout}s)")
                    
                    # Send notification to text channel
                    if self.target_text_channel_id:
                        text_channel = self.get_channel(self.target_text_channel_id)
                        if text_channel:
                            await text_channel.send(f"🚪 Left **{channel_name}** (alone for {timeout} seconds)")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Voice state update error: {e}", exc_info=True)
    
    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            self.logger.info("[EnhancedVoiceBot] Starting cleanup...")
            
            # Cleanup voice chat service
            if self.voice_chat_service:
                await self.voice_chat_service.cleanup()
            
            # Disconnect from voice if connected
            if self.voice_client:
                await self.voice_client.disconnect()
            
            # Cleanup other services
            if self.llm_service and hasattr(self.llm_service, 'cleanup'):
                await self.llm_service.cleanup()
            
            if self.memory_service and hasattr(self.memory_service, 'cleanup'):
                await self.memory_service.cleanup()
            
            if self.tts_service and hasattr(self.tts_service, 'cleanup'):
                await self.tts_service.cleanup()
            
            self.logger.info("[EnhancedVoiceBot] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"[EnhancedVoiceBot] Cleanup error: {e}", exc_info=True)


async def main():
    """Main entry point for the enhanced voice bot."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_global_settings()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Get Discord token
        token = config.get('DISCORD_BOT_TOKEN')
        if not token:
            logger.error("DISCORD_BOT_TOKEN not found in configuration")
            return
        
        # Create and run bot
        bot = EnhancedVoiceBot()
        
        try:
            logger.info("Starting DanzarAI Enhanced Voice Bot...")
            await bot.start(token)
        except KeyboardInterrupt:
            logger.info("Bot shutdown requested")
        except Exception as e:
            logger.error(f"Bot runtime error: {e}", exc_info=True)
        finally:
            await bot.cleanup()
            await bot.close()
            
    except Exception as e:
        logger.error(f"Main function error: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}") 