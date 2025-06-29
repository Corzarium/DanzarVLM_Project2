#!/usr/bin/env python3
"""
Vision Commands Cog
==================

Cog containing vision-related Discord commands for DanzarAI.
"""

import discord
from discord.ext import commands
import logging
from typing import Optional

class VisionCog(commands.Cog):
    """Cog for vision-related commands."""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = getattr(bot, 'logger', logging.getLogger(__name__))
        
        # Get vision integration service from bot
        self.vision_integration_service = getattr(bot, 'vision_integration_service', None)
        
        if self.vision_integration_service:
            self.logger.info("[VisionCog] Initialized with vision integration service")
        else:
            self.logger.warning("[VisionCog] No vision integration service available")
    
    @commands.command(name='watch')
    async def watch_command(self, ctx):
        """Start DanzarAI vision commentary (OBS feed)."""
        self.logger.info(f"👁️ !watch used by {ctx.author.name}")
        
        if not self.vision_integration_service:
            await ctx.send("❌ Vision integration service not available")
            return
        
        # Only allow one watcher at a time
        if getattr(self.vision_integration_service, 'is_watching', False):
            await ctx.send("👁️ I'm already watching and commenting on the OBS feed!")
            return
        
        # Start vision integration
        async def discord_text_callback(msg: str):
            if msg and msg.strip():
                await ctx.send(f"🦾 {msg}")
        
        await ctx.send("👁️ Starting vision commentary! DanzarAI will now watch the OBS feed and comment as things happen.")
        try:
            started = await self.vision_integration_service.start_watching(
                text_callback=discord_text_callback,
                tts_callback=None  # Add TTS to voice if desired
            )
            if started:
                await ctx.send("✅ Vision commentary started! Use !stopwatch to stop.")
            else:
                await ctx.send("❌ Failed to start vision commentary.")
        except Exception as e:
            self.logger.error(f"[VisionCog] Failed to start vision commentary: {e}")
            await ctx.send(f"❌ Error: {e}")

    @commands.command(name='stopwatch')
    async def stopwatch_command(self, ctx):
        """Stop DanzarAI vision commentary."""
        self.logger.info(f"🛑 !stopwatch used by {ctx.author.name}")
        
        if not self.vision_integration_service:
            await ctx.send("❌ Vision integration service not available")
            return
        
        if not getattr(self.vision_integration_service, 'is_watching', False):
            await ctx.send("👁️ I'm not currently watching the OBS feed.")
            return
        
        try:
            stopped = await self.vision_integration_service.stop_watching()
            if stopped:
                await ctx.send("🛑 Vision commentary stopped.")
            else:
                await ctx.send("❌ Failed to stop vision commentary.")
        except Exception as e:
            self.logger.error(f"[VisionCog] Failed to stop vision commentary: {e}")
            await ctx.send(f"❌ Error: {e}")

async def setup(bot):
    """Setup function for the cog."""
    await bot.add_cog(VisionCog(bot)) 