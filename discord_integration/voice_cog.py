#!/usr/bin/env python3
"""
Voice Commands Cog
=================

Cog containing voice-related Discord commands for DanzarAI.
Note: !join and !leave commands are handled by the main bot to avoid conflicts.
"""

import discord
from discord.ext import commands
import logging
from typing import Optional

class VoiceCog(commands.Cog):
    """Cog for voice-related commands."""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = getattr(bot, 'logger', logging.getLogger(__name__))
        
        # Get voice-related attributes from bot
        self.connections = getattr(bot, 'connections', {})
        self.current_text_channel = getattr(bot, 'current_text_channel', None)
        self.recording_finished = getattr(bot, 'recording_finished', None)
        
        self.logger.info(f"[VoiceCog] Initialized")
        self.logger.info(f"[VoiceCog] Note: !join and !leave commands are handled by main bot")

    # Note: !join and !leave commands are handled by the main bot to avoid conflicts
    # The main bot already has these commands defined in its add_commands() method

async def setup(bot):
    """Setup function for the cog."""
    await bot.add_cog(VoiceCog(bot)) 