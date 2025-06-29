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

async def setup(bot):
    """Setup function for the cog."""
    await bot.add_cog(VisionCog(bot)) 