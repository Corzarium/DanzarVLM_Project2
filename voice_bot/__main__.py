"""
Main entry point for the voice-enabled Discord bot.
"""
import logging
import os
from pathlib import Path
from .config.settings import get_settings
from .core.discord_bot import VoiceBot

def setup_logging(settings: dict):
    """Setup logging configuration."""
    logging.basicConfig(
        level=settings["logging"]["LEVEL"],
        format=settings["logging"]["FORMAT"],
        handlers=[
            logging.FileHandler(settings["logging"]["FILE"]),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point."""
    try:
        # Load settings
        settings = get_settings()
        
        # Setup logging
        setup_logging(settings)
        
        # Create and run bot
        bot = VoiceBot(settings)
        bot.run()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 