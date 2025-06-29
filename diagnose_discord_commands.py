#!/usr/bin/env python3
"""
Discord Commands Diagnostic
==========================

Diagnose why !watch command is not working.
"""

import sys
import os

def check_imports():
    """Check if all required imports work."""
    print("🔍 Checking imports...")
    
    try:
        import discord
        print("✅ discord imported")
    except Exception as e:
        print(f"❌ discord import failed: {e}")
        return False
    
    try:
        from discord.ext import commands
        print("✅ discord.ext.commands imported")
    except Exception as e:
        print(f"❌ discord.ext.commands import failed: {e}")
        return False
    
    try:
        from services.vision_integration_service import VisionIntegrationService
        print("✅ VisionIntegrationService imported")
    except Exception as e:
        print(f"❌ VisionIntegrationService import failed: {e}")
        return False
    
    try:
        from vision_pipeline import VisionPipeline
        print("✅ VisionPipeline imported")
    except Exception as e:
        print(f"❌ VisionPipeline import failed: {e}")
        return False
    
    return True

def check_bot_client():
    """Check if bot client can be imported."""
    print("\n🔍 Checking bot client...")
    
    try:
        from discord_integration.bot_client import DiscordBot
        print("✅ DiscordBot imported")
        
        # Check if commands are defined
        bot = DiscordBot(
            command_prefix='!',
            intents=discord.Intents.default(),
            stt_service=None,
            tts_service=None,
            llm_service=None,
            memory_service=None,
            app_context=None
        )
        print("✅ DiscordBot instantiated")
        
        # Check for watch command
        watch_cmd = bot.get_command('watch')
        stopwatch_cmd = bot.get_command('stopwatch')
        
        if watch_cmd:
            print("✅ !watch command found")
        else:
            print("❌ !watch command NOT found")
            
        if stopwatch_cmd:
            print("✅ !stopwatch command found")
        else:
            print("❌ !stopwatch command NOT found")
        
        # List all commands
        print(f"\n📋 All registered commands ({len(bot.commands)} total):")
        for cmd in bot.commands:
            print(f"   !{cmd.name} - {cmd.help or 'No description'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot client check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_config():
    """Check if configuration files exist."""
    print("\n🔍 Checking configuration...")
    
    config_files = [
        "config/vision_config.yaml",
        "config/profiles/everquest.yaml",
        "config/profiles/rimworld.yaml",
        "config/profiles/generic_game.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING")

def main():
    """Run diagnostics."""
    print("🔧 Discord Commands Diagnostic")
    print("=" * 40)
    
    # Check imports
    if not check_imports():
        print("\n❌ Import check failed - fix imports first")
        return
    
    # Check bot client
    if not check_bot_client():
        print("\n❌ Bot client check failed")
        return
    
    # Check config
    check_config()
    
    print("\n🎯 Troubleshooting Steps:")
    print("1. Make sure your Discord bot is restarted after the changes")
    print("2. Check that the bot has the correct permissions in Discord")
    print("3. Verify the bot token is correct")
    print("4. Check the bot logs for any error messages")
    print("5. Try the commands in a channel where the bot has access")

if __name__ == "__main__":
    main() 