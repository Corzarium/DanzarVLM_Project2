#!/usr/bin/env python3
"""
Example usage of Danzar Voice Pipeline

This script demonstrates how to set up and run the voice pipeline
with basic configuration and error handling.
"""

import asyncio
import logging
from danzar_voice_pipeline import DanzarVoicePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_pipeline_example():
    """Example of running the voice pipeline"""
    
    # Configuration
    config = {
        'command_prefix': '!',
        'voice_channel_id': None,  # Will be set from environment
        'text_channel_id': None,   # Will be set from environment
        'whisper_model': 'base',   # Good balance of speed/accuracy
        'vlm_endpoint': 'http://localhost:8083/chat/completions',
        'tts_endpoint': 'http://localhost:8055/tts'
    }
    
    # Load environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get bot token
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        logger.error("❌ DISCORD_BOT_TOKEN not found in environment!")
        return
    
    # Override config with environment variables
    if os.getenv('DISCORD_VOICE_CHANNEL_ID'):
        config['voice_channel_id'] = int(os.getenv('DISCORD_VOICE_CHANNEL_ID'))
    if os.getenv('DISCORD_TEXT_CHANNEL_ID'):
        config['text_channel_id'] = int(os.getenv('DISCORD_TEXT_CHANNEL_ID'))
    
    # Create bot instance
    bot = DanzarVoicePipeline(config)
    
    # Add custom commands
    @bot.command(name='analyze')
    async def analyze_command(ctx):
        """Trigger manual analysis of current audio"""
        if bot.voice_client and bot.voice_client.is_connected():
            await ctx.send("🎤 Manual analysis triggered! Check the logs for results.")
        else:
            await ctx.send("❌ Not connected to a voice channel!")
    
    @bot.command(name='models')
    async def models_command(ctx):
        """Show loaded model status"""
        embed = discord.Embed(title="🤖 Model Status", color=0x00ff00)
        
        # Check Whisper
        whisper_status = "✅ Loaded" if bot.whisper.model else "❌ Not loaded"
        embed.add_field(name="Whisper STT", value=whisper_status, inline=True)
        
        # Check Emotion Recognition
        emotion_status = "✅ Loaded" if bot.emotion_recognizer.classifier else "❌ Not loaded"
        embed.add_field(name="Emotion Recognition", value=emotion_status, inline=True)
        
        # Check Laughter Detection
        laughter_status = "✅ Loaded" if bot.laughter_detector.model else "❌ Not loaded"
        embed.add_field(name="Laughter Detection", value=laughter_status, inline=True)
        
        await ctx.send(embed=embed)
    
    try:
        logger.info("🚀 Starting Danzar Voice Pipeline...")
        await bot.start(bot_token)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
    finally:
        await bot.close()


def main():
    """Main entry point"""
    print("🎤 Danzar Voice Pipeline Example")
    print("=" * 40)
    print("This example demonstrates:")
    print("• Real-time voice analysis")
    print("• Emotion and laughter detection")
    print("• Multimodal VLM integration")
    print("• TTS response generation")
    print("=" * 40)
    
    # Run the pipeline
    asyncio.run(run_pipeline_example())


if __name__ == "__main__":
    main() 