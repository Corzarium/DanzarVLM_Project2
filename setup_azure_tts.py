#!/usr/bin/env python3
"""
Simple Azure TTS Setup Script
"""

import os
import sys

def main():
    print("🎤 Azure TTS Setup Helper")
    print("=" * 40)
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print(f"❌ {env_file} file not found")
        print("Creating .env file...")
        
        # Create .env file with Azure TTS configuration
        env_content = """# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_GUILD_ID=your_guild_id_here
DISCORD_TEXT_CHANNEL_ID=your_text_channel_id_here
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id_here

# Azure TTS Configuration (replaces Chatterbox TTS)
AZURE_TTS_SUBSCRIPTION_KEY=your_azure_speech_service_subscription_key_here
AZURE_TTS_REGION=eastus
AZURE_TTS_VOICE=en-US-AdamMultilingualNeural
AZURE_TTS_SPEECH_RATE=+0%
AZURE_TTS_PITCH=+0%
AZURE_TTS_VOLUME=+0%
"""
        
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"✅ Created {env_file} file")
        except Exception as e:
            print(f"❌ Failed to create {env_file}: {e}")
            return
    else:
        print(f"✅ {env_file} file found")
    
    # Check current configuration
    print("\n📋 Current Configuration:")
    
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
        return
    
    subscription_key = os.getenv('AZURE_TTS_SUBSCRIPTION_KEY')
    region = os.getenv('AZURE_TTS_REGION', 'eastus')
    voice = os.getenv('AZURE_TTS_VOICE', 'en-US-AdamMultilingualNeural')
    
    if subscription_key and subscription_key != "your_azure_speech_service_subscription_key_here":
        print(f"✅ Azure TTS subscription key: {subscription_key[:10]}...")
    else:
        print("❌ Azure TTS subscription key not configured")
    
    print(f"🌍 Region: {region}")
    print(f"🎤 Voice: {voice}")
    
    # Provide setup instructions
    print("\n🚀 Setup Instructions:")
    print("1. Go to https://portal.azure.com")
    print("2. Create a 'Speech service' resource")
    print("3. Copy the subscription key (Key 1)")
    print("4. Edit the .env file and replace 'your_azure_speech_service_subscription_key_here'")
    print("5. Run: python test_azure_tts.py")
    print("6. Start DanzarVLM: python DanzarVLM.py")
    
    # Test current setup
    print("\n🧪 Testing Current Setup:")
    try:
        from test_azure_tts import test_azure_tts_setup
        if test_azure_tts_setup():
            print("✅ Basic setup looks good!")
        else:
            print("❌ Setup needs configuration")
    except ImportError:
        print("⚠️ test_azure_tts.py not found")
    
    print("\n💡 For more details, see: AZURE_TTS_SETUP.md")

if __name__ == "__main__":
    main() 