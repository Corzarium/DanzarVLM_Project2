#!/usr/bin/env python3
"""
Azure TTS Environment Setup Script
Creates a .env file with Azure TTS configuration.
"""

import os
import sys

def create_env_file():
    """Create .env file with Azure TTS configuration."""
    
    env_content = """# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id_here
DISCORD_TEXT_CHANNEL_ID=your_text_channel_id_here

# Azure TTS Configuration
AZURE_TTS_SUBSCRIPTION_KEY=your_azure_subscription_key_here
AZURE_TTS_REGION=eastus
AZURE_TTS_VOICE=en-US-AdamMultilingualNeural
AZURE_TTS_SPEECH_RATE=+0%
AZURE_TTS_PITCH=+0%
AZURE_TTS_VOLUME=+0%

# Other Configuration
LOG_LEVEL=INFO
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with Azure TTS configuration")
        print("\n📝 Please edit the .env file and add your actual values:")
        print("   - DISCORD_BOT_TOKEN: Your Discord bot token")
        print("   - DISCORD_VOICE_CHANNEL_ID: Your voice channel ID")
        print("   - DISCORD_TEXT_CHANNEL_ID: Your text channel ID")
        print("   - AZURE_TTS_SUBSCRIPTION_KEY: Your Azure Speech Service subscription key")
        print("\n🔧 After editing .env, restart your bot for the changes to take effect.")
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required values."""
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    try:
        with open('.env', 'r') as f:
            content = f.read()
        
        required_vars = [
            'DISCORD_BOT_TOKEN',
            'AZURE_TTS_SUBSCRIPTION_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if f'{var}=' not in content or f'{var}=your_' in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️ Missing or placeholder values in .env: {', '.join(missing_vars)}")
            return False
        
        print("✅ .env file exists and has required values")
        return True
        
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def main():
    print("🔧 Azure TTS Environment Setup")
    print("=" * 40)
    
    if check_env_file():
        print("\n✅ Environment is properly configured!")
        return
    
    print("\n📝 Creating .env file...")
    create_env_file()

if __name__ == "__main__":
    main() 