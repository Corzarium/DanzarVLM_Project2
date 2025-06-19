# DanzarAI Voice Chat Setup Guide

This guide walks you through setting up voice chat functionality for your DanzarAI Discord bot, integrating Speech-to-Text (STT), Large Language Model (LLM) conversation, and Text-to-Speech (TTS) capabilities.

## 🔧 Prerequisites

### 1. Install Required Dependencies

Install the enhanced dependencies:

```bash
# Install the updated requirements
pip install -r requirements.txt

# Or install individually:
pip install discord.py[voice] discord-ext-voice-recv
pip install webrtcvad-wheels
pip install openai-whisper
pip install ollama
```

### 2. System Requirements

- **FFmpeg**: Required for Discord audio processing and Whisper
  - Windows: Already included in your project (`ffmpeg.exe`)
  - Linux/macOS: `sudo apt install ffmpeg` or `brew install ffmpeg`

- **Opus Library**: For Discord voice (handled automatically by discord.py[voice])

### 3. Services Setup

Ensure your DanzarAI services are running:

```bash
# Start Ollama (for LLM)
ollama serve

# Start Chatterbox TTS service (if using containerized setup)
docker-compose up chatterbox

# Start Qdrant (for RAG memory)
docker-compose up qdrant
```

## 🚀 Quick Start

### 1. Update Configuration

Ensure your `config/global_settings.yaml` includes voice chat settings:

```yaml
# Discord Configuration
DISCORD_BOT_TOKEN: "your_bot_token_here"
DISCORD_GUILD_ID: your_guild_id
DISCORD_TEXT_CHANNEL_ID: your_text_channel_id
DISCORD_VOICE_CHANNEL_ID: your_voice_channel_id
DISCORD_BOT_AUTO_REJOIN_ENABLED: true
DISCORD_AUTO_LEAVE_TIMEOUT_S: 600

# Voice Chat Settings
STT_ENABLED: true
WHISPER_MODEL_SIZE: "base.en"  # Options: tiny, base, small, medium, large
LANGUAGE: "en"

# Voice Activity Detection
OWW_VAD_THRESHOLD: 1  # 0-3, higher = more aggressive filtering

# Context Memory
SHORT_TERM_SIZE: 6  # Number of messages to keep in conversation context

# TTS Settings (using existing Chatterbox service)
ENABLE_TTS_FOR_CHAT_REPLIES: true
TTS_PLAYBACK_TIMEOUT_S: 30

# LLM Settings
DEFAULT_CONVERSATIONAL_LLM_MODEL: "your-model-name"
LLM_SERVER:
  endpoint: "http://localhost:11434/v1"  # Ollama endpoint
  timeout: 45
```

### 2. Run the Enhanced Voice Bot

You have several options to run the voice bot:

#### Option A: Use the Enhanced Voice Bot (Recommended)
```bash
python enhanced_voice_bot.py
```

#### Option B: Integrate with Existing Bot
```python
# In your existing bot file
from services.voice_chat_service import VoiceChatService

# Initialize and use the voice chat service
voice_chat = VoiceChatService(tts_service, llm_service, memory_service)
await voice_chat.initialize()
```

#### Option C: Use Your Simple Test Bot
You can enhance your existing `test_bot.py` with voice functionality.

## 🎤 Voice Chat Commands

Once the bot is running, use these Discord commands:

### Basic Commands
- `!join` - Join your current voice channel and start voice chat
- `!leave` - Leave the voice channel and stop voice chat
- `!voice_status` - Show current voice chat status
- `!test_voice` - Test TTS functionality

### Advanced Commands
- `!chat <message>` - Text chat that also plays TTS if in voice
- `!clear_context` - Clear conversation context memory

### Voice Interaction
1. **Join a voice channel** in Discord
2. **Use `!join`** to have DanzarAI join your channel
3. **Start talking!** The bot will:
   - Listen for your speech
   - Transcribe it using Whisper STT
   - Process it through your LLM
   - Respond with TTS audio
   - Continue listening for your next message

## 🔧 Configuration Options

### Voice Activity Detection (VAD)
```yaml
OWW_VAD_THRESHOLD: 1  # 0=least sensitive, 3=most sensitive
```

### Whisper STT Models
```yaml
WHISPER_MODEL_SIZE: "base.en"
```
Available models (size vs accuracy vs speed):
- `tiny` - Fastest, least accurate
- `base` - Good balance
- `small` - Better accuracy
- `medium` - High accuracy, slower
- `large` - Best accuracy, slowest

### Context Memory
```yaml
SHORT_TERM_SIZE: 6  # Messages kept in conversation context
```

### Auto-Management
```yaml
DISCORD_BOT_AUTO_REJOIN_ENABLED: true  # Auto-join configured channel on startup
DISCORD_AUTO_LEAVE_TIMEOUT_S: 600      # Auto-leave if alone (seconds)
```

## 🛠️ Architecture Overview

The voice chat system follows DanzarAI's service architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Discord Bot   │───▶│ VoiceChatService│───▶│   TTS Service   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   LLM Service   │───▶│ Memory Service  │
                       └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │ Whisper STT     │
                       └─────────────────┘
```

### Key Components:

1. **VoiceChatService**: Orchestrates the voice conversation pipeline
2. **Whisper STT**: Speech-to-text transcription
3. **LLM Service**: Your existing Ollama/OpenAI compatible service
4. **TTS Service**: Your existing Chatterbox/Piper service
5. **Memory Service**: Context management and RAG

## 🔍 Troubleshooting

### Common Issues

#### 1. Bot can't hear audio
- Ensure you have the correct Discord permissions
- Check that `discord-ext-voice-recv` is installed
- Verify FFmpeg is available

#### 2. TTS not working
- Check Chatterbox service is running
- Verify TTS_SERVER configuration in settings
- Test with `!test_voice` command

#### 3. STT not transcribing
- Check Whisper model is downloaded
- Verify microphone permissions in Discord
- Try a larger Whisper model for better accuracy

#### 4. LLM not responding
- Ensure Ollama is running and accessible
- Check LLM_SERVER endpoint configuration
- Verify the model name exists in Ollama

### Debug Logging

Enable debug logging in your configuration:

```yaml
LOG_LEVEL: DEBUG
```

Check logs in the `logs/` directory for detailed error information.

### Performance Optimization

#### For Better Response Time:
- Use smaller Whisper models (`tiny`, `base`)
- Optimize LLM model size
- Reduce context buffer size

#### For Better Accuracy:
- Use larger Whisper models (`medium`, `large`)
- Increase VAD sensitivity
- Use more powerful LLM models

## 🚀 Advanced Usage

### Multi-User Support

The system automatically handles multiple users by:
- Tracking user names in context
- Managing turn-taking (one speaker at a time)
- Maintaining conversation flow

### Custom Wake Words

To add wake word detection:

```yaml
WAKE_WORD_ENABLED: true
OWW_CUSTOM_MODEL_NAME: "Danzar"
OWW_CUSTOM_MODEL_PATH: "./models/Danzar.tflite"
```

### Session Summarization

For long conversations, the system can summarize context:

```yaml
AGENTIC_MEMORY:
  enabled: true
  enable_summarization: true
  buffer_max_tokens: 2000
```

## 🎮 Gaming Integration

The voice chat system integrates seamlessly with DanzarAI's gaming features:

- **Game-specific profiles**: Use different voice personalities per game
- **Screen capture integration**: Voice commentary on gameplay
- **Memory persistence**: Remember gaming sessions and strategies

Configure game-specific voice settings in `config/profiles/your_game.yaml`:

```yaml
# Voice-specific overrides
conversational_llm_model: "gaming-assistant-model"
system_prompt_chat: "You are Danzar, an experienced gaming companion..."
conversational_max_tokens: 150
conversational_temperature: 0.8
```

## 📝 Next Steps

1. **Test the basic functionality** with `!join` and `!test_voice`
2. **Customize the system prompts** for your gaming style
3. **Experiment with different models** for optimal performance
4. **Set up game-specific profiles** for different experiences
5. **Configure auto-join** for seamless gaming sessions

## 🆘 Support

If you encounter issues:

1. Check the logs in `logs/debug/` and `logs/errors/`
2. Verify all services are running with `!voice_status`
3. Test individual components with debug commands
4. Review the configuration against this guide

The voice chat system is designed to work with your existing DanzarAI setup, so most issues are related to service connectivity or configuration mismatches.

---

**Happy Gaming with DanzarAI Voice Chat! 🎮🎤** 