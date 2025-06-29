# Discord Voice Sink Test Guide

## Overview

This guide helps you test the Discord voice sink integration with STT (Speech-to-Text) and LLM (Language Model) services without TTS (Text-to-Speech). The test verifies that:

1. ✅ Discord voice recording works properly
2. ✅ STT processes audio correctly  
3. ✅ LLM generates responses to transcribed text
4. ✅ No TTS is involved in this test

## Prerequisites

### 1. Discord Bot Setup
- Discord bot token configured
- Bot has proper permissions in your server
- Bot can join voice channels
- Bot can send messages in text channels

### 2. Environment Variables
Set your Discord bot token:
```bash
set DISCORD_BOT_TOKEN=your_actual_token_here
```

### 3. Configuration
Ensure your `config/global_settings.yaml` has:
```yaml
DISCORD_BOT_TOKEN: "your_token_here"  # Or use environment variable
DISCORD_GUILD_ID: 12345
DISCORD_TEXT_CHANNEL_ID: 12345
DISCORD_VOICE_CHANNEL_ID: 12345
```

## Test Files

### 1. Simple Configuration Test
**File:** `test_discord_config_simple.py`

**Purpose:** Verify basic Discord connectivity before running voice tests

**Usage:**
```bash
python test_discord_config_simple.py
```

**What it tests:**
- Discord bot token validity
- Bot can connect to Discord
- Bot can send messages to configured text channel
- Basic bot responsiveness with `!ping` command

### 2. Full Voice STT/LLM Test
**File:** `test_discord_voice_stt_llm.py`

**Purpose:** Complete voice recording, STT, and LLM integration test

**Usage:**
```bash
python test_discord_voice_stt_llm.py
```

**What it tests:**
- Voice channel connection
- Audio recording via py-cord
- STT transcription (external service or local Whisper)
- LLM response generation
- Streaming text responses to Discord

## Test Commands

### Simple Test Bot Commands
- `!ping` - Test bot responsiveness

### Voice Test Bot Commands
- `!test` - Check service status
- `!join` - Join voice channel and start recording
- `!leave` - Stop recording and leave voice channel
- `!text <message>` - Test LLM with text input (bypass STT)

## Test Workflow

### Step 1: Configuration Test
1. Run the simple configuration test:
   ```bash
   python test_discord_config_simple.py
   ```

2. Verify:
   - Bot connects successfully
   - Test message appears in your Discord text channel
   - `!ping` command responds

### Step 2: Voice Integration Test
1. Run the full voice test:
   ```bash
   python test_discord_voice_stt_llm.py
   ```

2. In Discord:
   - Use `!test` to check service status
   - Join a voice channel
   - Use `!join` to start recording
   - Speak clearly into your microphone
   - Use `!leave` to stop recording
   - Check the text channel for transcription and LLM response

### Step 3: Text-Only LLM Test
1. Use `!text "Hello, how are you?"` to test LLM without voice
2. Verify streaming responses appear in Discord

## Expected Results

### Successful Test Output
```
🧪 Starting Discord Voice STT/LLM Test
==================================================
✅ Discord token found: MTIzNDU2N...
📋 Guild ID: 12345
📋 Text Channel ID: 12345
📋 Voice Channel ID: 12345
🚀 Starting Discord bot...
✅ Bot ready as TestDiscordVoiceBot#1234
🔧 Initializing services...
✅ STT service initialized
✅ LLM service initialized
✅ Streaming LLM service initialized
```

### Discord Channel Output
```
🧪 Test Discord Voice Bot Status:
✅ STT Service: Available
✅ LLM Service: Available
✅ Streaming LLM: Available
✅ Local Whisper: Available

🎙️ Joined Voice Channel - Recording started! Speak now...

🎤 @User said: Hello, this is a test message

🤖 Danzar: Hello! I heard you say "Hello, this is a test message". How can I help you today?
```

## Troubleshooting

### Common Issues

#### 1. Discord Token Issues
**Error:** `Discord bot token not found!`
**Solution:** Set environment variable or update config file

#### 2. Voice Recording Issues
**Error:** `Failed to join voice channel`
**Solution:** 
- Check bot permissions
- Ensure bot can join voice channels
- Verify voice channel ID is correct

#### 3. STT Issues
**Error:** `No STT service available`
**Solution:**
- Check if external STT server is running
- Verify Whisper model is installed
- Check audio input device

#### 4. LLM Issues
**Error:** `LLM service failed to initialize`
**Solution:**
- Check LLM server connectivity
- Verify model configuration
- Check network connectivity

### Debug Commands
- Use `!test` to check service status
- Check console logs for detailed error messages
- Verify all required services are running

## Service Dependencies

### Required Services
1. **Discord Bot** - Main communication
2. **STT Service** - Speech-to-text conversion
   - External Whisper server (preferred)
   - Local Whisper model (fallback)
3. **LLM Service** - Response generation
   - Qwen2.5-VL server
   - Other configured LLM providers

### Optional Services
- **TTS Service** - Not used in this test
- **Memory Service** - Not used in this test
- **RAG Service** - Not used in this test

## Configuration Files

### Key Configuration Sections
```yaml
# Discord Configuration
DISCORD_BOT_TOKEN: "your_token_here"
DISCORD_GUILD_ID: 12345
DISCORD_TEXT_CHANNEL_ID: 12345
DISCORD_VOICE_CHANNEL_ID: 12345

# STT Configuration
EXTERNAL_SERVERS:
  WHISPER_STT_SERVER:
    endpoint: http://localhost:8084/transcribe
    enabled: false  # Will use local Whisper

# LLM Configuration
LLM_SERVER:
  endpoint: http://localhost:8083/chat/completions
  provider: qwen2.5-vl
```

## Next Steps

After successful testing:

1. **Integration Testing** - Test with main DanzarVLM application
2. **TTS Integration** - Add TTS back to the pipeline
3. **Memory Integration** - Add conversation memory
4. **RAG Integration** - Add knowledge retrieval
5. **Performance Testing** - Test with longer conversations

## Support

If you encounter issues:
1. Check the console logs for detailed error messages
2. Verify all services are running and accessible
3. Test each component individually
4. Check Discord bot permissions and configuration 