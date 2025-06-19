# Voice-Enabled Discord Bot

A Python-based conversational AI for Discord that operates via voice chat. The bot uses voice activity detection, speech-to-text, and text-to-speech to enable natural voice conversations.

## Features

- Voice Activity Detection (VAD) for detecting speech
- Speech-to-Text using OpenAI's Whisper model
- Text-to-Speech using Chatterbox
- Short-term memory for conversation context
- Long-term memory using Qdrant vector database
- Turn-taking behavior to avoid interrupting users
- Session summarization and logging
- RAG integration for relevant past conversations
- Language model integration using Ollama
- Modular design for easy component swapping

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed and available in PATH
- Discord bot token
- Ollama server running
- Qdrant server running
- Chatterbox server running

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-bot
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv-win
.venv-win\Scripts\activate

# Linux/WSL
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
DISCORD_TOKEN=your_discord_bot_token
DISCORD_GUILD_ID=your_guild_id
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id
```

## Configuration

The bot can be configured through the `config/settings.yaml` file. Key settings include:

- Discord settings (token, guild ID, voice channel ID)
- Voice settings (VAD parameters, sample rate)
- STT settings (Whisper model, language)
- TTS settings (engine, voice)
- LLM settings (model, endpoint)
- Memory settings (short-term memory size, RAG configuration)
- Logging settings (level, format, file path)

## Usage

1. Start the bot:
```bash
python -m voice_bot
```

2. Use Discord commands:
- `!join` - Bot joins your current voice channel
- `!leave` - Bot leaves the voice channel
- `!voices` - List available TTS voices

## Architecture

The bot follows a modular microservices architecture:

- `core/discord_bot.py` - Main bot client and service orchestration
- `services/vad_service.py` - Voice activity detection
- `services/stt_service.py` - Speech-to-text using Whisper
- `services/llm_service.py` - Language model integration
- `services/tts_service.py` - Text-to-speech using Chatterbox
- `services/memory_service.py` - Memory management (short-term and RAG)

## Development

### Adding New Features

1. Create a new service module in `services/`
2. Implement the service interface
3. Add service configuration to `settings.yaml`
4. Integrate the service in `discord_bot.py`

### Testing

Run tests using pytest:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 