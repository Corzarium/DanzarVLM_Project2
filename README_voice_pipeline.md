# Danzar Voice Pipeline

A real-time multimodal voice analysis pipeline for Discord bots that performs speech-to-text, emotion recognition, laughter detection, and generates contextual responses using local VLM models.

## 🚀 Features

- **Real-time Audio Capture**: Captures live audio from Discord voice channels
- **Parallel Analysis**: Simultaneously processes:
  - **Speech-to-Text**: Using OpenAI Whisper
  - **Emotion Recognition**: Using SpeechBrain's pre-trained models
  - **Laughter Detection**: Using jrgillick's laughter-detection model
- **Multimodal Prompting**: Correlates all analysis results by timestamp
- **Local VLM Integration**: Sends prompts to local Qwen2.5-VL API
- **TTS Response**: Synthesizes and plays back responses via Chatterbox TTS
- **Async Architecture**: Non-blocking processing for smooth performance

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- FFmpeg (for audio processing)

### API Endpoints
- **Qwen2.5-VL API**: `http://localhost:8083/chat/completions`
- **Chatterbox TTS**: `http://localhost:8055/tts`

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <your-repo>
cd DanzarVLM_Project
pip install -r requirements_voice_pipeline.txt
```

### 2. Environment Configuration
Create a `.env` file:
```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id
DISCORD_TEXT_CHANNEL_ID=your_text_channel_id
```

### 3. Start Required Services

#### Qwen2.5-VL API Server
```bash
# Start your local Qwen2.5-VL server on port 8083
# This should match your existing DanzarAI setup
```

#### Chatterbox TTS Server
```bash
# Start your Chatterbox TTS server on port 8055
# This should match your existing DanzarAI setup
```

## 🎯 Usage

### Basic Usage
```bash
python danzar_voice_pipeline.py
```

### Configuration Options
You can modify the configuration in the `main()` function:

```python
config = {
    'command_prefix': '!',
    'voice_channel_id': 123456789,  # Your voice channel ID
    'text_channel_id': 987654321,   # Your text channel ID
    'whisper_model': 'base',        # Options: tiny, base, small, medium, large
    'vlm_endpoint': 'http://localhost:8083/chat/completions',
    'tts_endpoint': 'http://localhost:8055/tts'
}
```

### Discord Commands
- `!join [channel]` - Join a voice channel
- `!leave` - Leave the current voice channel
- `!status` - Show pipeline status and queue information

## 🔧 Architecture

### Pipeline Flow
```
Discord Voice → Audio Capture → Parallel Analysis → Multimodal Prompt → VLM → TTS → Discord Voice
     ↓              ↓                ↓                    ↓           ↓      ↓
  Raw PCM      Audio Chunks    STT + Emotion +    Timestamped    JSON    Audio
  Audio                        Laughter Detection   Prompt       Response  Response
```

### Key Components

#### 1. Audio Capture (`audio_capture_loop`)
- Captures raw PCM audio from Discord voice client
- Buffers audio in 100ms chunks
- Maintains timestamp correlation

#### 2. Parallel Analysis (`audio_processing_loop`)
- **WhisperTranscriber**: Converts speech to text
- **EmotionRecognizer**: Classifies speaker emotions
- **LaughterDetector**: Detects laughter segments
- All analyses run concurrently using `asyncio.gather()`

#### 3. Multimodal Prompting (`analysis_processing_loop`)
- Correlates all analysis results by timestamp
- Builds prompts like:
  ```
  [12.3s] TEXT: "That was hilarious!" [EMOTION: joy (0.92)] [LAUGHTER detected at 12.0–12.5s]
  ```

#### 4. VLM Integration (`VLMClient`)
- Sends multimodal prompts to local Qwen2.5-VL API
- Receives contextual responses
- Handles API errors and timeouts

#### 5. TTS Response (`response_processing_loop`)
- Synthesizes speech using Chatterbox TTS
- Plays audio back through Discord voice channel
- Sends text responses to Discord text channel

## 📊 Performance Optimization

### Model Selection
- **Whisper**: Choose model size based on accuracy vs. speed needs
  - `tiny`: Fastest, lowest accuracy
  - `base`: Good balance (default)
  - `medium`: Higher accuracy, slower
  - `large`: Highest accuracy, slowest

### GPU Acceleration
Enable CUDA for faster processing:
```python
# In requirements_voice_pipeline.txt
torch-cuda  # Uncomment for GPU acceleration
```

### Queue Management
- Audio queue: 100 chunks buffer
- Analysis queue: 50 results buffer
- Response queue: Unlimited (processes immediately)

## 🐛 Troubleshooting

### Common Issues

#### 1. "Whisper model not loaded"
```bash
# Ensure you have enough disk space for model downloads
# Check internet connection for initial model download
```

#### 2. "VLM API connection failed"
```bash
# Verify Qwen2.5-VL server is running on port 8083
# Check firewall settings
# Ensure API endpoint is correct in config
```

#### 3. "TTS synthesis failed"
```bash
# Verify Chatterbox TTS server is running on port 8055
# Check TTS server logs for errors
# Ensure text is not empty or too long
```

#### 4. "Discord connection issues"
```bash
# Verify bot token is correct
# Check bot permissions in Discord server
# Ensure bot has voice channel access
```

### Debug Mode
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Integration with Existing DanzarAI

This pipeline is designed to integrate with your existing DanzarAI system:

### Shared Components
- Uses same Discord bot token and channels
- Connects to existing Qwen2.5-VL API
- Uses existing Chatterbox TTS server
- Follows same configuration patterns

### Differences from Main Bot
- Focused on real-time voice analysis
- Parallel processing architecture
- Multimodal prompt generation
- Emotion and laughter detection

## 📈 Monitoring and Metrics

### Status Command Output
```
🎤 Danzar Voice Pipeline Status
├── Voice Connection: ✅ Connected
├── Channel: General
├── Audio Queue: 5 chunks
├── Analysis Queue: 2 results
└── Response Queue: 1 responses
```

### Log Messages
- `🎵 Audio processing loop started`
- `🧠 Analysis processing loop started`
- `🎵 Response processing loop started`
- `✅ Whisper transcription: "Hello there!"`
- `🎯 Emotion detected: joy (0.85)`
- `😄 Laughter detected (confidence: 0.92)`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **SpeechBrain** for emotion recognition
- **jrgillick** for laughter detection model
- **Discord.py** for Discord integration
- **Qwen2.5-VL** for multimodal understanding

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Join the Discord community

---

**Happy voice analyzing! 🎤🤖** 