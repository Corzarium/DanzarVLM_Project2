# DanzarVLM Virtual Audio Cable Setup Guide

## Overview

This version of DanzarVLM captures audio from virtual audio cables (like VB-Cable, VoiceMeeter, or Windows Stereo Mix) instead of Discord voice channels. This provides much more flexibility in audio routing and allows you to capture audio from any source.

## Architecture

```
Audio Source → Virtual Audio Cable → DanzarVLM → STT (Whisper) → LLM → TTS → Discord Bot
```

**Benefits:**
- Capture audio from any application (games, voice chat, system audio)
- No need to be in Discord voice channels for input
- More flexible audio routing
- Better audio quality control
- Can process multiple audio sources

## Prerequisites

### 1. Virtual Audio Cable Software

Choose one of these options:

#### Option A: VB-Audio Virtual Cable (Recommended)
- **Free** virtual audio cable
- Download from: https://vb-audio.com/Cable/
- Install `VBCABLE_Driver_Pack45.zip`
- Restart computer after installation
- Creates "CABLE Input" and "CABLE Output" devices

#### Option B: VoiceMeeter (Advanced)
- More complex but more powerful
- Download from: https://vb-audio.com/Voicemeeter/
- Includes virtual cables and audio mixing
- Choose: Voicemeeter, Voicemeeter Banana, or Voicemeeter Potato

#### Option C: Windows Stereo Mix (Basic)
- Built into Windows but often disabled
- Right-click speaker icon → Sounds → Recording tab
- Right-click empty space → Show Disabled Devices
- Enable "Stereo Mix" if available

### 2. Python Dependencies

All required dependencies are already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies for virtual audio:
- `sounddevice` - Audio capture
- `numpy` - Audio processing
- `torch` - VAD (Voice Activity Detection)
- `openai-whisper` - Speech-to-text

## Setup Instructions

### Step 1: Configure Virtual Audio Cable

#### For VB-Cable:
1. **Set up audio routing:**
   - Right-click speaker icon in system tray
   - Select "Open Sound settings"
   - Under "Output", select your normal speakers/headphones as default
   - Under "Input", you'll see "CABLE Output" - this is what DanzarVLM will capture from

2. **Route application audio to VB-Cable:**
   - For **game audio**: In game settings, set audio output to "CABLE Input (VB-Audio Virtual Cable)"
   - For **Discord/voice chat**: Set Discord output device to "CABLE Input"
   - For **system audio**: Set Windows default playback device to "CABLE Input"

3. **Enable monitoring (so you can hear audio):**
   - Right-click speaker icon → Sounds → Recording tab
   - Find "CABLE Output" → Properties
   - Go to "Listen" tab
   - Check "Listen to this device"
   - Set "Playback through this device" to your speakers/headphones
   - Set level to 100%

### Step 2: Test Audio Setup

1. **List available audio devices:**
   ```bash
   python DanzarVLM_VirtualAudio.py --list-devices
   ```

2. **Look for virtual audio devices:**
   - Find "CABLE Output" or similar
   - Note the device ID number
   - Virtual devices will be marked with ⭐

### Step 3: Configure DanzarVLM

1. **Update your `config/global_settings.yaml`:**
   ```yaml
   # Discord Bot Configuration
   DISCORD_BOT_TOKEN: "your_bot_token_here"
   DISCORD_GUILD_ID: your_guild_id
   DISCORD_TEXT_CHANNEL_ID: your_text_channel_id
   DISCORD_VOICE_CHANNEL_ID: your_voice_channel_id  # For TTS output
   
   # TTS Configuration
   TTS_ENDPOINT: "http://localhost:8055/tts"  # Chatterbox TTS
   
   # LLM Configuration  
   LLM_ENDPOINT: "http://192.168.0.102:1234"  # LM Studio
   
   # Virtual Audio Settings
   VIRTUAL_AUDIO_DEVICE_ID: null  # Auto-detect, or specify device ID
   SAMPLE_RATE: 44100
   CHANNELS: 2
   ```

## Usage

### Start DanzarVLM Virtual Audio

1. **Auto-detect virtual audio device:**
   ```bash
   python DanzarVLM_VirtualAudio.py
   ```

2. **Specify specific audio device:**
   ```bash
   python DanzarVLM_VirtualAudio.py --device 5
   ```

3. **List available devices first:**
   ```bash
   python DanzarVLM_VirtualAudio.py --list-devices
   ```

### Discord Bot Commands

Once DanzarVLM is running:

1. **Connect Discord bot to voice channel:**
   ```
   !connect
   ```

2. **Test TTS:**
   ```
   !say Hello, this is a test message!
   ```

3. **Disconnect when done:**
   ```
   !disconnect
   ```

### Complete Workflow

1. **Start DanzarVLM:**
   ```bash
   python DanzarVLM_VirtualAudio.py
   ```

2. **Connect Discord bot to voice channel:**
   - Join a voice channel in Discord
   - Type `!connect` in text channel

3. **Route audio to virtual cable:**
   - Set your game/application audio output to "CABLE Input"
   - Or set Windows default playback to "CABLE Input" for all system audio

4. **Start talking/playing:**
   - DanzarVLM will automatically detect speech
   - Transcribe with Whisper
   - Process with LLM
   - Respond via TTS in Discord voice channel

## Common Audio Routing Scenarios

### Scenario 1: Game Commentary
```
Game Audio Output → CABLE Input → CABLE Output → DanzarVLM
DanzarVLM Response → Discord Voice Channel
```

### Scenario 2: Voice Chat Assistant
```
Discord Voice Input → CABLE Input → CABLE Output → DanzarVLM  
DanzarVLM Response → Discord Voice Channel
```

### Scenario 3: System Audio Monitoring
```
All System Audio → CABLE Input → CABLE Output → DanzarVLM
DanzarVLM Response → Discord Voice Channel
```

## Troubleshooting

### No Audio Detected
1. **Check device selection:**
   ```bash
   python DanzarVLM_VirtualAudio.py --list-devices
   ```
   
2. **Verify VB-Cable is working:**
   - Play audio from any application
   - Set that application's output to "CABLE Input"
   - Check Windows Sound settings → Recording → CABLE Output should show activity

3. **Check audio levels:**
   - Windows Sound settings → Recording → CABLE Output → Properties → Levels
   - Set both sliders to 100%

### Poor Speech Recognition
1. **Ensure good audio quality:**
   - Clear, loud speech
   - Minimal background noise
   - Good microphone if using voice input

2. **Adjust VAD settings in code:**
   - Lower `speech_threshold` for more sensitive detection
   - Adjust `min_speech_duration` and `max_silence_duration`

### Discord TTS Not Working
1. **Check Discord bot permissions:**
   - Bot needs "Connect" and "Speak" permissions in voice channel
   - Bot needs "Send Messages" permission in text channel

2. **Verify TTS service:**
   - Ensure Chatterbox TTS is running on localhost:8055
   - Test with `!say test message`

### Audio Feedback/Echo
1. **Disable audio monitoring in wrong direction:**
   - Don't set DanzarVLM's output back to CABLE Input
   - Only monitor CABLE Output → your speakers

2. **Check Discord echo cancellation:**
   - Enable "Echo Cancellation" in Discord voice settings

## Advanced Configuration

### Multiple Virtual Cables
For complex setups, install VB-Cable A+B or C+D:
- Route different applications to different cables
- Capture from multiple sources simultaneously

### VoiceMeeter Integration
Use VoiceMeeter for advanced audio mixing:
- Mix multiple audio sources
- Apply effects and filters
- Route to multiple outputs

### Custom VAD Settings
Modify VAD parameters in the code:
```python
# In VirtualAudioCapture class
self.speech_threshold = 0.3      # Lower = more sensitive
self.silence_threshold = 0.2     # Lower = more sensitive  
self.min_speech_duration = 0.5   # Minimum speech length
self.max_silence_duration = 2.0  # Silence before processing
```

## Performance Tips

1. **Audio Buffer Settings:**
   - Increase `chunk_size` for stability
   - Decrease for lower latency

2. **Whisper Model Selection:**
   - `tiny` - Fastest, least accurate
   - `base` - Good balance (default)
   - `small` - Better accuracy, slower
   - `medium` - High accuracy, much slower

3. **System Resources:**
   - Close unnecessary audio applications
   - Use dedicated audio device if possible
   - Monitor CPU usage during processing

## Security Notes

- Virtual audio cables can capture any system audio
- Be aware of what applications are routing through the cable
- DanzarVLM processes audio locally (no cloud services)
- Audio is processed in real-time and not stored permanently 