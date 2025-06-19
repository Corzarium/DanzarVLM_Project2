# Virtual Audio Integration for DanzarVLM

The main DanzarVLM program now includes integrated virtual audio capture functionality, allowing you to capture audio from virtual audio cables instead of Discord voice channels while still using Discord for TTS output.

## What Was Added

### 1. Virtual Audio Capture Class
- **VirtualAudioCapture**: Complete virtual audio capture system with VAD
- Supports VB-Cable, VoiceMeeter, Windows Stereo Mix, and other virtual audio devices
- Real-time Voice Activity Detection using Silero VAD
- Automatic speech segment processing

### 2. Enhanced Discord Bot Commands
- **!virtual list** - List all available virtual audio devices
- **!virtual start [device_id]** - Start virtual audio recording from specified device
- **!virtual stop** - Stop virtual audio recording
- **!virtual status** - Show virtual audio status
- **!status** - Enhanced to show both Discord and virtual audio modes

### 3. Command Line Options
- **--virtual-audio** - Enable virtual audio capture mode
- **--audio-device ID** - Specify audio device ID for virtual audio
- **--list-devices** - List available audio devices and exit

### 4. Dual Mode Operation
- **Discord Voice Mode**: Traditional Discord voice channel capture (default)
- **Virtual Audio Mode**: Capture from virtual audio cables
- Seamless switching between modes via commands or configuration

## Installation Requirements

```bash
# Install virtual audio support
pip install sounddevice

# Install VB-Cable (recommended virtual audio solution)
# Download from: https://vb-audio.com/Cable/
```

## Usage Examples

### 1. List Available Audio Devices
```bash
python DanzarVLM.py --list-devices
```

### 2. Start with Virtual Audio Mode
```bash
python DanzarVLM.py --virtual-audio --audio-device 8
```

### 3. Discord Commands
```
!virtual list                    # Show virtual audio devices
!virtual start 8                 # Start recording from device 8
!virtual stop                    # Stop virtual audio recording
!virtual status                  # Show virtual audio status
!status                         # Show overall bot status
```

## Configuration

Add to your `config/global_settings.yaml`:

```yaml
# Enable virtual audio mode by default
USE_VIRTUAL_AUDIO: true

# Other existing settings...
DISCORD_BOT_TOKEN: "your_token_here"
TTS_ENDPOINT: "http://localhost:8055/tts"
LLM_ENDPOINT: "http://192.168.0.102:1234/chat/completions"
```

## Audio Routing Scenarios

### 1. Game Audio Capture
```
Game Audio → VB-Cable Input → DanzarVLM (captures from VB-Cable Output)
```

### 2. Voice Chat Capture
```
Discord/TeamSpeak → VB-Cable Input → DanzarVLM (captures from VB-Cable Output)
```

### 3. System Audio Capture
```
System Audio → Stereo Mix → DanzarVLM (captures from Stereo Mix)
```

### 4. Multiple Source Capture
```
Multiple Apps → VoiceMeeter → Virtual Output → DanzarVLM
```

## Features

### Voice Activity Detection
- **Silero VAD**: Advanced speech detection
- **Smart Buffering**: Handles audio fragmentation
- **Configurable Thresholds**: Adjustable sensitivity
- **Natural Speech Flow**: Processes complete speech segments

### Audio Processing
- **High Quality**: 44.1kHz stereo capture
- **Real-time Processing**: Low-latency speech recognition
- **Noise Handling**: Basic noise reduction
- **Format Conversion**: Automatic audio format handling

### Integration Benefits
- **Unified Interface**: Single program for all audio modes
- **Service Integration**: Full STT → LLM → TTS pipeline
- **Discord Output**: TTS responses via Discord bot
- **Flexible Routing**: Capture from any audio source

## Troubleshooting

### Virtual Audio Not Available
```
❌ sounddevice not available - install with: pip install sounddevice
```
**Solution**: Install sounddevice library

### No Virtual Devices Found
```
⚠️ No virtual audio devices detected. Install VB-Cable or enable Stereo Mix.
```
**Solutions**:
- Install VB-Cable from https://vb-audio.com/Cable/
- Enable Windows Stereo Mix in Sound settings
- Install VoiceMeeter for advanced audio routing

### Device Selection Failed
```
❌ Failed to select device X
```
**Solutions**:
- Use `!virtual list` to see available devices
- Try different device IDs
- Check if device is in use by another application
- Restart the program

### Audio Quality Issues
- **Low Volume**: Increase virtual cable levels in Windows Sound settings
- **Distortion**: Lower input levels to prevent clipping
- **Latency**: Use smaller buffer sizes (advanced users)

## Technical Details

### Audio Pipeline
1. **Capture**: sounddevice captures from virtual audio device
2. **VAD Processing**: Silero VAD detects speech segments
3. **STT**: Whisper transcribes speech to text
4. **LLM**: Process text with language model
5. **TTS**: Generate speech response
6. **Output**: Play TTS via Discord bot

### Threading Model
- **Main Thread**: Discord bot and async operations
- **Audio Thread**: Real-time audio capture callback
- **Processing Thread**: VAD and speech processing
- **Queue System**: Thread-safe audio data transfer

### Memory Management
- **Circular Buffers**: Prevent memory leaks
- **Automatic Cleanup**: Resources freed on stop
- **Configurable Limits**: Maximum buffer sizes

## Comparison: Discord vs Virtual Audio

| Feature | Discord Voice | Virtual Audio |
|---------|---------------|---------------|
| **Audio Source** | Discord voice channels only | Any application |
| **Setup Complexity** | Simple | Moderate (requires virtual cables) |
| **Audio Quality** | Discord-compressed | High quality |
| **Flexibility** | Limited to Discord users | Any audio source |
| **Latency** | Higher (Discord processing) | Lower (direct capture) |
| **Multi-source** | Single voice channel | Multiple apps via mixer |

## Best Practices

1. **Use VB-Cable** for single application capture
2. **Use VoiceMeeter** for complex audio routing
3. **Monitor audio levels** to prevent clipping
4. **Test device selection** before long sessions
5. **Keep virtual cables updated** for stability

## Future Enhancements

- **Multiple Device Support**: Capture from multiple devices simultaneously
- **Audio Mixing**: Combine multiple audio sources
- **Advanced VAD**: Custom VAD models for specific use cases
- **Audio Effects**: Real-time audio processing
- **Recording**: Save audio segments for analysis

This integration provides a powerful, flexible audio capture system while maintaining the simplicity and functionality of the original DanzarVLM program. 