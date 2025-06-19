# DanzarAI faster-whisper Integration

## Overview

We've successfully integrated **faster-whisper** into DanzarAI, replacing the previous Vosk STT system. This provides significant performance improvements while maintaining high accuracy.

## Benefits of faster-whisper

Based on the [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) project:

- **4x faster** than OpenAI Whisper with same accuracy
- **Lower memory usage** compared to standard Whisper
- **GPU acceleration** with FP16/INT8 quantization support
- **Built-in VAD** (Voice Activity Detection) for better speech segmentation
- **CTranslate2 backend** for optimized inference
- **Real-time processing** capabilities

## Performance Comparison

| Model | Speed Improvement | Memory Usage | Accuracy |
|-------|------------------|--------------|----------|
| faster-whisper tiny | ~4x faster | ~50% less | Same as Whisper |
| faster-whisper base | ~4x faster | ~50% less | Same as Whisper |
| faster-whisper small | ~4x faster | ~50% less | Same as Whisper |

## Installation

### 1. Install faster-whisper

```bash
pip install faster-whisper
```

### 2. Optional: GPU Support

For CUDA GPU acceleration:
```bash
# Ensure you have CUDA toolkit installed
pip install faster-whisper[gpu]
```

### 3. Test the Integration

Run the integration test:
```bash
python test_faster_whisper_integration.py
```

## Configuration

The faster-whisper service automatically detects the best configuration:

- **GPU Available**: Uses CUDA with FP16 precision
- **CPU Only**: Uses INT8 quantization for speed

### Model Sizes

- `tiny`: Fastest, good for real-time applications
- `base`: Balanced speed/accuracy (recommended)
- `small`: Better accuracy, slightly slower
- `medium`: High accuracy, moderate speed
- `large-v3`: Best accuracy, slower processing

## Integration Details

### Service Architecture

```
Discord Voice → VAD → faster-whisper STT → LLM → TTS → Discord
```

### Key Components

1. **FasterWhisperSTTService**: Main STT service
2. **VADVoiceReceiver**: Voice activity detection + audio processing
3. **Enhanced Audio Preprocessing**: Discord-optimized audio filtering
4. **Hallucination Detection**: Filters out common transcription errors

### Audio Processing Pipeline

1. **48kHz → 16kHz Resampling**: Optimized for faster-whisper
2. **Discord Audio Fixes**: Pre-emphasis, noise reduction, bandpass filtering
3. **VAD Integration**: Built-in voice activity detection
4. **Quality Filtering**: Removes low-confidence and hallucinated results

## Usage

### Discord Commands

```
!join          # Start voice processing with faster-whisper
!status        # Check faster-whisper service status
!virtual start # Use virtual audio input (VB-Cable)
!leave         # Stop voice processing
```

### Virtual Audio Support

Works with:
- VB-Audio VB-Cable
- Voicemeeter
- Windows Stereo Mix
- Any virtual audio device

## Performance Optimization

### CPU Optimization

```python
# Automatic configuration for CPU
device = "cpu"
compute_type = "int8"  # 8-bit quantization for speed
cpu_threads = 4        # Optimal thread count
```

### GPU Optimization

```python
# Automatic configuration for GPU
device = "cuda"
compute_type = "float16"  # FP16 for speed/accuracy balance
```

### Real-time Parameters

```python
transcribe_params = {
    "beam_size": 1,                    # Single beam for speed
    "best_of": 1,                      # Single candidate
    "temperature": 0.0,                # Deterministic output
    "condition_on_previous_text": False, # Prevent context bias
    "vad_filter": True,                # Built-in VAD
    "vad_parameters": {
        "min_silence_duration_ms": 500,  # Quick silence detection
        "speech_pad_ms": 400             # Minimal padding
    }
}
```

## Troubleshooting

### Common Issues

1. **Import Error**: Install faster-whisper
   ```bash
   pip install faster-whisper
   ```

2. **CUDA Issues**: Verify CUDA installation
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Model Download**: First run downloads models automatically
   - Models cached in `~/.cache/huggingface/`
   - Tiny model: ~39MB
   - Base model: ~74MB

### Performance Issues

1. **Slow Transcription**: 
   - Use smaller model (tiny/base)
   - Enable GPU acceleration
   - Check CPU thread count

2. **High Memory Usage**:
   - Use INT8 quantization on CPU
   - Use smaller model size
   - Close other applications

### Audio Quality Issues

1. **Poor Transcription**:
   - Check audio input levels
   - Verify virtual audio setup
   - Test with different VAD settings

2. **Hallucinations**:
   - Enhanced filtering is built-in
   - Adjust confidence thresholds
   - Check audio preprocessing

## Comparison with Previous System

### Vosk vs faster-whisper

| Feature | Vosk | faster-whisper |
|---------|------|----------------|
| Speed | Fast | 4x faster than Whisper |
| Accuracy | Good | Whisper-level accuracy |
| Model Size | 50MB | 39MB (tiny), 74MB (base) |
| GPU Support | No | Yes (CUDA) |
| Language Support | Limited | 99+ languages |
| Hallucinations | Some | Better filtering |
| Real-time | Yes | Yes, optimized |

### Migration Benefits

- **Better Accuracy**: Whisper-level transcription quality
- **Faster Processing**: 4x speed improvement over standard Whisper
- **GPU Acceleration**: Significant speedup on compatible hardware
- **Built-in VAD**: No separate VAD model needed
- **Better Discord Support**: Optimized for compressed audio

## Future Enhancements

1. **Streaming Transcription**: Real-time word-by-word output
2. **Multi-language Support**: Automatic language detection
3. **Custom Model Fine-tuning**: Domain-specific improvements
4. **Batch Processing**: Multiple audio streams simultaneously

## Resources

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [Whisper Model Cards](https://huggingface.co/openai)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

## Support

For issues with the faster-whisper integration:

1. Check the logs for error messages
2. Run the integration test: `python test_faster_whisper_integration.py`
3. Verify dependencies are installed correctly
4. Check Discord voice permissions and audio setup 