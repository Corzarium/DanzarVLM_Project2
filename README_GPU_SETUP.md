# DanzarVLM GPU-Accelerated TTS Setup

This guide will help you set up Chatterbox TTS with GPU acceleration on your RTX 2080 Ti to fix the TTS issues.

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** with WSL2 backend
2. **NVIDIA Container Toolkit** for GPU support
3. **RTX 2080 Ti** with latest drivers

### Installation Steps

#### 1. Install NVIDIA Container Toolkit

```powershell
# Download and install from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# Or use chocolatey:
choco install nvidia-docker
```

#### 2. Start Services

```powershell
# Run the PowerShell startup script
.\start_services.ps1
```

#### 3. Verify Services

The script will automatically check all services. You should see:
- ✅ Chatterbox TTS is ready (http://localhost:8055)
- ✅ Ollama is ready (http://localhost:11434)  
- ✅ Qdrant is ready (http://localhost:6333)

#### 4. Start DanzarVLM

```powershell
python DanzarVLM.py --log-level INFO
```

## 🔧 Configuration Changes Made

### TTS Service Updates

- **Provider**: Changed to Chatterbox TTS with GPU acceleration
- **Endpoint**: `http://localhost:8055/tts` (Docker service)
- **Parameters**: Optimized for natural speech:
  - `exaggeration: 0.3` (reduced from 0.5)
  - `cfg_weight: 0.5` (classifier-free guidance)
  - `timeout: 60s` (increased for GPU processing)

### Docker Services

- **Chatterbox**: Python 3.11 with CUDA 11.8, GPU device 0
- **Ollama**: Shared GPU access for LLM processing
- **Qdrant**: Vector database for RAG functionality

### Pipeline Processing

- **Batch Size**: 3 sentences processed simultaneously
- **TTS Generation**: Background threads with 60s timeout
- **Audio Format**: WAV with FFmpeg conversion for Discord

## 🐛 Troubleshooting

### TTS Not Playing

1. **Check Service Status**:
   ```powershell
   docker-compose ps
   docker-compose logs chatterbox
   ```

2. **Test TTS Directly**:
   ```powershell
   curl -X POST http://localhost:8055/tts -H "Content-Type: application/json" -d '{"text":"test","exaggeration":0.3}' --output test.wav
   ```

3. **Check GPU Usage**:
   ```powershell
   nvidia-smi
   ```

### Discord Voice Issues

1. **Voice Client Disconnections**: The system will auto-reconnect
2. **Audio Format Errors**: Fixed with FFmpeg conversion
3. **Queue Management**: Improved error handling for task_done()

### Memory Service Errors

- Fixed `query_rag` method calls to use correct `query` method
- Updated Qdrant service integration

## 📊 Performance Improvements

### Before (Legacy TTS)
- ❌ 25-30s timeouts
- ❌ Connection failures
- ❌ Audio format issues
- ❌ Queue management errors

### After (GPU Chatterbox)
- ✅ GPU-accelerated generation
- ✅ Reliable service endpoints
- ✅ Proper audio format handling
- ✅ Robust error handling
- ✅ Pipeline processing with background TTS generation

## 🎛️ Advanced Configuration

### Voice Cloning

Upload custom voice files to use with Chatterbox:

```powershell
# Upload a voice file for cloning
curl -X POST http://localhost:8055/upload_voice -F "voice_file=@your_voice.wav"

# List available voices
curl http://localhost:8055/voices
```

### GPU Memory Management

If you encounter GPU memory issues:

```yaml
# In docker-compose.yml, adjust memory limits
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256  # Reduce from 512
```

### Service Scaling

To run services on different GPUs:

```yaml
# Chatterbox on GPU 0
environment:
  - CUDA_VISIBLE_DEVICES=0

# Ollama on GPU 1 (if available)
environment:
  - CUDA_VISIBLE_DEVICES=1
```

## 🔄 Service Management

### Start Services
```powershell
.\start_services.ps1
```

### Stop Services
```powershell
docker-compose down
```

### Restart Single Service
```powershell
docker-compose restart chatterbox
```

### View Logs
```powershell
docker-compose logs -f chatterbox
docker-compose logs -f ollama
docker-compose logs -f qdrant
```

### Update Services
```powershell
docker-compose pull
docker-compose up -d
```

## 📈 Monitoring

### Check Service Health
- Chatterbox: http://localhost:8055/health
- Ollama: http://localhost:11434/api/tags
- Qdrant: http://localhost:6333/health

### GPU Monitoring
```powershell
# Real-time GPU usage
nvidia-smi -l 1

# Docker GPU usage
docker stats
```

## 🆘 Support

If you encounter issues:

1. Check the service logs: `docker-compose logs [service_name]`
2. Verify GPU drivers: `nvidia-smi`
3. Test individual services using the health endpoints
4. Restart services: `docker-compose restart`

The new setup should resolve all TTS playback issues and provide much faster, more reliable audio generation using your RTX 2080 Ti! 