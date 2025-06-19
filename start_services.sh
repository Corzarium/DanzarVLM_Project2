#!/bin/bash

echo "🚀 Starting DanzarVLM Services with GPU Support..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  GPU support not available. Services will run on CPU."
    echo "   Make sure nvidia-docker2 is installed for GPU acceleration."
fi

# Create necessary directories
mkdir -p docker/chatterbox/voices
mkdir -p docker/chatterbox/models

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🔧 Building and starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."

# Wait for Chatterbox TTS
echo "   Checking Chatterbox TTS..."
for i in {1..60}; do
    if curl -s http://localhost:8055/health > /dev/null 2>&1; then
        echo "   ✅ Chatterbox TTS is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ❌ Chatterbox TTS failed to start"
        docker-compose logs chatterbox
        exit 1
    fi
    sleep 2
done

# Wait for Ollama
echo "   Checking Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Ollama failed to start"
        docker-compose logs ollama
        exit 1
    fi
    sleep 2
done

# Wait for Qdrant
echo "   Checking Qdrant..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "   ✅ Qdrant is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Qdrant failed to start"
        docker-compose logs qdrant
        exit 1
    fi
    sleep 2
done

echo ""
echo "🎉 All services are running!"
echo ""
echo "📊 Service Status:"
echo "   • Chatterbox TTS: http://localhost:8055"
echo "   • Ollama LLM:     http://localhost:11434"
echo "   • Qdrant Vector:  http://localhost:6333"
echo ""
echo "🔧 To check logs:"
echo "   docker-compose logs -f [service_name]"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"
echo ""

# Test TTS service
echo "🧪 Testing TTS service..."
curl -X POST http://localhost:8055/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a test of the Chatterbox TTS system running on GPU.","exaggeration":0.3,"cfg_weight":0.5}' \
  --output test_tts.wav \
  --silent

if [ -f test_tts.wav ] && [ -s test_tts.wav ]; then
    echo "✅ TTS test successful! Generated test_tts.wav"
    rm test_tts.wav
else
    echo "❌ TTS test failed"
    echo "Check Chatterbox logs: docker-compose logs chatterbox"
fi

echo ""
echo "🚀 Ready to start DanzarVLM!"
echo "   python DanzarVLM.py --log-level INFO" 