# PowerShell script to start DanzarVLM services with GPU support

Write-Host "🚀 Starting DanzarVLM Services with GPU Support..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if nvidia-docker is available
try {
    docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi | Out-Null
    Write-Host "✅ GPU support available" -ForegroundColor Green
} catch {
    Write-Host "⚠️  GPU support not available. Services will run on CPU." -ForegroundColor Yellow
    Write-Host "   Make sure nvidia-docker2 is installed for GPU acceleration." -ForegroundColor Yellow
}

# Create necessary directories
New-Item -ItemType Directory -Force -Path "docker\chatterbox\voices" | Out-Null
New-Item -ItemType Directory -Force -Path "docker\chatterbox\models" | Out-Null

# Stop any existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down

# Build and start services
Write-Host "🔧 Building and starting services..." -ForegroundColor Cyan
docker-compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Cyan

# Wait for Chatterbox TTS
Write-Host "   Checking Chatterbox TTS..." -ForegroundColor Gray
for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8055/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ Chatterbox TTS is ready" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }
    
    if ($i -eq 60) {
        Write-Host "   ❌ Chatterbox TTS failed to start" -ForegroundColor Red
        docker-compose logs chatterbox
        exit 1
    }
    Start-Sleep -Seconds 2
}

# Wait for Ollama
Write-Host "   Checking Ollama..." -ForegroundColor Gray
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ Ollama is ready" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }
    
    if ($i -eq 30) {
        Write-Host "   ❌ Ollama failed to start" -ForegroundColor Red
        docker-compose logs ollama
        exit 1
    }
    Start-Sleep -Seconds 2
}

# Wait for Qdrant
Write-Host "   Checking Qdrant..." -ForegroundColor Gray
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ Qdrant is ready" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }
    
    if ($i -eq 30) {
        Write-Host "   ❌ Qdrant failed to start" -ForegroundColor Red
        docker-compose logs qdrant
        exit 1
    }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "🎉 All services are running!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Service Status:" -ForegroundColor Cyan
Write-Host "   • Chatterbox TTS: http://localhost:8055" -ForegroundColor White
Write-Host "   • Ollama LLM:     http://localhost:11434" -ForegroundColor White
Write-Host "   • Qdrant Vector:  http://localhost:6333" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To check logs:" -ForegroundColor Cyan
Write-Host "   docker-compose logs -f [service_name]" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop services:" -ForegroundColor Cyan
Write-Host "   docker-compose down" -ForegroundColor White
Write-Host ""

# Test TTS service
Write-Host "🧪 Testing TTS service..." -ForegroundColor Cyan
try {
    $body = @{
        text = "Hello, this is a test of the Chatterbox TTS system running on GPU."
        exaggeration = 0.3
        cfg_weight = 0.5
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "http://localhost:8055/tts" -Method POST -Body $body -ContentType "application/json" -OutFile "test_tts.wav"
    
    if (Test-Path "test_tts.wav" -and (Get-Item "test_tts.wav").Length -gt 0) {
        Write-Host "✅ TTS test successful! Generated test_tts.wav" -ForegroundColor Green
        Remove-Item "test_tts.wav" -Force
    } else {
        Write-Host "❌ TTS test failed" -ForegroundColor Red
        Write-Host "Check Chatterbox logs: docker-compose logs chatterbox" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ TTS test failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check Chatterbox logs: docker-compose logs chatterbox" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Ready to start DanzarVLM!" -ForegroundColor Green
Write-Host "   python DanzarVLM.py --log-level INFO" -ForegroundColor White 