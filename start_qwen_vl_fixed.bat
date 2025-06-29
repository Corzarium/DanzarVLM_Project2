@echo off
echo 🚀 Starting Qwen2.5-VL Server with Multi-GPU Support...

REM Check if CUDA is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ NVIDIA GPU not detected. Please check your GPU drivers.
    pause
    exit /b 1
)

echo ✅ NVIDIA GPU detected
echo 📊 GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo.
echo 🔧 Starting Qwen2.5-VL Server on GPU 1 (RTX 4070 SUPER) - the faster card...
echo 📍 Server will be available at: http://localhost:8083
echo 🎯 Using GPU 1 (RTX 4070 SUPER) for Qwen2.5-VL - faster performance
echo 💾 Leaving GPU 0 (RTX 2080 Ti) free for Whisper

REM Start the Qwen2.5-VL server on GPU 1 (faster card)
cd llama-cpp-cuda

REM Use CUDA_VISIBLE_DEVICES to select GPU 1 (RTX 4070 SUPER)
set CUDA_VISIBLE_DEVICES=1

REM Use llama-server with GPU 1 configuration (faster RTX 4070 SUPER)
llama-server.exe ^
    --model models-gguf\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf ^
    --mmproj models-gguf\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf ^
    --host 0.0.0.0 ^
    --port 8083 ^
    --n-gpu-layers 99 ^
    --ctx-size 4096 ^
    --threads 8 ^
    --temp 0.7 ^
    --top-p 0.9 ^
    --repeat-penalty 1.1 ^
    --server-timeout 120

echo.
echo 🛑 Qwen2.5-VL Server stopped.
pause 