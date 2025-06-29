@echo off
echo 🚀 Starting Qwen2.5-VL Server with Memory Optimization...
echo.

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
echo 🔧 Starting Qwen2.5-VL Server with memory optimization...
echo 📍 Server will be available at: http://localhost:8083
echo 💾 Using memory-optimized settings for better resource sharing

REM Start the Qwen2.5-VL server with memory optimization
cd llama-cpp-cuda

REM Use CUDA_VISIBLE_DEVICES to select GPU 1 (RTX 4070 SUPER)
REM Use memory-optimized settings
set CUDA_VISIBLE_DEVICES=1

REM Start with memory optimization flags
llama-server.exe ^
  --model models-gguf\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf ^
  --mmproj models-gguf\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf ^
  --host 0.0.0.0 ^
  --port 8083 ^
  --ctx-size 2048 ^
  --gpu-layers 50 ^
  --threads 4 ^
  --temp 0.7 ^
  --repeat-penalty 1.1 ^
  --n-predict 256 ^
  --n-keep 128 ^
  --rope-scaling linear ^
  --rope-freq-base 10000 ^
  --rope-freq-scale 0.5 ^
  --mul-mat-q ^
  --no-mmap ^
  --no-mlock

echo.
echo 🔥 Qwen2.5-VL Server stopped.
pause 