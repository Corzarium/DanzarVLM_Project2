@echo off
chcp 65001 >nul
echo ========================================
echo DanzarAI - Manual Server Startup
echo ========================================
echo.

echo 🧹 Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8083') do (
    echo Killing process %%a on port 8083
    taskkill /F /PID %%a 2>nul
)

timeout /t 2 /nobreak >nul

echo.
echo 🚀 Starting Required Servers...
echo.

echo 1️⃣ Starting Qwen2.5-VL CUDA Server (Port 8083)...
echo    This may take 30-60 seconds to load the model...
start "Qwen2.5-VL Server" cmd /k "cd /d %~dp0llama-cpp-cuda && llama-server.exe --model ..\models-gguf\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj ..\models-gguf\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf --port 8083 --host 0.0.0.0 --n-gpu-layers 99 --ctx-size 4096 --threads 8"

echo.
echo ⏳ Waiting 15 seconds for Qwen2.5-VL server to start...
timeout /t 15 /nobreak >nul

echo.
echo ✅ Server should now be running:
echo    - Qwen2.5-VL: http://localhost:8083
echo.
echo 🎮 You can now run the main bot with:
echo    start_danzar_fixed.bat
echo.
echo 💡 To test server manually:
echo    - Qwen2.5-VL: curl http://localhost:8083/v1/models
echo.
pause 