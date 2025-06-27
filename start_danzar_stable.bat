@echo off
chcp 65001 >nul
echo ========================================
echo DanzarAI Voice Bot - Stable Startup Script
echo ========================================
echo.

echo 🔧 PRE-STARTUP CHECKS:
echo.

REM Check if Discord is running
tasklist /FI "IMAGENAME eq Discord.exe" 2>NUL | find /I /N "Discord.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ⚠️  WARNING: Discord is currently running
    echo 💡 For best voice connection results:
    echo    1. Close Discord completely
    echo    2. Restart Discord
    echo    3. Try joining voice channels manually first
    echo    4. Then run this script
    echo.
    pause
)

REM Kill any existing processes on our ports
echo 🧹 Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8083') do (
    echo Killing process %%a on port 8083
    taskkill /F /PID %%a 2>nul
)

timeout /t 3 /nobreak >nul

echo.
echo 🚀 STARTING REQUIRED SERVERS:
echo.

echo 1️⃣ Starting Qwen2.5-VL CUDA Server (Port 8083)...
start "Qwen2.5-VL Server" cmd /k "cd /d %~dp0llama-cpp-cuda && llama-server.exe --model ..\models-gguf\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj ..\models-gguf\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf --port 8083 --host 0.0.0.0 --n-gpu-layers 99 --ctx-size 4096 --threads 8"

echo.
echo ⏳ Waiting for server to start...
echo 💡 This may take 30-60 seconds for model to load
echo.

REM Wait for server to be ready
echo 🔍 Checking server availability...
:check_server
timeout /t 10 /nobreak >nul

REM Check Qwen2.5-VL server
curl -s http://localhost:8083/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⏳ Qwen2.5-VL server not ready yet...
    goto check_server
)

echo ✅ Server is ready!
echo.

echo 🎯 STARTING DANZARAI VOICE BOT:
echo.

REM Start the main bot
cd /d %~dp0
call .venv-win\Scripts\activate.bat
python DanzarVLM.py

echo.
echo 🔥 Bot stopped. Press any key to exit...
pause >nul 