@echo off
echo ========================================
echo DanzarAI Complete Startup Script
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

echo 🚀 KILLING EXISTING PROCESSES...
echo.

REM Kill any processes on port 8083 (Qwen2.5-VL)
echo Killing Qwen2.5-VL processes on port 8083...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8083 2^>nul') do (
    echo Killing process %%a
    taskkill /f /pid %%a 2>nul
)

echo ✅ Process cleanup completed
echo.

echo 🚀 STARTING SERVICES...
echo.

REM Start Qwen2.5-VL server
echo Starting Qwen2.5-VL CUDA server on port 8083...
start "Qwen2.5-VL Server" cmd /k "cd /d E:\DanzarVLM_Project && .\start_qwen_vl_server.bat"

REM Wait for Qwen2.5-VL to start
echo Waiting for Qwen2.5-VL server to start...
timeout /t 15 /nobreak >nul

echo 🔍 CHECKING SERVICE HEALTH...
echo.

REM Check Qwen2.5-VL server health
echo Checking Qwen2.5-VL server health...
curl -s http://localhost:8083/v1/models >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Qwen2.5-VL server is healthy
) else (
    echo ❌ Qwen2.5-VL server health check failed
)

echo.
echo 🎤 STARTING DANZARAI BOT...
echo.

REM Start the main bot
cd /d E:\DanzarVLM_Project
python DanzarVLM.py

echo.
echo 🛑 DanzarAI stopped
pause 