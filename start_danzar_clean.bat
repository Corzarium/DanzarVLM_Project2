@echo off
echo ========================================
echo DanzarAI - Clean Startup Script
echo ========================================
echo.

echo 🧹 Clearing old logs...
if exist "logs\danzar_voice.log" (
    echo 📁 Backing up previous log...
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
    set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
    set "datestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"
    move "logs\danzar_voice.log" "logs\danzar_voice_%datestamp%.log"
    echo ✅ Log backed up
) else (
    echo ℹ️ No previous log found
)

echo.
echo 🚀 Starting DanzarAI...
echo.

python DanzarVLM.py

echo.
echo 🛑 DanzarAI stopped
pause 