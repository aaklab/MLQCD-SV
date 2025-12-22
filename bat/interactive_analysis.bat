@echo off
echo ============================================
echo MLQCD-SV Interactive Analysis Tool
echo ============================================
echo.
cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

echo.
echo Starting interactive analysis menu...
echo.
python src\interactive_analysis.py
pause