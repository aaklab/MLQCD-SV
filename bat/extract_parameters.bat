@echo off
echo Extracting spectral fit parameters...
cd /d "%~dp0\.."
python src\extract_spectral_parameters.py
pause