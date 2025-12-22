@echo off
echo Combining spectral fit results...
cd /d "%~dp0\.."
python src\utils\combine_results.py
echo.
echo Results saved to combined_spectral_results.csv
pause