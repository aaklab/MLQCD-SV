@echo off
echo ============================================
echo MLQCD-SV Spectral Fit Analysis - 7 Datasets
echo ============================================
echo.

REM Set up environment
set PYTHON=python
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
echo Running spectral fit analysis on 7 datasets...
echo Model: GBR (as configured in config.py)
echo.

REM Run the seven datasets mentioned in the historical batch file
echo [1/7] Running K_ll_to_qsq0...
%PYTHON% src\run_experiment.py K_ll_to_qsq0
if errorlevel 1 (
    echo ERROR: K_ll_to_qsq0 failed
    pause
    exit /b 1
)

echo.
echo [2/7] Running K_ll_to_2qsqmaxby3...
%PYTHON% src\run_experiment.py K_ll_to_2qsqmaxby3
if errorlevel 1 (
    echo ERROR: K_ll_to_2qsqmaxby3 failed
    pause
    exit /b 1
)

echo.
echo [3/7] Running localscalar_T16_to_qsq0...
%PYTHON% src\run_experiment.py localscalar_T16_to_qsq0
if errorlevel 1 (
    echo ERROR: localscalar_T16_to_qsq0 failed
    pause
    exit /b 1
)

echo.
echo [4/7] Running localscalar_T19_to_qsqmaxby3...
%PYTHON% src\run_experiment.py localscalar_T19_to_qsqmaxby3
if errorlevel 1 (
    echo ERROR: localscalar_T19_to_qsqmaxby3 failed
    pause
    exit /b 1
)

echo.
echo [5/7] Running localscalar_T22_to_2qsqmaxby3...
%PYTHON% src\run_experiment.py localscalar_T22_to_2qsqmaxby3
if errorlevel 1 (
    echo ERROR: localscalar_T22_to_2qsqmaxby3 failed
    pause
    exit /b 1
)

echo.
echo [6/7] Running localtempvector_T16_to_qsq0...
%PYTHON% src\run_experiment.py localtempvector_T16_to_qsq0
if errorlevel 1 (
    echo ERROR: localtempvector_T16_to_qsq0 failed
    pause
    exit /b 1
)

echo.
echo [7/7] Running localtempvector_T22_to_2qsqmaxby3...
%PYTHON% src\run_experiment.py localtempvector_T22_to_2qsqmaxby3
if errorlevel 1 (
    echo ERROR: localtempvector_T22_to_2qsqmaxby3 failed
    pause
    exit /b 1
)

echo.
echo ============================================
echo ALL 7 EXPERIMENTS COMPLETED SUCCESSFULLY
echo ============================================
echo.
echo Results are saved in the results/batch/ directory
echo Each experiment produces:
echo   - PDF report with spectral fit plots
echo   - Spectral fit parameters (E0, E1, A0, A1)
echo   - Bayesian fit results with uncertainties
echo.

REM Run the results aggregation script
echo Aggregating spectral fit parameters...
%PYTHON% src\aggregate_spectral_results.py

echo.
echo Check 'spectral_fit_summary.csv' for combined results
echo.
pause