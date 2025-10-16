@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM Automated Chlorophyll Pipeline Runner with Enhanced Logging
REM ====================================================================

REM Set paths - CHANGE THESE IF NEEDED
set SCRIPT_DIR=C:\Users\23755118\OneDrive - UWA\Documents\PhD_Varshani\CODING\chl_time
set SCRIPT_NAME=daily_chl_pipeline.py
set LOG_FILE=%SCRIPT_DIR%\automation_log.txt
set ERROR_LOG=%SCRIPT_DIR%\automation_errors.txt

REM Python path - UPDATE THIS WITH YOUR ACTUAL PYTHON PATH
REM Based on the error log, you're using Miniconda with py3_13 environment
REM Uncomment and use ONE of these options:
REM 

REM Option 1: If using Miniconda with py3_13 environment (RECOMMENDED FOR YOU)
set CONDA_PATH=C:\Users\23755118\AppData\Local\miniconda3
set CONDA_ENV=py3_13
set PYTHON_EXE=%CONDA_PATH%\envs\%CONDA_ENV%\python.exe

REM Option 2: If using base conda environment
REM set CONDA_PATH=C:\Users\23755118\AppData\Local\miniconda3
REM set PYTHON_EXE=%CONDA_PATH%\python.exe

REM Option 3: If Python is in PATH (currently not working for you)
REM set PYTHON_EXE=python

REM Change to script directory
cd /d "%SCRIPT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not change to directory %SCRIPT_DIR% >> "%ERROR_LOG%"
    exit /b 1
)

REM Log start time
echo ================================================ >> "%LOG_FILE%"
echo Task started at: %date% %time% >> "%LOG_FILE%"
echo Directory: %CD% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Check if Python is available
echo Checking Python... >> "%LOG_FILE%"
"%PYTHON_EXE%" --version >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found at: %PYTHON_EXE% >> "%ERROR_LOG%"
    echo ERROR: Python not found at: %date% %time% >> "%LOG_FILE%"
    echo Tried to use: %PYTHON_EXE% >> "%LOG_FILE%"
    exit /b 1
)
echo Python found successfully >> "%LOG_FILE%"

REM Check if script exists
if not exist "%SCRIPT_NAME%" (
    echo ERROR: Script %SCRIPT_NAME% not found in %CD% >> "%ERROR_LOG%"
    echo ERROR: Script not found at: %date% %time% >> "%LOG_FILE%"
    exit /b 1
)

REM Activate conda environment if needed
REM Uncomment and modify these lines if you're using conda:
REM echo Activating conda environment... >> "%LOG_FILE%"
REM call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_ENV%" >> "%LOG_FILE%" 2>&1
REM if %ERRORLEVEL% NEQ 0 (
REM     echo ERROR: Failed to activate conda environment >> "%ERROR_LOG%"
REM     exit /b 1
REM )

REM Run Python script and capture output
echo Running Python script... >> "%LOG_FILE%"
"%PYTHON_EXE%" "%SCRIPT_NAME%" >> "%LOG_FILE%" 2>> "%ERROR_LOG%"

REM Capture the exit code
set SCRIPT_EXIT_CODE=%ERRORLEVEL%

REM Log completion status
echo. >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
if %SCRIPT_EXIT_CODE% EQU 0 (
    echo SUCCESS: Task completed at: %date% %time% >> "%LOG_FILE%"
    echo Exit Code: %SCRIPT_EXIT_CODE% >> "%LOG_FILE%"
) else (
    echo FAILURE: Task FAILED at: %date% %time% >> "%LOG_FILE%"
    echo Exit Code: %SCRIPT_EXIT_CODE% >> "%LOG_FILE%"
    echo Check %ERROR_LOG% for error details >> "%LOG_FILE%"
    echo ================================================ >> "%ERROR_LOG%"
    echo Task failed with exit code %SCRIPT_EXIT_CODE% at %date% %time% >> "%ERROR_LOG%"
    echo ================================================ >> "%ERROR_LOG%"
)
echo ================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Keep log files manageable (keep last 1000 lines)
if exist "%LOG_FILE%.tmp" del "%LOG_FILE%.tmp"
powershell -Command "Get-Content '%LOG_FILE%' -Tail 1000 | Set-Content '%LOG_FILE%.tmp'" 2>nul
if exist "%LOG_FILE%.tmp" (
    move /y "%LOG_FILE%.tmp" "%LOG_FILE%" >nul
)

endlocal
exit /b %SCRIPT_EXIT_CODE%