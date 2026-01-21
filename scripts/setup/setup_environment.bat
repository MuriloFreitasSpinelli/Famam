@echo off
REM ============================================
REM Famam Complete Environment Setup (Windows)
REM ============================================
REM Sets up conda environment with TensorFlow GPU support (CUDA 11.2)
REM Run this ONCE to set up everything, then use run_training.bat
REM
REM Requirements:
REM   - Miniconda or Anaconda installed
REM   - NVIDIA GPU with drivers installed (optional, for GPU training)
REM
REM Usage: setup_environment.bat

setlocal enabledelayedexpansion

set ENV_NAME=famam_gpu
set PYTHON_VERSION=3.10

echo.
echo ============================================
echo   Famam Environment Setup
echo   Platform: Windows
echo ============================================
echo.

REM ============================================
REM Step 1: Find Conda
REM ============================================
echo [1/8] Locating conda installation...

set CONDA_EXE=
for %%p in (
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
    "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\miniconda3\Scripts\conda.exe"
    "C:\anaconda3\Scripts\conda.exe"
) do (
    if exist %%p (
        set CONDA_EXE=%%~p
        goto :found_conda
    )
)

REM Check if conda is in PATH
where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where conda') do (
        set CONDA_EXE=%%i
        goto :found_conda
    )
)

echo.
echo ERROR: Conda not found!
echo.
echo Please install Miniconda first:
echo   1. Download from: https://docs.conda.io/en/latest/miniconda.html
echo   2. Choose: Miniconda3 Windows 64-bit
echo   3. Run the installer
echo   4. Restart Command Prompt and run this script again
echo.
pause
exit /b 1

:found_conda
echo        Found: %CONDA_EXE%

REM ============================================
REM Step 2: Check for existing environment
REM ============================================
echo.
echo [2/8] Checking for existing environment...

call "%CONDA_EXE%" env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo        Environment '%ENV_NAME%' already exists.
    set /p RECREATE="        Recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo        Removing existing environment...
        call "%CONDA_EXE%" env remove -n %ENV_NAME% -y >nul 2>&1
    ) else (
        echo        Keeping existing environment. Updating packages...
        goto :install_packages
    )
)

REM ============================================
REM Step 3: Create conda environment
REM ============================================
echo.
echo [3/8] Creating conda environment with Python %PYTHON_VERSION%...
call "%CONDA_EXE%" create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)

REM ============================================
REM Step 4: Install CUDA toolkit
REM ============================================
:install_packages
echo.
echo [4/8] Installing CUDA Toolkit 11.2 and cuDNN 8.1...
echo        (This downloads ~1.2GB, may take a while)
call "%CONDA_EXE%" install -n %ENV_NAME% -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
if %errorlevel% neq 0 (
    echo WARNING: CUDA installation had issues. GPU may not work.
)

REM ============================================
REM Step 5: Install TensorFlow
REM ============================================
echo.
echo [5/8] Installing TensorFlow 2.10.1...
call "%CONDA_EXE%" run -n %ENV_NAME% pip install tensorflow==2.10.1 "numpy<2"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install TensorFlow
    pause
    exit /b 1
)

REM ============================================
REM Step 6: Install core dependencies
REM ============================================
echo.
echo [6/8] Installing core project dependencies...
call "%CONDA_EXE%" run -n %ENV_NAME% pip install ^
    h5py ^
    scikit-learn ^
    matplotlib ^
    pandas ^
    tqdm ^
    pretty_midi ^
    music21 ^
    muspy ^
    mido ^
    requests

if %errorlevel% neq 0 (
    echo WARNING: Some core dependencies may have failed
)

REM ============================================
REM Step 7: Install additional dependencies
REM ============================================
echo.
echo [7/8] Installing additional dependencies...
call "%CONDA_EXE%" run -n %ENV_NAME% pip install ^
    jupyterlab ^
    datasets ^
    scikit-optimize ^
    midi2audio

if %errorlevel% neq 0 (
    echo WARNING: Some optional dependencies may have failed
    echo          (midi2audio requires FluidSynth system install)
)

REM ============================================
REM Step 8: Setup activation scripts and verify
REM ============================================
echo.
echo [8/8] Configuring environment and verifying GPU...

REM Get conda env path
for /f "tokens=*" %%i in ('call "%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(sys.prefix)"') do set ENV_PATH=%%i

REM Create activation script for CUDA PATH
set ACTIVATE_DIR=%ENV_PATH%\etc\conda\activate.d
if not exist "%ACTIVATE_DIR%" mkdir "%ACTIVATE_DIR%"

echo @echo off > "%ACTIVATE_DIR%\cuda_path.bat"
echo set "PATH=%ENV_PATH%\Library\bin;%%PATH%%" >> "%ACTIVATE_DIR%\cuda_path.bat"

REM Verify installation
echo.
echo Verifying installation...
set "PATH=%ENV_PATH%\Library\bin;%PATH%"
call "%CONDA_EXE%" run -n %ENV_NAME% python -c "import tensorflow as tf; print('  TensorFlow:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('  GPUs found:', len(gpus)); [print('    -', g.name) for g in gpus]"

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Environment: %ENV_NAME%
echo.
echo Quick commands:
echo   conda activate %ENV_NAME%     - Activate the environment
echo   run_training.bat              - Run model training
echo   jupyter lab                   - Start JupyterLab
echo.
echo Installed packages:
echo   - TensorFlow 2.10.1 (GPU support via CUDA 11.2)
echo   - NumPy, Pandas, Matplotlib, Scikit-learn
echo   - Music21, MusPy, Pretty-MIDI, Mido
echo   - JupyterLab, Scikit-optimize, HuggingFace Datasets
echo.
pause
