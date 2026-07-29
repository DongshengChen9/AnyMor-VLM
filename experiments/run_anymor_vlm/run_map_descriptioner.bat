@echo off
setlocal

cd /d "%~dp0\..\.."

set "OPENAI_API_KEY=PASTE_YOUR_OPENAI_API_KEY_HERE"

rem Map_descriptioner.py parameters
set "DATASET_NAME=urbanform"
set "SAVE_DIR=checkpoints\gpt"
set "MODEL=gpt-4o-2024-08-06"
set "MAX_TOKENS=77"
set "TEMPERATURE=0.0"
set "TOP_P=1.0"
set "FREQUENCY_PENALTY=0.0"
set "PRESENCE_PENALTY=0.0"
set "SEED=42"
set "MAX_IMAGES="
set "START_IDX=0"
set "IMAGE_DETAIL=low"

if "%OPENAI_API_KEY%"=="PASTE_YOUR_OPENAI_API_KEY_HERE" (
    echo ERROR: Set your OpenAI API key in this batch file first.
    exit /b 1
)

if not defined PYTHON_EXE set "PYTHON_EXE=python"

set "OPTIONAL_ARGS="
if defined MAX_IMAGES set "OPTIONAL_ARGS=--max_images %MAX_IMAGES%"

"%PYTHON_EXE%" Map_descriptioner.py ^
    --dataset_name "%DATASET_NAME%" ^
    --save_dir "%SAVE_DIR%" ^
    --model "%MODEL%" ^
    --max_tokens %MAX_TOKENS% ^
    --temperature %TEMPERATURE% ^
    --top_p %TOP_P% ^
    --frequency_penalty %FREQUENCY_PENALTY% ^
    --presence_penalty %PRESENCE_PENALTY% ^
    --seed %SEED% ^
    --start_idx %START_IDX% ^
    --image_detail %IMAGE_DETAIL% ^
    %OPTIONAL_ARGS% %*

if errorlevel 1 (
    echo Map_descriptioner.py failed.
    exit /b %errorlevel%
)

endlocal
