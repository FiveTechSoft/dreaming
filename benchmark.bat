@echo off
REM benchmark.bat — Auto-research verifier for dreaming engine
REM Compiles, runs, and captures tok/s + output quality

setlocal enabledelayedexpansion

set MODEL=C:\dreaming\tinyLlama.gguf
set PROMPT=DREAM
set TOKENS=50
set TEMP=0
set TOPK=1
set LOG_DIR=C:\dreaming\benchmarks
set TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [1/3] Compiling llm_inference.c ...
gcc -O2 -Wall -o llm_inference.exe llm_inference.c -lm 2>compile_err.txt
if errorlevel 1 (
    echo COMPILE FAILED
    type compile_err.txt
    exit /b 1
)
echo       OK

echo [2/3] Running benchmark (prompt=%PROMPT%, tokens=%TOKENS%) ...
llm_inference.exe %MODEL% "%PROMPT%" %TOKENS% %TEMP% %TOPK% > "%LOG_DIR%\%TIMESTAMP%.txt" 2>&1
echo       Done

echo [3/3] Results:
findstr /C:"Generated" /C:"tok/s" "%LOG_DIR%\%TIMESTAMP%.txt"

echo.
echo Full log: %LOG_DIR%\%TIMESTAMP%.txt
type "%LOG_DIR%\%TIMESTAMP%.txt"
