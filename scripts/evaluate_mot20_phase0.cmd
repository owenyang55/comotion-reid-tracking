@echo off
setlocal

REM Evaluate generated MOT20 Phase 0 predictions in cmd/Anaconda Prompt.
REM Default target:
REM   predictions: results\phase0_mot20_02_smoke\predictions
REM   output:      results\phase0_mot20_02_smoke
REM   run name:    baseline
REM   GT root:     D:\github\data\datasets\MOT\MOT20\train
REM
REM Usage with defaults:
REM   scripts\evaluate_mot20_phase0.cmd
REM
REM Usage with explicit paths:
REM   scripts\evaluate_mot20_phase0.cmd results\phase0_mot20_02_smoke\predictions results\phase0_mot20_02_smoke baseline D:\github\data\datasets\MOT\MOT20\train

set "PRED_DIR=%~1"
set "OUT_DIR=%~2"
set "RUN_NAME=%~3"
set "GT_ROOT=%~4"

if "%PRED_DIR%"=="" set "PRED_DIR=results\phase0_mot20_02_smoke\predictions"
if "%OUT_DIR%"=="" set "OUT_DIR=results\phase0_mot20_02_smoke"
if "%RUN_NAME%"=="" set "RUN_NAME=baseline"
if "%GT_ROOT%"=="" set "GT_ROOT=D:\github\data\datasets\MOT\MOT20\train"

echo [Phase0] Prediction dir: %PRED_DIR%
echo [Phase0] Output dir: %OUT_DIR%
echo [Phase0] Run name: %RUN_NAME%
echo [Phase0] GT root: %GT_ROOT%

if not exist "%PRED_DIR%\" (
  echo.
  echo [Phase0] Prediction directory does not exist:
  echo   %PRED_DIR%
  echo Run the inference command first, or pass the correct prediction directory.
  exit /b 1
)

dir /b "%PRED_DIR%\*.txt" >nul 2>nul
if errorlevel 1 (
  echo.
  echo [Phase0] No MOT txt files found in:
  echo   %PRED_DIR%
  echo Expected a file such as:
  echo   %PRED_DIR%\MOT20-02.txt
  echo Run the inference command first, or pass the correct prediction directory.
  exit /b 1
)

python evaluation\phase0_baseline.py ^
  --skip-demo ^
  --prediction-dir "%PRED_DIR%" ^
  --output-dir "%OUT_DIR%" ^
  --run-name "%RUN_NAME%" ^
  --run-trackeval ^
  --gt-folder "%GT_ROOT%" ^
  --skip-split-folder ^
  --seqmap-from-predictions

if errorlevel 1 (
  echo.
  echo [Phase0] Evaluation failed.
  echo If the error is "No module named scipy", run:
  echo   python -m pip install scipy
  exit /b 1
)

echo.
echo [Phase0] Evaluation finished. Check:
echo   %OUT_DIR%\phase0_report.md
echo   %OUT_DIR%\phase0_trackeval_metrics.csv
echo   %OUT_DIR%\phase0_summary.json
