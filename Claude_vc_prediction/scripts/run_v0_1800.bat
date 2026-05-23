@echo off
:: Score the 1800-image VIS2025 set with the V0 pure zero-shot baseline prompt
:: at temperature=0 on opus-4.6 — same setup as run_tw_dyn_1800.bat but with V0.
::
:: Run from the scripts/ directory:  cd scripts && run_v0_1800.bat

set CONCURRENCY=5
set INPUT=..\..\results\vc_api_1800_v0_tw_input.csv
set OUTDIR=..\..\results\vc_api_1800_v0
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0 (pure zero-shot) — 1800 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v0.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0 1800-image run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
