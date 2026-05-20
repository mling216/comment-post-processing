@echo off
:: Score the full 510-image production set with the V3-dyn prompt
:: (V3 with dynamic self-assigned dimension weights instead of fixed H/M/L)
:: at temperature=0 on opus-4.6 — matching the V1 (V3) 510-image run setup exactly
:: for a clean prompt-only comparison.
::
:: Run from the scripts/ directory:  cd scripts && run_v3_dyn_510_t0.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_510_v0_tw_input.csv
set OUTDIR=..\results\vc_api_510_v3_dyn_t0
set MODEL=claude-opus-4-6

echo ============================================================
echo  V3-dyn prompt — 510 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v3_dyn.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL% --temperature 0
if errorlevel 1 ( echo ERROR on V3-dyn 510-image t=0 run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
