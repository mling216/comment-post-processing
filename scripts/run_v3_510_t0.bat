@echo off
:: Score the full 510-image production set with the V3 prompt
:: (a.k.a. "V1" in later naming — V0 + Topics + Calibration + Anchors + Weighted,
::  with per-dimension JSON output) at temperature=0 on opus-4.6.
:: This aligns run conditions with the V0+TW 510-image run for a clean
:: prompt-only comparison in V0_Variants_Comparison.ipynb.
::
:: Run from the scripts/ directory:  cd scripts && run_v3_510_t0.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_510_v0_tw_input.csv
set OUTDIR=..\results\vc_api_510_v3_t0
set MODEL=claude-opus-4-6

echo ============================================================
echo  V3 prompt — 510 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v3.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL% --temperature 0
if errorlevel 1 ( echo ERROR on V3 510-image t=0 run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
