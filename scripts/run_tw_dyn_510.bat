@echo off
:: Score the full 510-image production set with the V0+TW-dyn prompt
:: (V0+TW with dynamic self-assigned dimension weights instead of fixed H/M/L)
:: at temperature=0 on opus-4.6 — matching the V0+TW 510-image run setup exactly.
::
:: Run from the scripts/ directory:  cd scripts && run_tw_dyn_510.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_510_v0_tw_input.csv
set OUTDIR=..\results\vc_api_510_v0_tw_dyn
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TW-dyn — 510 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v0_tw_dyn.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TW-dyn 510-image run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
