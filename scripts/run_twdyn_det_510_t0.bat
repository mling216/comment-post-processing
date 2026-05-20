@echo off
:: Score the full 510-image production set with the V0+TWdyn-det prompt
:: (V1 minus calibration, with dynamic self-assigned dimension weights)
:: at temperature=0 on opus-4.6 — matching the V1 510-image setup.
::
:: Run from the scripts/ directory:  cd scripts && run_twdyn_det_510_t0.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_510_v0_tw_input.csv
set OUTDIR=..\results\vc_api_510_v0_twdyn_det_t0
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TWdyn-det — 510 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v0_twdyn_det.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWdyn-det 510-image run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
