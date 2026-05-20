@echo off
:: Run V0+TWdyn-det at temperature=0 on claude-opus-4-6 — 46-image pilot.
:: V0+TWdyn-det = V1 minus calibration, with dynamic self-assigned dimension weights.
::
:: Run from the scripts/ directory:  cd scripts && run_twdyn_det_t0.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set OUTDIR=..\results\vc_api_46gt_v0_twdyn_det
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TWdyn-det — 46-image pilot (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v0_twdyn_det.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWdyn-det 46-image run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
