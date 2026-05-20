@echo off
:: Run V1-dyn (V3-dyn) at temperature=0 on claude-opus-4-6 — 46-image pilot set.
:: Matches the t=0 condition used for V1 (V3) to isolate the prompt change from temperature.
::
:: Run from the scripts/ directory:  cd scripts && run_v3_dyn_t0.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set OUTDIR=..\results\vc_api_46gt_v3_dyn_t0
set MODEL=claude-opus-4-6

echo ============================================================
echo  V1-dyn (V3-dyn) — 46-image pilot (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v3_dyn.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL% --temperature 0
if errorlevel 1 ( echo ERROR on V1-dyn t=0 run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
