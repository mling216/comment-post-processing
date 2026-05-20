@echo off
:: Topic Selection (top-3 F1 task) on the full 510-image set
:: Variants: V0+T and V0+TW (opus-4.6, t=0)
:: Uses same input CSV as the 510-image VC scoring run.
::
:: Run from the scripts/ directory:  cd scripts && run_topicsel_510.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_510_v0_tw_input.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  Topic Selection (top-3) — 510 images (%MODEL%, t=0)
echo  Input: %INPUT%
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo --- V0+T ---
set OUTDIR=..\results\vc_api_510_topicsel_v0_t
python _vc_score_api_v0_t_top3.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+T & exit /b 1 )
echo.

echo --- V0+TW ---
set OUTDIR=..\results\vc_api_510_topicsel_v0_tw
python _vc_score_api_v0_tw_top3.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TW & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in:
echo    ..\results\vc_api_510_topicsel_v0_t
echo    ..\results\vc_api_510_topicsel_v0_tw
echo ============================================================
