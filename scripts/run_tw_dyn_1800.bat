@echo off
:: Score the 1800-image VIS2025 dataset (Claude_vc_prediction\FormalExp1800Images.csv)
:: with the V0+TW-dyn prompt at temperature=0 on opus-4.6 — matching the V0+TW
:: 510-image production run setup exactly. Resumable: re-run to retry failures.
::
:: Input CSV is built from FormalExp1800Images.csv (columns ImageName, ImageLink
:: renamed to imageName, imageURL).
::
:: Run from the scripts/ directory:  cd scripts && run_tw_dyn_1800.bat

set CONCURRENCY=5
set INPUT=..\results\vc_api_1800_v0_tw_input.csv
set OUTDIR=..\results\vc_api_1800_v0_tw_dyn
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TW-dyn — 1800 images (temperature=0, %MODEL%)
echo  Input:        %INPUT%
echo  Output dir:   %OUTDIR%
echo  Concurrency:  %CONCURRENCY%
echo ============================================================
echo.

python _vc_score_api_v0_tw_dyn.py --input-csv %INPUT% --outdir %OUTDIR% --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TW-dyn 1800-image run & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in %OUTDIR%
echo ============================================================
