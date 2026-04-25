@echo off
:: Run r1 and r2 (both temperature=0) for V0+Topic+Weighted (V0+TW).
:: Run from the scripts/ directory:  cd scripts && run_tw_r1_r2.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TW (Topic+Weighted) — Run 1 + Run 2 (temperature=0, %MODEL%)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/2] V0+TW r1 (temperature=0)
python _vc_score_api_v0_topic_weighted.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tw --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TW r1 & exit /b 1 )
echo.

echo [2/2] V0+TW r2 (temperature=0)
python _vc_score_api_v0_topic_weighted.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tw_r2 --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TW r2 & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in results/vc_api_46gt_v0_tw/ and results/vc_api_46gt_v0_tw_r2/
echo ============================================================
