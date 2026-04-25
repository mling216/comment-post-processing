@echo off
:: Run r4 for V0 and V0+TCA with adaptive thinking (second thinking run)
:: NOTE: thinking mode uses temperature=1 — results may vary from r3
:: Run from the scripts/ directory:  cd scripts && run_r4_thinking.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv

echo ============================================================
echo  V0 / V0+TCA — Run 4 (adaptive thinking)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/2] V0 (r4, thinking)
python _vc_score_api_v0.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_r4 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0 & exit /b 1 )
echo.

echo [2/2] V0+TCA (r4, thinking)
python _vc_score_api_v0_topic_calib_anchor.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tca_r4 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+TCA & exit /b 1 )
echo.

echo ============================================================
echo  Run 4 (thinking) complete. Results in *_r4 directories.
echo ============================================================
