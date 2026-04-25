@echo off
:: Run r3 for V0, V0+Topic, V0+TCA with adaptive thinking
:: NOTE: thinking mode uses temperature=1 (not 0) — results may vary across calls
:: Run from the scripts/ directory:  cd scripts && run_r3_thinking.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv

echo ============================================================
echo  V0 / V0+Topic / V0+TCA — Run 3 (adaptive thinking)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/3] V0 (r3, thinking)
python _vc_score_api_v0.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0 & exit /b 1 )
echo.

echo [2/3] V0+Topic (r3, thinking)
python _vc_score_api_v0_topic.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_topic_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+Topic & exit /b 1 )
echo.

echo [3/3] V0+TCA (r3, thinking)
python _vc_score_api_v0_topic_calib_anchor.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tca_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+TCA & exit /b 1 )
echo.

echo ============================================================
echo  Run 3 (thinking) complete. Results in *_r3 directories.
echo ============================================================
