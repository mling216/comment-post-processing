@echo off
:: Run r3 (adaptive thinking) for the 3 new combination variants: TC, TA, CA
:: Run from the scripts/ directory:  cd scripts && run_new_variants_r3.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv

echo ============================================================
echo  New Combination Variants — Run 3 (adaptive thinking)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/3] V0+Topic+Calibration (r3, thinking)
python _vc_score_api_v0_topic_calib.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_tc_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+TC & exit /b 1 )
echo.

echo [2/3] V0+Topic+Anchors (r3, thinking)
python _vc_score_api_v0_topic_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ta_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+TA & exit /b 1 )
echo.

echo [3/3] V0+Calibration+Anchors (r3, thinking)
python _vc_score_api_v0_calib_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ca_r3 --concurrency %CONCURRENCY% --thinking
if errorlevel 1 ( echo ERROR on V0+CA & exit /b 1 )
echo.

echo ============================================================
echo  Run 3 (thinking) complete. Results in vc_api_46gt_v0_tc/ta/ca_r3/
echo ============================================================
