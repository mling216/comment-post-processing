@echo off
:: Run r1 (temperature=0) for the 3 new combination variants: TC, TA, CA
:: Run from the scripts/ directory:  cd scripts && run_new_variants_r1.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv

echo ============================================================
echo  New Combination Variants — Run 1 (temperature=0)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/3] V0+Topic+Calibration (r1)
python _vc_score_api_v0_topic_calib.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_tc --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+TC & exit /b 1 )
echo.

echo [2/3] V0+Topic+Anchors (r1)
python _vc_score_api_v0_topic_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ta --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+TA & exit /b 1 )
echo.

echo [3/3] V0+Calibration+Anchors (r1)
python _vc_score_api_v0_calib_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ca --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+CA & exit /b 1 )
echo.

echo ============================================================
echo  Run 1 complete. Results in vc_api_46gt_v0_tc/ta/ca/
echo ============================================================
