@echo off
:: Second run of all V0 variants (temperature=0) saved to _r2 directories
:: Run from the scripts/ directory:  cd scripts && run_extra_runs.bat
:: Adjust --concurrency as needed (default 5)

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv

echo ============================================================
echo  V0 Variants — Run 2 (temperature=0, saved to _r2 dirs)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/4] V0 (run 2)
python _vc_score_api_v0.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_r2 --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0 & exit /b 1 )
echo.

echo [2/4] V0+Topic (run 2)
python _vc_score_api_v0_topic.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_topic_r2 --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+Topic & exit /b 1 )
echo.

echo [3/4] V0+Calibration (run 2)
python _vc_score_api_v0_calibration.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_calibration_r2 --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+Calibration & exit /b 1 )
echo.

echo [4/4] V0+Anchors (run 2)
python _vc_score_api_v0_anchors.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_anchors_r2 --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+Anchors & exit /b 1 )
echo.

echo [5/5] V0+Topic+Calibration+Anchors (run 2)
python _vc_score_api_v0_topic_calib_anchor.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tca_r2 --concurrency %CONCURRENCY%
if errorlevel 1 ( echo ERROR on V0+TCA & exit /b 1 )
echo.

echo ============================================================
echo  Run 2 complete. Results saved to *_r2 directories.
echo ============================================================
