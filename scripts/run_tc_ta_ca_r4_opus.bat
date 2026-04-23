@echo off
:: Run r4 (adaptive thinking, opus-4.6) for TC, TA, CA variants
:: Run from the scripts/ directory:  cd scripts && run_tc_ta_ca_r4_opus.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TC, V0+TA, V0+CA  — Run 4 (adaptive thinking, %MODEL%)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/3] V0+Topic+Calibration (r4, thinking, opus)
python _vc_score_api_v0_topic_calib.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_tc_r4 --concurrency %CONCURRENCY% --thinking --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TC & exit /b 1 )
echo.

echo [2/3] V0+Topic+Anchors (r4, thinking, opus)
python _vc_score_api_v0_topic_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ta_r4 --concurrency %CONCURRENCY% --thinking --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TA & exit /b 1 )
echo.

echo [3/3] V0+Calibration+Anchors (r4, thinking, opus)
python _vc_score_api_v0_calib_anchor.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_ca_r4 --concurrency %CONCURRENCY% --thinking --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+CA & exit /b 1 )
echo.

echo ============================================================
echo  Run 4 (opus thinking) complete. Results in vc_api_46gt_v0_tc/ta/ca_r4/
echo ============================================================
