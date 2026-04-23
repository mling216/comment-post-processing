@echo off
:: Run r1 (temperature=0) and r3 (adaptive thinking, opus) for V0+TCA+Weighted
:: Run from the scripts/ directory:  cd scripts && run_tcaw_r1_r3.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TCA+Weighted — Run 1 (temperature=0, %MODEL%) + Run 3 (thinking, %MODEL%)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/2] V0+TCA+Weighted r1 (temperature=0)
python _vc_score_api_v0_tca_weighted.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_tcaw --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TCA+W r1 & exit /b 1 )
echo.

echo [2/2] V0+TCA+Weighted r3 (adaptive thinking)
python _vc_score_api_v0_tca_weighted.py --input-csv %INPUT% --outdir ..\vc_api_46gt_v0_tcaw_r3 --concurrency %CONCURRENCY% --thinking --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TCA+W r3 & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in vc_api_46gt_v0_tcaw/ and vc_api_46gt_v0_tcaw_r3/
echo ============================================================
