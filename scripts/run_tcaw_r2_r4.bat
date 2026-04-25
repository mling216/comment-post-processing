@echo off
:: Run r2 (temperature=0) and r4 (adaptive thinking, opus) for V0+TCA+Weighted
:: Run from the scripts/ directory:  cd scripts && run_tcaw_r2_r4.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TCA+Weighted — Run 2 (temperature=0, %MODEL%) + Run 4 (thinking, %MODEL%)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/2] V0+TCA+Weighted r2 (temperature=0)
python _vc_score_api_v0_tca_weighted.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tcaw_r2 --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TCA+W r2 & exit /b 1 )
echo.

echo [2/2] V0+TCA+Weighted r4 (adaptive thinking)
python _vc_score_api_v0_tca_weighted.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tcaw_r4 --concurrency %CONCURRENCY% --thinking --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TCA+W r4 & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in results/vc_api_46gt_v0_tcaw_r2/ and results/vc_api_46gt_v0_tcaw_r4/
echo ============================================================
