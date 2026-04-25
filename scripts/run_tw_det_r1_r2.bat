@echo off
:: Run r1 and r2 (both temperature=0) for V0+TW + V1-detailed-method (V0+TWdet).
:: Note: V1 itself was run at t=1 single-pass. Here we use t=0 for parity with
:: the rest of the E1 ablation.
:: Run from the scripts/ directory:  cd scripts && run_tw_det_r1_r2.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TWdet (TW + per-dim anchors + per-dim output) r1 + r2
echo  temperature=0, %MODEL%, concurrency=%CONCURRENCY%
echo ============================================================
echo.

echo [1/2] V0+TWdet r1
python _vc_score_api_v0_tw_det.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tw_det --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWdet r1 & exit /b 1 )
echo.

echo [2/2] V0+TWdet r2
python _vc_score_api_v0_tw_det.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_tw_det_r2 --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWdet r2 & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in results/vc_api_46gt_v0_tw_det[_r2]/
echo ============================================================
