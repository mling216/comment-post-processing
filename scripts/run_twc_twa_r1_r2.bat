@echo off
:: Run r1 and r2 (both temperature=0) for the two new T+W triples: V0+TWC, V0+TWA.
:: Run from the scripts/ directory:  cd scripts && run_twc_twa_r1_r2.bat

set CONCURRENCY=5
set INPUT=..\Claude_vc_prediction\gt_all_46.csv
set MODEL=claude-opus-4-6

echo ============================================================
echo  V0+TWC and V0+TWA — r1 + r2 (temperature=0, %MODEL%)
echo  Concurrency: %CONCURRENCY%
echo ============================================================
echo.

echo [1/4] V0+TWC r1
python _vc_score_api_v0_twc.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_twc --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWC r1 & exit /b 1 )
echo.

echo [2/4] V0+TWC r2
python _vc_score_api_v0_twc.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_twc_r2 --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWC r2 & exit /b 1 )
echo.

echo [3/4] V0+TWA r1
python _vc_score_api_v0_twa.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_twa --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWA r1 & exit /b 1 )
echo.

echo [4/4] V0+TWA r2
python _vc_score_api_v0_twa.py --input-csv %INPUT% --outdir ..\results\vc_api_46gt_v0_twa_r2 --concurrency %CONCURRENCY% --model %MODEL%
if errorlevel 1 ( echo ERROR on V0+TWA r2 & exit /b 1 )
echo.

echo ============================================================
echo  Done. Results in results/vc_api_46gt_v0_twc[_r2]/ and results/vc_api_46gt_v0_twa[_r2]/
echo ============================================================
