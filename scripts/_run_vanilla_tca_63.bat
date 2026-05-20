@echo off
cd /d "%~dp0"

set INPUT_CSV=..\Claude_vc_prediction\gt_all_66.csv
set MODEL=claude-opus-4-6
set CONCUR=5

echo === Step 1/4: Vanilla V0+T (topics, no role sentence) ===
python _vc_score_api_vanilla_t.py --input-csv %INPUT_CSV% --outdir ..\results\vc_api_63_vanilla_t --concurrency %CONCUR% --model %MODEL%
if errorlevel 1 goto :error

echo.
echo === Step 2/4: Vanilla V0+C (calibration, no role sentence) ===
python _vc_score_api_vanilla_c.py --input-csv %INPUT_CSV% --outdir ..\results\vc_api_63_vanilla_c --concurrency %CONCUR% --model %MODEL%
if errorlevel 1 goto :error

echo.
echo === Step 3/4: Vanilla V0+A (anchors, no role sentence) ===
python _vc_score_api_vanilla_a.py --input-csv %INPUT_CSV% --outdir ..\results\vc_api_63_vanilla_a --concurrency %CONCUR% --model %MODEL%
if errorlevel 1 goto :error

echo.
echo === Step 4/4: Vanilla comparison (all 5 variants) ===
python _compare_vanilla_all_63.py
goto :end

:error
echo ERROR: Script failed with errorlevel %errorlevel%
exit /b 1

:end
