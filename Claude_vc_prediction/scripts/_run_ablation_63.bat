@echo off
cd /d "%~dp0"

set INPUT_CSV=..\gt_all_66.csv
set MODEL=claude-opus-4-6
set CONCUR=5

echo === Step 1/3: V0+TW-dyn (with role sentence) ===
python _vc_score_api_v0_tw_dyn.py --input-csv %INPUT_CSV% --outdir ..\..\results\vc_api_63_v0_tw_dyn --concurrency %CONCUR% --model %MODEL%
if errorlevel 1 goto :error

echo.
echo === Step 2/3: Vanilla TW-dyn (no role sentence) ===
python _vc_score_api_vanilla_tw_dyn.py --input-csv %INPUT_CSV% --outdir ..\..\results\vc_api_63_vanilla_tw_dyn --concurrency %CONCUR% --model %MODEL%
if errorlevel 1 goto :error

echo.
echo === Step 3/3: 4-way comparison ===
python _compare_v0_vanilla_63.py
goto :end

:error
echo ERROR: Script failed with errorlevel %errorlevel%
exit /b 1

:end
