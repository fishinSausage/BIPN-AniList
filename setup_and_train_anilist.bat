@echo off
rem BIPN on AniList: sets up this PC, converts the data and trains. Double-click to run.
rem   1. installs Python 3.10 and the packages, if they are missing
rem   2. converts data\AniList\interactions.csv into BIPN's format (the CSV is only read)
rem   3. trains BIPN on 6 status + 3 rating behaviors, with "like" as the target
rem Safe to close and run again: finished steps are skipped and training continues
rem from its last save (made every SAVE_MINUTES minutes and after every epoch).
rem
rem Folder layout: this file and grid_search_hyperparams.bat in the BIPN folder, the
rem Python code in codes\, the CSV and its converter in data\AniList\.
rem
rem To force a PyTorch build:  setup_and_train_anilist.bat cpu   (or cu113, cu128)
rem grid_search_hyperparams.bat runs this same file in grid-search mode.
rem LightGCN\run_lightgcn.bat runs it in setup mode: step 1 only, then it returns.
setlocal
cd /d "%~dp0"

rem ---- Training settings ------------------------------------------------------
rem Starting point: the settings the authors used for ML10M. Replace LR, BETA and
rem GAMMA with the best values from grid_search_hyperparams.bat once you have them.
rem (To change them after a training run has started, first delete
rem check_point\anilist_resume.pt, which holds the saved progress.)
set "LR=0.001"
set "BETA=0.3"
set "GAMMA=0.001"
set "BATCH_SIZE=512"
set "SAVE_MINUTES=30"
rem Share of users the grid search runs on (0.1 = 10%). To change it later,
rem delete the data\AniList_sample folder so that the sample is made again.
set "GRID_SAMPLE=0.1"
rem ---- Package versions (this replaces requirements.txt) --------------------
rem Same as the authors' requirements.txt, except numpy: their 1.21.5 breaks
rem scipy 1.10.1 on Windows, and 1.23.5 is the newest numpy that still has
rem np.float / np.bool, which metrics.py uses.
set "PACKAGES=loguru==0.7.2 numpy==1.23.5 scikit_learn==1.3.2 scipy==1.10.1 tqdm==4.66.1"
rem PyTorch: the authors' build for most NVIDIA cards, the same version for
rem PCs without NVIDIA, and a newer build for RTX 50 cards, which 1.12.1 cannot use.
set "TORCH_CU113=torch==1.12.1+cu113"
set "TORCH_CPU=torch==1.12.1+cpu"
set "TORCH_CU128=torch==2.7.1+cu128"
rem ---------------------------------------------------------------------------

set "PYVER=3.10.11"
set "PYMD5=a55e9c1e6421c84a4bd8b4be41492f51"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"
set "PYINST=%TEMP%\python-%PYVER%-amd64.exe"
set "PYEXE="
set "MODE=train"
set "BUILD="
for %%a in (%*) do (
    if /i "%%~a"=="grid" set "MODE=grid"
    if /i "%%~a"=="setup" set "MODE=setup"
    if /i "%%~a"=="cpu" set "BUILD=cpu"
    if /i "%%~a"=="cu113" set "BUILD=cu113"
    if /i "%%~a"=="cu128" set "BUILD=cu128"
)
if /i "%MODE%"=="setup" goto step1
title BIPN on AniList - %MODE%

echo ============================================================
echo  BIPN on AniList - %MODE%
echo ============================================================
echo.
:step1
if /i "%MODE%"=="setup" (echo [1/4] Python and packages ...) else (echo [1/3] Python and packages ...)
call :find_python
if not defined PYEXE goto need_install
if defined BUILD goto need_install
"%PYEXE%" -c "import numpy, scipy.sparse, sklearn, loguru, tqdm, torch" >nul 2>nul
if not errorlevel 1 goto installed

:need_install
echo       Something is missing. This downloads and installs:
echo         - Python %PYVER% from python.org, only if Python 3.10 is missing
echo         - PyTorch from download.pytorch.org, up to about 2.5 GB
echo         - numpy, scipy, scikit-learn, loguru and tqdm from pypi.org
echo.
echo       Press any key to start, or close this window to cancel.
pause >nul
echo.
if defined PYEXE goto have_python

echo       Downloading Python %PYVER% from python.org, 28 MB ...
if exist "%PYINST%" del "%PYINST%"
curl.exe -L --fail --retry 3 -o "%PYINST%" "%PYURL%"
if errorlevel 1 powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYURL%' -OutFile '%PYINST%'"
if not exist "%PYINST%" goto fail_download
certutil -hashfile "%PYINST%" MD5 | findstr /i /x "%PYMD5%" >nul
if errorlevel 1 goto fail_hash
echo       Installing Python %PYVER% for this user ...
start /wait "" "%PYINST%" /passive InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_test=0
del "%PYINST%" >nul 2>nul
call :find_python
if not defined PYEXE goto fail_install

:have_python
if defined BUILD goto check_build
set "BUILD=cpu"
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul
if errorlevel 1 goto no_nvidia
set "BUILD=cu113"
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul | findstr /r /i /c:"RTX 50[5-9]0" /c:"Blackwell" >nul
if not errorlevel 1 set "BUILD=cu128"
goto check_build

:no_nvidia
echo       No NVIDIA driver found, so the CPU build will be installed.
echo       If this PC has an NVIDIA card, install its driver and run this again.

:check_build
if /i "%BUILD%"=="cpu" (set "BUILD=cpu" & set "TORCH=%TORCH_CPU%" & goto build_ok)
if /i "%BUILD%"=="cu113" (set "BUILD=cu113" & set "TORCH=%TORCH_CU113%" & goto build_ok)
if /i "%BUILD%"=="cu128" (set "BUILD=cu128" & set "TORCH=%TORCH_CU128%" & goto build_ok)
echo ERROR: unknown build "%BUILD%". Use cpu, cu113 or cu128.
goto failed

:build_ok
echo       PyTorch build: %TORCH%
if /i "%BUILD%"=="cu128" echo       Note: RTX 50 cards need PyTorch 2.7, newer than the 1.12.1 the authors used.
"%PYEXE%" -m pip install "%TORCH%" --extra-index-url https://download.pytorch.org/whl/%BUILD%
if errorlevel 1 goto fail_pip
"%PYEXE%" -m pip install %PACKAGES%
if errorlevel 1 goto fail_pip
"%PYEXE%" -c "import numpy, scipy.sparse, sklearn, loguru, tqdm, torch; print('torch', torch.__version__, '| numpy', numpy.__version__)"
if errorlevel 1 goto fail_check

:installed
echo       Using %PYEXE%
set "DEV=cpu"
"%PYEXE%" -c "import sys, torch; ok = torch.cuda.is_available() and (torch.ones(2, device='cuda') * 2).sum().item() == 4; sys.exit(0 if ok else 1)" >nul 2>nul
if not errorlevel 1 set "DEV=cuda:0"
echo       Training device: %DEV%
if "%DEV%"=="cpu" echo       WARNING: no usable NVIDIA GPU was found. Training on the CPU is extremely slow.
if /i "%MODE%"=="setup" goto setup_done

echo.
echo [2/3] AniList data ...
if not exist "codes\main.py" goto fail_nofiles
if not exist "data\AniList\data_process.py" goto fail_nofiles
if /i "%MODE%"=="grid" goto convert_sample
if exist "data\AniList\convert_report.txt" goto converted
if not exist "data\AniList\interactions.csv" goto fail_nocsv
"%PYEXE%" "data\AniList\data_process.py"
if errorlevel 1 goto fail_convert
goto converted

:convert_sample
if exist "data\AniList_sample\convert_report.txt" goto converted
if not exist "data\AniList\interactions.csv" goto fail_nocsv
"%PYEXE%" "data\AniList\data_process.py" --sample %GRID_SAMPLE% --out "data\AniList_sample"
if errorlevel 1 goto fail_convert

:converted
echo       Converted data is ready.
rem codes\main.py saves the trained model and the training progress here.
if not exist check_point mkdir check_point

echo.
if /i "%MODE%"=="grid" goto grid
echo [3/3] Training on the full data, device %DEV%.
echo       To stop, close this window. Run this file again to continue from the last save.
echo.
"%PYEXE%" "codes\main.py" --data_name anilist --device %DEV% --lr %LR% --log_reg %BETA% --reg_weight %GAMMA% --decay 0 --batch_size %BATCH_SIZE% --model_name anilist --resume --save_minutes %SAVE_MINUTES%
if errorlevel 1 goto train_stopped
if not exist "check_point\anilist_result.json" goto train_stopped
echo.
echo ============================================================
echo  Training finished.
echo  Result:    check_point\anilist_result.json
echo  Full log:  the newest file in log\anilist
echo ============================================================
echo.
pause
exit /b 0

:train_stopped
echo.
echo Training stopped before it finished. If an error is shown above, fix it first
echo (for "out of memory", lower BATCH_SIZE at the top of this file and delete
echo check_point\anilist_resume.pt). Then run this file again to continue.
goto failed

:grid
echo [3/3] Grid search on the user sample, device %DEV%.
echo       To stop, close this window. Run grid_search_hyperparams.bat again to continue.
echo.
"%PYEXE%" "codes\grid_search.py" --device %DEV% --batch_size %BATCH_SIZE% --save_minutes %SAVE_MINUTES%
if errorlevel 1 goto failed
echo.
pause
exit /b 0

:fail_nofiles
echo ERROR: codes\main.py or data\AniList\data_process.py is missing. Keep this
echo file in the BIPN folder, next to the codes and data folders.
goto failed

:fail_nocsv
echo ERROR: data\AniList\interactions.csv was not found. Put the AniList CSV there
echo and run this file again.
goto failed

:fail_convert
echo.
echo ERROR: converting the AniList data failed, see the message above.
goto failed

:fail_download
echo.
echo ERROR: could not download the Python installer.
echo Check the internet connection and run this file again.
goto failed

:fail_hash
echo.
echo ERROR: the downloaded Python installer does not match the checksum
echo published on python.org, so it was not run. Run this file again.
del "%PYINST%" >nul 2>nul
goto failed

:fail_install
echo.
echo ERROR: Python was not installed. If you cancelled the installer, run this
echo file again.
goto failed

:fail_pip
echo.
echo ERROR: pip could not install the packages, see the message above.
echo Check the internet connection and run this file again.
goto failed

:fail_check
echo.
echo ERROR: the packages installed, but they could not be loaded, see above.
goto failed

:failed
echo.
pause
exit /b 1

rem Setup mode ends here: the Python path and the device go back to the .bat file
rem that called this one (LightGCN\run_lightgcn.bat).
:setup_done
endlocal & set "PYEXE=%PYEXE%" & set "DEV=%DEV%"
exit /b 0

rem Sets PYEXE to a 64-bit Python 3.10, or leaves it empty if there is none.
:find_python
set "PYEXE="
for /f "delims=" %%i in ('py -3.10 -c "import sys; sys.maxsize.bit_length() == 63 and print(sys.executable)" 2^>nul ^<nul') do set "PYEXE=%%i"
if defined PYEXE if exist "%PYEXE%" exit /b 0
set "PYEXE="
if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python310\python.exe"
exit /b 0
