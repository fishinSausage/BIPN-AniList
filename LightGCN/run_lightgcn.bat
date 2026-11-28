@echo off
rem LightGCN baseline on AniList: grid search for the best hyper-parameters. Double-click to run.
rem   1. installs Python 3.10 and the packages, if they are missing (done by
rem      ..\setup_and_train_anilist.bat, the same setup BIPN uses)
rem   2. asks for the TRAIN csv and the VALID csv in two Explorer windows
rem   3. keeps the rows with scaled_like = 1 (columns user_id, media_id, scaled_like)
rem   4. trains LightGCN on the NVIDIA card for every combination of the values listed
rem      at the top of grid_search.py and ranks them by validation recall@20
rem Safe to close and run again: finished combinations are skipped.
rem
rem What is kept, in the results folder next to this file:
rem   best_result.json   the best hyper-parameters and their validation result
rem   best_model.pth     the trained model of that best run
rem   grid_results.csv   every finished combination, best first
rem
rem To force a PyTorch build:  run_lightgcn.bat cpu   (or cu113, cu128)
setlocal
title LightGCN on AniList - grid search
echo ============================================================
echo  LightGCN on AniList - grid search
echo ============================================================
echo.
if not exist "%~dp0..\setup_and_train_anilist.bat" goto fail_nosetup
findstr /c:":setup_done" "%~dp0..\setup_and_train_anilist.bat" >nul
if errorlevel 1 goto fail_nosetup
set "PYEXE="
set "DEV="
call "%~dp0..\setup_and_train_anilist.bat" setup %*
if errorlevel 1 exit /b 1
if not defined PYEXE goto fail_nosetup
if not exist "%~dp0grid_search.py" goto fail_nofiles
echo.
"%PYEXE%" "%~dp0grid_search.py" --device %DEV%
if errorlevel 1 goto failed
echo.
pause
exit /b 0

:fail_nosetup
echo ERROR: setup_and_train_anilist.bat was not found, or it is an older version
echo without the setup mode. Keep the LightGCN folder inside the BIPN folder, next
echo to setup_and_train_anilist.bat, and use the version uploaded with this folder.
goto failed

:fail_nofiles
echo ERROR: grid_search.py is missing. Keep this file in the LightGCN folder, next
echo to grid_search.py, lightgcn.py and preprocess.py.
goto failed

:failed
echo.
pause
exit /b 1
