@echo off
rem LightGCN baseline on AniList: grid search for the best hyper-parameters. Double-click to run.
rem (START.bat at the top of the repository runs the same three stages with a summary screen.)
rem   1. installs Python 3.10 and the packages, if they are missing (done by
rem      ..\..\BIPN\scripts\setup_and_train_anilist.bat, the same setup BIPN uses)
rem   2. asks for the TRAIN, VALIDATION and TEST csv in three Explorer windows
rem   3. reads them: the train rows with scaled_like = 1 are the likes it learns from,
rem      the validation and test rows with eval_like = 1 are the answers
rem   4. trains LightGCN on the NVIDIA card for every combination of the values listed
rem      at the top of grid_search.py and ranks them by validation ndcg@20;
rem      recall@20 and ndcg@20 on test are reported for the best one
rem   5. when the grid search has finished, starts the plain BIPN baseline by itself
rem      (BIPN\scripts\train_likes_only.bat): BIPN with "like" as its only behavior, on the same
rem      three CSVs and with the same evaluation. One training run, not a grid search.
rem   6. when that has finished, starts the recently_read cut-off search by itself
rem      (BIPN\scripts\search_recently_read.bat): BIPN with recently_read + like at the latest
rem      10, 20, ... 90 percent of each user's rows on a sample of the users, then the
rem      best 3 cut-offs on all users, each compared with plain BIPN (SEARCH_SAMPLE and
rem      SEARCH_TOP in BIPN\scripts\setup_and_train_anilist.bat).
rem THEN_BIPN below turns steps 5 and 6 off.
rem Safe to close and run again: finished combinations are skipped, and the BIPN run
rem continues from its last save.
rem
rem What is kept, in LightGCN\results:
rem   best_result.json   the best hyper-parameters, their validation and test result
rem   best_model.pth     the trained model of that best run
rem   grid_results.csv   every finished combination, best first
rem The BIPN baseline's result: BIPN\check_point\anilist_likes_result.json
rem
rem To force a PyTorch build:  run_lightgcn.bat cpu   (or cu113, cu128)
setlocal

rem ---- Settings ---------------------------------------------------------------
rem What happens when the LightGCN grid search has finished:
rem 0 = nothing more
rem 1 = the plain BIPN baseline (likes only)
rem 2 = the plain BIPN baseline, then the recently_read cut-off search
set "THEN_BIPN=2"
rem ---------------------------------------------------------------------------

title LightGCN on AniList - grid search
echo ============================================================
echo  LightGCN on AniList - grid search
echo ============================================================
echo.
if not exist "%~dp0..\..\BIPN\scripts\setup_and_train_anilist.bat" goto fail_nosetup
findstr /c:":setup_done" "%~dp0..\..\BIPN\scripts\setup_and_train_anilist.bat" >nul
if errorlevel 1 goto fail_nosetup
set "PYEXE="
set "DEV="
call "%~dp0..\..\BIPN\scripts\setup_and_train_anilist.bat" setup %*
if errorlevel 1 exit /b 1
if not defined PYEXE goto fail_nosetup
if not exist "%~dp0..\grid_search.py" goto fail_nofiles
echo.
"%PYEXE%" "%~dp0..\grid_search.py" --device %DEV%
if errorlevel 1 goto failed
if "%THEN_BIPN%"=="0" goto finished
if not exist "%~dp0..\..\BIPN\scripts\train_likes_only.bat" goto finished
echo.
echo ============================================================
echo  The LightGCN grid search has finished.
echo  Next: the plain BIPN baseline (likes only), same CSVs.
echo ============================================================
echo.
rem CHAINED: the BIPN baseline does not wait for a key when it has finished, so that the
rem next step starts by itself. It still stops and waits if something goes wrong.
if "%THEN_BIPN%"=="2" set "CHAINED=1"
call "%~dp0..\..\BIPN\scripts\train_likes_only.bat" %*
if errorlevel 1 exit /b 1
if not "%THEN_BIPN%"=="2" exit /b 0
set "CHAINED="
if not exist "%~dp0..\..\BIPN\scripts\search_recently_read.bat" goto finished
echo.
echo ============================================================
echo  The plain BIPN baseline has finished.
echo  Next: the recently_read cut-off search, same CSVs.
echo ============================================================
echo.
call "%~dp0..\..\BIPN\scripts\search_recently_read.bat" %*
exit /b %errorlevel%

:finished
echo.
pause
exit /b 0

:fail_nosetup
echo ERROR: BIPN\scripts\setup_and_train_anilist.bat was not found, or it is an older
echo version without the setup mode. Keep the LightGCN folder next to the BIPN
echo folder, and use the setup_and_train_anilist.bat uploaded with this folder.
goto failed

:fail_nofiles
echo ERROR: LightGCN\grid_search.py is missing. Keep this file in LightGCN\scripts;
echo grid_search.py, lightgcn.py and preprocess.py belong one folder up, in LightGCN.
goto failed

:failed
echo.
pause
exit /b 1
