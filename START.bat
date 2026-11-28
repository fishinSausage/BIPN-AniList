@echo off
rem AniList baselines: everything in one go. Double-click to run.
rem
rem   stage 1  LightGCN grid search: 20 combinations; keeps the best hyper-parameters,
rem            their result and the weights (LightGCN\results)
rem   stage 2  plain BIPN, "like" as its only behavior: one training run
rem            (BIPN\check_point\anilist_likes_result.json)
rem   stage 3  cut-off search of the derived behavior recently_read:
rem            step 1  the latest 10, 20, ... 90 percent of each user's rows, on a sample
rem                    of the users (about 3,500); the best 3 become the candidates
rem            step 2  the 3 candidates on all users; the best one is kept
rem            each compared with plain BIPN on the same users
rem            (BIPN\recently_read_search_results.csv, BIPN\recently_read_search_best.json)
rem
rem It asks once for the TRAIN, VALIDATION and TEST csv; all three stages read those files
rem and are scored the same way (recall@20 and ndcg@20 on the rows with eval_like = 1, best
rem epoch by validation ndcg@20).
rem
rem The window is split in two: the top part stays in place and shows every model's result
rem in the same columns, the bottom part shows what the running stage prints.
rem
rem Safe to close and run again: everything that has finished is skipped.
rem To run one stage alone, use the start files in LightGCN\scripts and BIPN\scripts.
rem BIPN's settings, the sample share and the number of candidates are at the top of
rem BIPN\scripts\setup_and_train_anilist.bat.
rem To force a PyTorch build:  START.bat cpu   (or cu113, cu128)
setlocal
title AniList baselines
if not exist "%~dp0BIPN\scripts\setup_and_train_anilist.bat" goto fail_files
if not exist "%~dp0scripts\dashboard.py" goto fail_files
set "PYEXE="
set "DEV="
call "%~dp0BIPN\scripts\setup_and_train_anilist.bat" setup %*
if errorlevel 1 exit /b 1
if not defined PYEXE goto fail_files
rem A window large enough for the summary (25 rows) and the running stage's output.
mode con: cols=116 lines=42 >nul 2>nul
"%PYEXE%" "%~dp0scripts\dashboard.py" --device %DEV%
set "CODE=%errorlevel%"
rem 0 = all stages finished, 3 = a stage failed; in both cases the screen has already waited
rem for a key. Anything else is an unexpected stop, so its message stays up.
if "%CODE%"=="0" exit /b 0
if "%CODE%"=="3" exit /b 1
goto failed

:fail_files
echo ERROR: a file that START.bat needs is missing. Keep START.bat at the top of the
echo repository, next to the BIPN, LightGCN and scripts folders.
goto failed

:failed
echo.
pause
exit /b 1
