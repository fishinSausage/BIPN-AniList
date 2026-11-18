@echo off
rem Grid search for BIPN's hyper-parameters on a sample of AniList users.
rem Double-click to run. It uses setup_and_train_anilist.bat for the setup, makes
rem the user sample (data\AniList_sample) and then trains every combination of
rem the values listed at the top of codes\grid_search.py, one after another.
rem Safe to close and run again: finished combinations are skipped.
rem The ranking is written to grid_search_results.csv.
call "%~dp0setup_and_train_anilist.bat" grid %*
