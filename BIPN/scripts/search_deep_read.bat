@echo off
rem Does a "deep_read" behavior help BIPN, and which cut-off is best? Double-click to run.
rem deep_read is not a column of the CSVs: it is derived from the train csv. A title is
rem deep_read when the user is reading it or stopped reading it (status CURRENT, PAUSED or
rem DROPPED), it has at least 20 chapters, and progress / chapters is at least the cut-off.
rem
rem Same procedure as search_recently_read.bat (codes\column_search.py):
rem   step 1  on a sample of the users (SEARCH_SAMPLE): plain BIPN and BIPN with
rem           deep_read + like at 10, 20, ... 90 percent of the chapters.
rem           The best SEARCH_TOP cut-offs become the candidates.
rem   step 2  on all users: plain BIPN and the candidates. The best one is kept.
rem The cut-offs are compared by validation ndcg@20. SEARCH_SAMPLE, SEARCH_TOP and the
rem training settings are at the top of setup_and_train_anilist.bat.
rem
rem Safe to close and run again: finished runs are skipped.
rem Kept in the BIPN folder: deep_read_search_results.csv and deep_read_search_best.json.
set "SEARCH_COLUMN=deep_read"
call "%~dp0setup_and_train_anilist.bat" column %*
