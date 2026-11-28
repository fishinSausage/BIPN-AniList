@echo off
rem Does a "recently_read" behavior help BIPN, and which cut-off is best? Double-click to run.
rem recently_read is not a column of the CSVs: it is derived from updated_at of the train csv
rem (the latest share of each user's rows). The share is the cut-off that is searched.
rem
rem It uses setup_and_train_anilist.bat for the setup and then runs codes\column_search.py:
rem   step 1  on a sample of the users (SEARCH_SAMPLE, about 3,500 users): plain BIPN and
rem           BIPN with recently_read + like at the latest 10, 20, ... 90 percent.
rem           The best SEARCH_TOP (3) cut-offs become the candidates.
rem   step 2  on all users: plain BIPN and the candidates. The best one is kept.
rem Every run uses the same train, validation and test csv, the settings at the top of
rem setup_and_train_anilist.bat and the same evaluation as the LightGCN baseline. The
rem cut-offs are compared by validation ndcg@20.
rem
rem Safe to close and run again: finished runs are skipped.
rem Kept in the BIPN folder: recently_read_search_results.csv (every run) and
rem recently_read_search_best.json (the cut-off that was kept).
rem
rem For another derived column: add it to DERIVED in data\AniList\data_process.py and to
rem COLUMNS in codes\column_search.py, copy this file and change SEARCH_COLUMN below.
set "SEARCH_COLUMN=recently_read"
call "%~dp0setup_and_train_anilist.bat" column %*
