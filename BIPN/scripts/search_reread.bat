@echo off
rem Does a "reread" behavior help BIPN, and which cut-off is best? Double-click to run.
rem reread is not a column of the CSVs: it is derived from the train csv. A title is reread
rem when the user has read it again at least N times (repeat is N or more) and its status
rem is not PLANNING. N is the cut-off that is searched: 1, 2, 3 and 4.
rem
rem It runs codes\column_search.py like search_recently_read.bat, but without step 1:
rem there are only 4 cut-offs, so plain BIPN and all 4 run on all users (all_users in
rem COLUMNS of column_search.py) and the best one by validation ndcg@20 is kept. The training settings
rem are at the top of setup_and_train_anilist.bat.
rem
rem Safe to close and run again: finished runs are skipped.
rem Kept in the BIPN folder: reread_search_results.csv and reread_search_best.json.
set "SEARCH_COLUMN=reread"
call "%~dp0setup_and_train_anilist.bat" column %*
