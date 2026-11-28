@echo off
rem Does a "liked_author" behavior help BIPN, and which cut-off is best? Double-click to run.
rem liked_author is not a column of the CSVs: it is derived from authors_json and scaled_like
rem of the train csv. A title gets it when the user likes more than the cut-off share of the
rem OTHER titles by one of its authors; the title itself is left out, so its own like is not
rem part of the behavior.
rem
rem It uses setup_and_train_anilist.bat for the setup and then runs codes\column_search.py:
rem plain BIPN and BIPN with liked_author + like at the cut-offs above 0, 20, 40, 50 and 90
rem percent, each on all users (this column marks few rows, so there is no sample step).
rem The best one by validation ndcg@20 is kept.
rem Every run uses the same train, validation and test csv, the settings at the top of
rem setup_and_train_anilist.bat and the same evaluation as the LightGCN baseline.
rem
rem Safe to close and run again: finished runs are skipped.
rem Kept in the BIPN folder: liked_author_search_results.csv (every run) and
rem liked_author_search_best.json (the cut-off that was kept).
set "SEARCH_COLUMN=liked_author"
call "%~dp0setup_and_train_anilist.bat" column %*
