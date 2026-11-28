@echo off
rem Plain BIPN baseline: BIPN trained with "like" (scaled_like = 1) as its only behavior.
rem Double-click to run. It uses setup_and_train_anilist.bat for everything:
rem   - the same train, validation and test csv as the LightGCN baseline (the ones chosen
rem     for the LightGCN grid search; if there are none yet, it asks for them)
rem   - the same evaluation: recall@20 and ndcg@20 on the validation / test rows with
rem     eval_like = 1, best epoch chosen by validation ndcg@20
rem   - one training run with the settings at the top of setup_and_train_anilist.bat,
rem     not a grid search
rem ..\..\LightGCN\scripts\run_lightgcn.bat starts this by itself when its grid search has finished.
rem Safe to close and run again: training continues from its last save.
rem Result: BIPN\check_point\anilist_likes_result.json
call "%~dp0setup_and_train_anilist.bat" likes %*
