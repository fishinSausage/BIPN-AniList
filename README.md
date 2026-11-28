!!!!!!\
double-click START.bat and select the train, validation and test .csv when it asks\
!!!!!!

## What START.bat runs

| stage | what | what it keeps |
|---|---|---|
| 1 | LightGCN grid search (20 combinations) | `LightGCN/results/`: best hyper-parameters, result, weights |
| 2 | plain BIPN, "like" as its only behavior (one run) | `BIPN/check_point/anilist_likes_result.json` |
| 3 | cut-off search of the derived behavior `recently_read` | `BIPN/recently_read_search_results.csv`, `BIPN/recently_read_search_best.json` |
| 4 | cut-off search of the derived behavior `liked_author` | `BIPN/liked_author_search_results.csv`, `BIPN/liked_author_search_best.json` |

All stages read the same three CSVs and are scored the same way: training likes are the
train rows with `scaled_like = 1`, the answers are the validation / test rows with
`eval_like = 1`, the measures are Recall@20 and NDCG@20, and the best epoch is chosen by
validation NDCG@20. The top part of the window keeps every result in sight.

To run one stage alone, use the start files in `LightGCN/scripts` and `BIPN/scripts`.
BIPN's settings are at the top of `BIPN/scripts/setup_and_train_anilist.bat`.

## How a new column is tested (the cut-off search)

A derived column is a behavior that is not in the CSVs but is worked out from the train
csv with one number, its cut-off. Every derived column is tested by
`BIPN/codes/column_search.py`, always against plain BIPN (likes only) on the same users,
by validation NDCG@20; test numbers are only reported.

The standard procedure (used for `recently_read`: the latest N percent of each user's rows
by `updated_at`, N = 10, 20, ... 90):

1. **Sample (about 3,500 users, `SEARCH_SAMPLE=0.1`)**: plain BIPN and BIPN with the
   column + like at every cut-off. The best 3 cut-offs (`SEARCH_TOP=3`) become the candidates.
2. **All users (about 35,000)**: plain BIPN and the 3 candidates. The best one is kept.

A column that marks few rows skips the sample and runs each of its few cut-offs on all
users (`all_users` in `COLUMNS`). This is used for `liked_author`: a title gets it when the
user likes more than N percent of the *other* titles by one of its authors (N = 0, 20, 40,
50, 90). The title itself is left out, so its own like is not part of the behavior.

To add a column:

1. `BIPN/data/AniList/data_process.py`: write a function that returns, for one user's
   train rows and a cut-off, the titles that get the behavior, and list it in `DERIVED`.
   Use the train csv only, and never let a row's own like decide the row's own behavior.
2. `BIPN/codes/column_search.py`: add the column and its cut-offs (in percent) to `COLUMNS`.
3. `BIPN/scripts`: copy `search_recently_read.bat` and change `SEARCH_COLUMN` in the copy.
4. `scripts/dashboard.py`: add the column to `SEARCH_COLUMNS` so that START.bat runs it.

Two more columns have only their own start file in `BIPN/scripts` (START.bat does not run them):

| column | a title gets the behavior when | cut-offs | start file |
|---|---|---|---|
| `deep_read` | status is CURRENT, PAUSED or DROPPED, it has at least 20 chapters, and `progress / chapters` is at least N percent | 10 ... 90 percent, sample first | `search_deep_read.bat` |
| `reread` | `repeat` is at least N and status is not PLANNING | 1, 2, 3, 4 (times, not percent), all users | `search_reread.bat` |
