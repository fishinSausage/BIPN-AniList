!!!!!!\
double-click START.bat and select the train, validation and test .csv when it asks\
!!!!!!

## What START.bat runs

| stage | what | what it keeps |
|---|---|---|
| 1 | LightGCN grid search (20 combinations) | `LightGCN/results/`: best hyper-parameters, result, weights |
| 2 | plain BIPN, "like" as its only behavior (one run) | `BIPN/check_point/anilist_likes_result.json` |
| 3 | cut-off search of the derived behavior `recently_read` | `BIPN/recently_read_search_results.csv`, `BIPN/recently_read_search_best.json` |

All stages read the same three CSVs and are scored the same way: training likes are the
train rows with `scaled_like = 1`, the answers are the validation / test rows with
`eval_like = 1`, the measures are Recall@20 and NDCG@20, and the best epoch is chosen by
validation NDCG@20. The top part of the window keeps every result in sight.

To run one stage alone, use the start files in `LightGCN/scripts` and `BIPN/scripts`.
BIPN's settings are at the top of `BIPN/scripts/setup_and_train_anilist.bat`.

## How a new column is tested (the cut-off search)

A derived column is a behavior that is not in the CSVs but is worked out from the train
csv with one number, its cut-off (`recently_read`: the latest N percent of each user's
rows by `updated_at`). Every derived column is tested the same way, by
`BIPN/codes/column_search.py`:

1. **Sample (about 3,500 users, `SEARCH_SAMPLE=0.1`)**: plain BIPN and BIPN with the
   column + like at every cut-off of the column's list. The best 3 cut-offs by validation
   NDCG@20 (`SEARCH_TOP=3`) become the candidates.
2. **All users (about 35,000)**: plain BIPN and the 3 candidates. The best one by
   validation NDCG@20 is the cut-off that is kept.

Plain BIPN on the same users is the baseline in both steps; test numbers are only reported.

To add a column:

1. `BIPN/data/AniList/data_process.py`: write a function that returns, for one user's
   train rows and a cut-off, the titles that get the behavior, and list it in `DERIVED`.
   Use the train csv only.
2. `BIPN/codes/column_search.py`: add the column and its cut-offs (in percent) to `COLUMNS`.
3. `BIPN/scripts`: copy `search_recently_read.bat` and change `SEARCH_COLUMN` in the copy.
