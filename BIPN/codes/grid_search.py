#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grid search for BIPN's hyper-parameters on the AniList user sample.

Started by grid_search_hyperparams.bat. Every combination of the three lists below
is trained once with main.py, and the combinations are ranked by validation
ndcg@20, the same number training uses to choose its best epoch. recall@20 and
ndcg@20 are computed as in the LightGCN baseline (validation / test rows with
eval_like = 1).

Safe to stop and start again: finished combinations are skipped, and the one
that was running continues from its last save.

The lists are the search ranges given in the paper (Section 4.1.2):
learning rate, beta (cross-entropy loss against BPR loss) and gamma
(regularization). Shorten them to make the search faster.
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time

LEARNING_RATES = [0.01, 0.003, 0.001, 0.0001]   # --lr
BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]               # --log_reg
GAMMAS = [0.01, 0.001, 0.0001]                  # --reg_weight
FIRST = (0.001, 0.3, 0.001)                     # the authors' ML10M setting is tried first

DATA_NAME = 'anilist_sample'
MAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')   # main.py sits next to this file
ROOT = os.getcwd()   # the BIPN folder (holds data, check_point and log); the .bat file starts this script from there
CHECK_POINT = os.path.join(ROOT, 'check_point')
RESULTS_CSV = os.path.join(ROOT, 'grid_search_results.csv')
KEY = ''   # set in main()


def data_key():
    """Identifies the CSVs the sample was made from, so results of other CSVs are never mixed in."""
    path = os.path.join(ROOT, 'data', 'AniList_sample', 'data_key.txt')
    if not os.path.exists(path):
        sys.exit('ERROR: %s is missing. Start the grid search with grid_search_hyperparams.bat.' % path)
    with open(path) as f:
        return f.read().strip()


def run_name(lr, beta, gamma):
    return 'grid_%s_lr%g_b%g_g%g' % (KEY[:6], lr, beta, gamma)


def read_result(name):
    path = os.path.join(CHECK_POINT, name + '_result.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def metric(result, part, key):
    return float((result.get(part) or {}).get(key, 0.0))


def show(done, total):
    """Prints the best 10 so far and writes the whole ranking to grid_search_results.csv."""
    rows = sorted(done, key=lambda r: (-metric(r[3], 'validation', 'ndcg@20'), -metric(r[3], 'validation', 'recall@20')))
    print('\n%4s %8s %5s %7s %6s | %-19s | %-19s' % ('rank', 'lr', 'beta', 'gamma', 'epoch', 'validation', 'test'))
    print('%33s | %9s %9s | %9s %9s' % ('', 'recall@20', 'ndcg@20', 'recall@20', 'ndcg@20'))
    with open(RESULTS_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'lr', 'beta', 'gamma', 'best_epoch'] +
                        ['%s_%s' % (part, key) for part in ('validation', 'test')
                         for key in ('recall@20', 'ndcg@20')])
        for rank, (lr, beta, gamma, res) in enumerate(rows, 1):
            v = [metric(res, part, key) for part in ('validation', 'test')
                 for key in ('recall@20', 'ndcg@20')]
            if rank <= 10:
                print('%4d %8g %5g %7g %6s | %9.4f %9.4f | %9.4f %9.4f'
                      % (rank, lr, beta, gamma, res.get('best_epoch', '?'), v[0], v[1], v[2], v[3]))
            writer.writerow([rank, lr, beta, gamma, res.get('best_epoch', '')] + ['%.4f' % x for x in v])
    print('%d of %d combinations finished (ranked by validation ndcg@20, best 10 shown). Full table: %s'
          % (len(done), total, RESULTS_CSV), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Grid search on the AniList sample.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', default='512')
    parser.add_argument('--save_minutes', default='30')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(ROOT, 'data')):
        sys.exit('ERROR: no data folder in %s. Start the grid search with grid_search_hyperparams.bat.' % ROOT)
    global KEY
    KEY = data_key()
    combos = list(itertools.product(LEARNING_RATES, BETAS, GAMMAS))
    if FIRST in combos:
        combos.remove(FIRST)
        combos.insert(0, FIRST)
    os.makedirs(CHECK_POINT, exist_ok=True)
    print('Grid search: %d combinations on %s (device %s)' % (len(combos), DATA_NAME, args.device), flush=True)

    done, failed, seconds = [], [], []
    for lr, beta, gamma in combos:
        result = read_result(run_name(lr, beta, gamma))
        if result is not None:
            done.append((lr, beta, gamma, result))
    if done:
        print('%d combinations were already finished and are skipped.' % len(done))

    for lr, beta, gamma in combos:
        name = run_name(lr, beta, gamma)
        if read_result(name) is not None:
            continue
        left = len(combos) - len(done) - len(failed)
        eta = ''
        if seconds:
            eta = ' | about %.1f hours left at %.1f hours per combination' % (
                left * sum(seconds) / len(seconds) / 3600, sum(seconds) / len(seconds) / 3600)
        print('\n=== %d of %d: lr=%g beta=%g gamma=%g%s ===' % (
            len(done) + len(failed) + 1, len(combos), lr, beta, gamma, eta), flush=True)
        start = time.time()
        code = subprocess.call([sys.executable, MAIN_PY, '--data_name', DATA_NAME, '--device', args.device,
                                '--lr', '%g' % lr, '--log_reg', '%g' % beta, '--reg_weight', '%g' % gamma,
                                '--decay', '0', '--batch_size', args.batch_size, '--model_name', name,
                                '--resume', '--save_minutes', args.save_minutes], cwd=ROOT)
        result = read_result(name)
        if code != 0 or result is None:
            failed.append(name)
            print('This combination did not finish (see the error above). It is retried the next time '
                  'the grid search is started.', flush=True)
            continue
        seconds.append(time.time() - start)
        done.append((lr, beta, gamma, result))
        show(done, len(combos))

    if not done:
        sys.exit('\nNo combination finished. Fix the error shown above and start the grid search again.')
    rows = show(done, len(combos))
    lr, beta, gamma, best = rows[0]
    print('\nBest so far: lr=%g  beta=%g  gamma=%g  (validation ndcg@20 %.4f, test ndcg@20 %.4f)'
          % (lr, beta, gamma, metric(best, 'validation', 'ndcg@20'), metric(best, 'test', 'ndcg@20')))
    print('To train the full data with it, open setup_and_train_anilist.bat in Notepad and set')
    print('    set "LR=%g"\n    set "BETA=%g"\n    set "GAMMA=%g"' % (lr, beta, gamma))
    if failed:
        print('\n%d combination(s) did not finish: %s' % (len(failed), ', '.join(failed)))
        sys.exit(1)


if __name__ == '__main__':
    main()
