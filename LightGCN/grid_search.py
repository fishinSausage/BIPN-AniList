#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grid search for LightGCN's hyper-parameters on a train, a validation and a test CSV.

Started by START.bat (or LightGCN/scripts/run_lightgcn.bat). It asks for the three CSVs, reads them (preprocess.py) and
trains LightGCN (lightgcn.py) once for every combination of the two lists below.

  * trained on the train rows with scaled_like = 1
  * the combinations are ranked by ndcg@20 (SELECT in lightgcn.py) on the validation rows
    with eval_like = 1
  * recall@20 and ndcg@20 on the test rows with eval_like = 1 are reported for the best
    epoch of every combination; they are never used to choose anything

The lists are the ranges searched in the LightGCN paper (Section 4.1.2): the L2
regularization coefficient lambda and the number of layers K. Everything else is fixed, as
in the paper; see the top of lightgcn.py. Shorten the lists to make the search faster.

Safe to stop and start again: finished combinations are skipped. A combination that was
running when the window was closed starts over.

What is kept, in the results folder next to this file:
    best_result.json   the best hyper-parameters, their validation and test result
    best_model.pth     the embeddings of that best run
    grid_results.csv   every finished combination, best first
    log.txt            everything that was printed
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import torch

import lightgcn
import preprocess

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]   # L2 regularization coefficient
LAYERS = [1, 2, 3, 4]                      # number of layers K
FIRST = (1e-4, 3)                          # the paper's usual best is tried first

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
RUNS = os.path.join(RESULTS, 'runs')
RECALL, NDCG = lightgcn.METRICS


def log(text=''):
    print(text, flush=True)
    with open(os.path.join(RESULTS, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def replace(tmp, path):
    """os.replace, tried again for a few seconds: Windows refuses it while another program has the file open."""
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def write_json(path, content):
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=1)
    replace(path + '.tmp', path)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_name(lam, layers):
    return 'lambda%g_layers%d' % (lam, layers)


def prepare_results(search):
    """Makes the results folder. Results of other CSVs or other settings are moved aside, not deleted."""
    old = read_json(os.path.join(RESULTS, 'search.json'))
    if os.path.isdir(RESULTS) and old != search:
        moved = RESULTS + time.strftime('_%Y-%m-%d_%H-%M-%S')
        os.rename(RESULTS, moved)
        print('The results folder was made from other CSV files or settings. It was kept as\n      %s' % moved)
    os.makedirs(RUNS, exist_ok=True)
    write_json(os.path.join(RESULTS, 'search.json'), search)


def ranked(done):
    """Best first: by lightgcn.SELECT on validation; equal values are ordered by the other measure."""
    other = NDCG if lightgcn.SELECT == RECALL else RECALL
    return sorted(done, key=lambda r: (-r['validation'][lightgcn.SELECT], -r['validation'][other]))


def show(done, total):
    """Prints the best 10 so far and writes the whole ranking to grid_results.csv."""
    rows = ranked(done)
    with open(os.path.join(RESULTS, 'grid_results.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'lambda', 'layers', 'best_epoch', 'epochs', 'minutes',
                         'validation_' + RECALL, 'validation_' + NDCG, 'test_' + RECALL, 'test_' + NDCG])
        for rank, r in enumerate(rows, 1):
            writer.writerow([rank, '%g' % r['lambda'], r['layers'], r['best_epoch'], r['epochs'], '%.1f' % (r['seconds'] / 60)]
                            + ['%.4f' % r[part][key] for part in ('validation', 'test') for key in (RECALL, NDCG)])
    log('\n%4s %8s %6s %6s | %-19s | %-19s' % ('rank', 'lambda', 'layers', 'epoch', 'validation', 'test'))
    log('%27s | %9s %9s | %9s %9s' % ('', RECALL, NDCG, RECALL, NDCG))
    for rank, r in enumerate(rows[:10], 1):
        log('%4d %8g %6d %6d | %9.4f %9.4f | %9.4f %9.4f' % (
            rank, r['lambda'], r['layers'], r['best_epoch'],
            r['validation'][RECALL], r['validation'][NDCG], r['test'][RECALL], r['test'][NDCG]))
    log('%d of %d combinations finished (ranked by validation %s, best 10 shown).' % (len(done), total, lightgcn.SELECT))
    log('best so far: lambda=%g layers=%d | validation %s %.4f'
        % (rows[0]['lambda'], rows[0]['layers'], lightgcn.SELECT, rows[0]['validation'][lightgcn.SELECT]))
    return rows


def save_best(best, data, search, state=None):
    """Remembers the best combination: best_result.json, and best_model.pth when its embeddings are given."""
    if state is not None:
        path = os.path.join(RESULTS, 'best_model.pth')
        torch.save(dict(state, layers=best['layers'], **{'lambda': best['lambda']},
                        user_ids=torch.from_numpy(data['user_ids']), item_ids=torch.from_numpy(data['item_ids'])),
                   path + '.tmp')
        replace(path + '.tmp', path)
    write_json(os.path.join(RESULTS, 'best_result.json'), {
        'model': 'LightGCN', 'chosen_by': 'validation ' + lightgcn.SELECT,
        'best_hyperparameters': {'lambda': best['lambda'], 'layers': best['layers']},
        'best_epoch': best['best_epoch'], 'validation': best['validation'], 'test': best['test'],
        'fixed_settings': search['settings'], 'searched': {'lambda': LAMBDAS, 'layers': LAYERS},
        'train_column': preprocess.TRAIN_LIKE_COL, 'answer_column': preprocess.EVAL_LIKE_COL,
        'files': data['files'], 'users': int(data['n_users']), 'titles': int(data['n_items']),
        'train_likes': int(len(data['train_u'])), 'validation_answers': int(len(data['valid_u'])),
        'test_answers': int(len(data['test_u'])), 'weights': 'best_model.pth'})


def main():
    parser = argparse.ArgumentParser(description='Grid search for LightGCN on a train, a validation and a test CSV.')
    for part in preprocess.PARTS:
        parser.add_argument('--' + part, help='%s CSV; an Explorer window asks for it when left out' % part)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=lightgcn.MAX_EPOCHS,
                        help='epoch limit of every run (the paper trains up to 1000)')
    args = parser.parse_args()

    device = args.device
    if device != 'cpu' and not torch.cuda.is_available():
        print('WARNING: no usable NVIDIA GPU was found. Training on the CPU is extremely slow.')
        device = 'cpu'

    print('[2/4] Train, validation and test CSV ...', flush=True)
    paths = preprocess.pick_files({part: getattr(args, part) for part in preprocess.PARTS})
    print('\n[3/4] Preprocessing: train rows with %s = 1, validation and test rows with %s = 1 ...'
          % (preprocess.TRAIN_LIKE_COL, preprocess.EVAL_LIKE_COL), flush=True)
    data = preprocess.load(paths)

    settings = dict(lightgcn.settings(), max_epochs=args.max_epochs)
    search = {'data': data['key'], 'settings': settings}
    prepare_results(search)
    combos = list(itertools.product(LAMBDAS, LAYERS))
    if FIRST in combos:
        combos.remove(FIRST)
        combos.insert(0, FIRST)
    name = torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'
    log('\n[4/4] Grid search: %d combinations, device %s (%s), started %s'
        % (len(combos), device, name, time.strftime('%Y-%m-%d %H:%M')))
    log('      To stop, close this window. Start it again to continue.')

    done = [r for r in (read_json(os.path.join(RUNS, run_name(*c) + '.json')) for c in combos) if r]
    if done:
        log('%d combinations were already finished and are skipped.' % len(done))
    failed, seconds = [], []
    for lam, layers in combos:
        run_file = os.path.join(RUNS, run_name(lam, layers) + '.json')
        if os.path.exists(run_file) and read_json(run_file):
            continue
        eta = ''
        if seconds:
            mean = sum(seconds) / len(seconds)
            eta = ' | about %.1f hours left at %.1f hours per combination' % (
                (len(combos) - len(done) - len(failed)) * mean / 3600, mean / 3600)
        log('\n=== %d of %d: lambda=%g layers=%d%s ===' % (len(done) + len(failed) + 1, len(combos), lam, layers, eta))
        try:
            result = lightgcn.train(data, layers, lam, device, max_epochs=args.max_epochs, log=log)
        except RuntimeError as error:
            failed.append(run_name(lam, layers))
            log('This combination did not finish: %s' % error)
            log('It is tried again the next time the grid search is started.')
            if device != 'cpu':
                torch.cuda.empty_cache()
            continue
        state = result.pop('state')
        run = dict(result, **{'lambda': lam, 'layers': layers})
        log('  best epoch %d | validation %s %.4f %s %.4f | test %s %.4f %s %.4f' % (
            run['best_epoch'], RECALL, run['validation'][RECALL], NDCG, run['validation'][NDCG],
            RECALL, run['test'][RECALL], NDCG, run['test'][NDCG]))
        seconds.append(result['seconds'])
        done.append(run)
        if ranked(done)[0] is run:   # best so far: remember it before the run counts as finished
            save_best(run, data, search, state)
        write_json(run_file, run)
        del state
        if device != 'cpu':
            torch.cuda.empty_cache()
        show(done, len(combos))

    if not done:
        sys.exit('\nNo combination finished. Fix the error shown above and start the grid search again.')
    best = show(done, len(combos))[0]
    save_best(best, data, search)
    log('\n============================================================')
    log(' Best hyper-parameters: lambda = %g, layers = %d  (best epoch %d)' % (best['lambda'], best['layers'], best['best_epoch']))
    log(' Validation: %s %.4f | %s %.4f' % (RECALL, best['validation'][RECALL], NDCG, best['validation'][NDCG]))
    log(' Test:       %s %.4f | %s %.4f' % (RECALL, best['test'][RECALL], NDCG, best['test'][NDCG]))
    log(' Remembered in %s' % RESULTS)
    log('   best_result.json  best_model.pth  grid_results.csv')
    log('============================================================')
    if failed:
        log('\n%d combination(s) did not finish: %s' % (len(failed), ', '.join(failed)))
        sys.exit(1)


if __name__ == '__main__':
    main()
