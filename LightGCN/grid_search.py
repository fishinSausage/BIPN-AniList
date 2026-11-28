#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grid search for LightGCN's hyper-parameters on a train and a valid CSV.

Started by run_lightgcn.bat. It asks for the two CSVs, keeps their rows with
scaled_like = 1 (preprocess.py) and trains LightGCN (lightgcn.py) once for every
combination of the two lists below. The combinations are ranked by validation recall@20.

The lists are the ranges searched in the LightGCN paper (Section 4.1.2): the L2
regularization coefficient lambda and the number of layers K. Everything else is fixed, as
in the paper; see the top of lightgcn.py. Shorten the lists to make the search faster.

Safe to stop and start again: finished combinations are skipped. A combination that was
running when the window was closed starts over.

What is kept, in the results folder next to this file:
    best_result.json   the best hyper-parameters and their validation result
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
KEYS = ['%s@%d' % (name, k) for k in lightgcn.TOPKS for name in ('recall', 'ndcg', 'hit')]


def log(text=''):
    print(text, flush=True)
    with open(os.path.join(RESULTS, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def write_json(path, content):
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=1)
    os.replace(path + '.tmp', path)


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
    return sorted(done, key=lambda r: (-r['validation'][lightgcn.SELECT], -r['validation']['ndcg@20']))


def show(done, total):
    """Prints the best 10 so far and writes the whole ranking to grid_results.csv."""
    rows = ranked(done)
    with open(os.path.join(RESULTS, 'grid_results.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'lambda', 'layers', 'best_epoch', 'epochs', 'minutes'] + KEYS)
        for rank, r in enumerate(rows, 1):
            writer.writerow([rank, '%g' % r['lambda'], r['layers'], r['best_epoch'], r['epochs'],
                             '%.1f' % (r['seconds'] / 60)] + ['%.4f' % r['validation'][key] for key in KEYS])
    log('\n%4s %8s %6s %6s | %9s %8s %7s %7s' % ('rank', 'lambda', 'layers', 'epoch', 'recall@20', 'ndcg@20', 'hit@20', 'ndcg@10'))
    for rank, r in enumerate(rows[:10], 1):
        v = r['validation']
        log('%4d %8g %6d %6d | %9.4f %8.4f %7.4f %7.4f'
            % (rank, r['lambda'], r['layers'], r['best_epoch'], v['recall@20'], v['ndcg@20'], v['hit@20'], v['ndcg@10']))
    log('%d of %d combinations finished (validation, best 10 shown).' % (len(done), total))
    return rows


def save_best(best, data, search, state=None):
    """Remembers the best combination: best_result.json, and best_model.pth when its embeddings are given."""
    if state is not None:
        path = os.path.join(RESULTS, 'best_model.pth')
        torch.save({'user_embedding': state['user_embedding'], 'item_embedding': state['item_embedding'],
                    'layers': best['layers'], 'lambda': best['lambda'],
                    'user_ids': torch.from_numpy(data['user_ids']), 'item_ids': torch.from_numpy(data['item_ids'])},
                   path + '.tmp')
        os.replace(path + '.tmp', path)
    write_json(os.path.join(RESULTS, 'best_result.json'), {
        'model': 'LightGCN', 'chosen_by': 'validation ' + lightgcn.SELECT,
        'best_hyperparameters': {'lambda': best['lambda'], 'layers': best['layers']},
        'best_epoch': best['best_epoch'], 'validation': best['validation'],
        'fixed_settings': search['settings'], 'searched': {'lambda': LAMBDAS, 'layers': LAYERS},
        'train_csv': data['train_file'], 'valid_csv': data['valid_file'],
        'users': int(data['n_users']), 'titles': int(data['n_items']),
        'train_likes': int(len(data['train_u'])), 'valid_likes': int(len(data['valid_u'])),
        'weights': 'best_model.pth'})


def main():
    parser = argparse.ArgumentParser(description='Grid search for LightGCN on a train and a valid CSV.')
    parser.add_argument('--train', help='train CSV; an Explorer window asks for it when left out')
    parser.add_argument('--valid', help='valid CSV; an Explorer window asks for it when left out')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=lightgcn.MAX_EPOCHS,
                        help='epoch limit of every run (the paper trains up to 1000)')
    args = parser.parse_args()

    device = args.device
    if device != 'cpu' and not torch.cuda.is_available():
        print('WARNING: no usable NVIDIA GPU was found. Training on the CPU is extremely slow.')
        device = 'cpu'

    print('[2/4] Train and valid CSV ...', flush=True)
    train_file, valid_file = preprocess.pick_files(args.train, args.valid)
    print('\n[3/4] Preprocessing: rows with %s = 1 ...' % preprocess.LIKE_COL, flush=True)
    data = preprocess.load(train_file, valid_file)

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
    log('      To stop, close this window. Run run_lightgcn.bat again to continue.')

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
        run = {'lambda': lam, 'layers': layers, 'best_epoch': result['best_epoch'], 'epochs': result['epochs'],
               'validation': result['metrics'], 'seconds': result['seconds'],
               'seconds_per_epoch': result['seconds_per_epoch']}
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
    v = best['validation']
    log('\n============================================================')
    log(' Best hyper-parameters: lambda = %g, layers = %d  (best epoch %d)' % (best['lambda'], best['layers'], best['best_epoch']))
    log(' Validation: recall@20 %.4f | ndcg@20 %.4f | hit@20 %.4f' % (v['recall@20'], v['ndcg@20'], v['hit@20']))
    log(' Remembered in %s' % RESULTS)
    log('   best_result.json  best_model.pth  grid_results.csv')
    log('============================================================')
    if failed:
        log('\n%d combination(s) did not finish: %s' % (len(failed), ', '.join(failed)))
        sys.exit(1)


if __name__ == '__main__':
    main()
