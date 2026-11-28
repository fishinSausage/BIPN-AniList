#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does one more, derived behavior help BIPN, and which cut-off is best?

Started by search_recently_read.bat for the column recently_read, and by search_<column>.bat
for the other columns. A derived behavior is
not a column of the CSVs: the converter works it out from the train csv with one number,
the cut-off (data/AniList/data_process.py, DERIVED).

The procedure is the same for every column:

    step 1, on a sample of the users (SEARCH_SAMPLE, about 3,500 users at 0.1)
        plain BIPN ("like" is the only behavior) and BIPN with the column + like at every
        cut-off of the column's list in COLUMNS. The SEARCH_TOP best cut-offs by validation
        ndcg@20 are the candidates.
    step 2, on all users
        plain BIPN and BIPN with the column + like at each candidate. The best of them by
        validation ndcg@20 is the cut-off that is kept.

Step 1 is cheap and only narrows the list down, because neighboring cut-offs differ by
little and a run on all users takes hours; step 2 decides. A column marked all_users in
COLUMNS (liked_author) has no step 1: each of its few cut-offs is run on all users. Every run uses the same
settings and the same train, validation and test csv, and is scored like the LightGCN
baseline (recall@20 and ndcg@20 on the validation / test rows with eval_like = 1). Test is
only reported, never used to choose.

To add a column:
    1. data/AniList/data_process.py: write its function and list it in DERIVED
    2. here: add the column and its cut-offs (in percent) to COLUMNS
    3. scripts: copy search_recently_read.bat and change SEARCH_COLUMN in the copy

Safe to stop and start again: finished runs are skipped, and the run that was going on
continues from its last save. What is kept, next to the codes folder:
    <column>_search_results.csv   every run of both steps
    <column>_search_best.json     the cut-off that was kept and its results
"""
import argparse
import csv
import json
import os
import subprocess
import sys

# column -> how its cut-off is searched
#   percents   the cut-offs to try, in percent (what a cut-off means is decided by the column's function)
#   sign       put before the percent in the tables ('>' for "share above ...")
#   all_users  True = no sample step, every cut-off is run on all users. For a column that marks few rows:
#              a run is then cheap, and the differences are too small to rank the cut-offs on a sample.
#   count      True = the cut-offs are whole numbers given to the converter as they are, not percents
#              (the tables and files still call them cutoff_percent, so the screen can read them)
COLUMNS = {
    # the latest N percent of each user's rows
    'recently_read': {'percents': [10, 20, 30, 40, 50, 60, 70, 80, 90]},
    # the user likes more than N percent of the other titles by the author. The share is nearly always a
    # simple fraction (1 of 2, 1 of 3, all), so these five cut-offs are the ones that select different rows.
    'liked_author': {'percents': [0, 20, 40, 50, 90], 'sign': '>', 'all_users': True},
    # unfinished titles of 20+ chapters of which the user has read at least N percent
    'deep_read': {'percents': [10, 20, 30, 40, 50, 60, 70, 80, 90]},
    # titles the user has read again at least N times; only 4 cut-offs, so each is run on all users
    'reread': {'percents': [1, 2, 3, 4], 'count': True, 'all_users': True},
}
SELECT = 'ndcg@20'    # what main.py chooses the best epoch by (args.select_metric); the cut-offs are compared by it
OTHER = 'recall@20'

CODES = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(CODES, 'main.py')
ROOT = os.getcwd()   # the BIPN folder (holds data, check_point and log); the .bat file starts this script from there
DATA = os.path.join(ROOT, 'data')
DATA_PROCESS = os.path.join(DATA, 'AniList', 'data_process.py')
SOURCE_FILES = os.path.join(DATA, 'AniList', 'source_files.json')
CHECK_POINT = os.path.join(ROOT, 'check_point')
PARTS = ('train', 'validation', 'test')
KEYS = [(part, key) for part in ('validation', 'test') for key in (SELECT, OTHER)]


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def csv_files(files_json):
    """The three CSVs every run must use: the ones chosen for LightGCN, else the ones of the last conversion."""
    for path in (files_json, SOURCE_FILES):
        listed = read_json(path) if path else None
        if isinstance(listed, dict) and all(os.path.isfile(str(listed.get(part, ''))) for part in PARTS):
            return {part: listed[part] for part in PARTS}
    return None


def metric(result, part, key):
    return float((result.get(part) or {}).get(key, 0.0))


class Search:
    def __init__(self, args):
        self.args, self.column = args, args.column
        self.sign = COLUMNS[args.column].get('sign', '')
        self.unit = '' if COLUMNS[args.column].get('count') else '%'
        self.files = csv_files(args.files_json)
        self.rows = []   # one per finished run: {'step': 'sample' or 'all', 'percent': None or int, 'result': ...}
        self.candidates, self.best = [], None
        self.results_csv = os.path.join(ROOT, '%s_search_results.csv' % self.column)
        self.best_json = os.path.join(ROOT, '%s_search_best.json' % self.column)

    def label(self, percent):
        return 'like  (plain BIPN)' if percent is None else '+ %s %s%d%s' % (self.column, self.sign, percent, self.unit)

    def convert(self, folder, percent, sample):
        """Converts the CSVs into folder; the converter skips it if it is already done."""
        command = [sys.executable, DATA_PROCESS, '--behaviors', 'like', '--out', folder]
        if percent is not None:
            command += ['--extra', self.column, '--cutoff', repr(percent / 100.0 if self.unit else percent)]
        if sample < 1.0:
            command += ['--sample', repr(sample)]
        for part in PARTS:
            if self.files:
                command += ['--' + part, self.files[part]]
        if subprocess.call(command, cwd=ROOT) != 0:
            sys.exit('ERROR: converting the CSVs failed, see the message above.')
        self.files = self.files or csv_files(None)   # the first conversion asks for the CSVs; the later ones reuse them

    def train(self, data_name, folder):
        """Trains BIPN on a converted folder and returns its result, or None if the run did not finish."""
        args = self.args
        with open(os.path.join(folder, 'data_key.txt')) as f:
            key = f.read().strip()
        # plain BIPN on all users keeps the name train_likes_only.bat uses; every other run is named after its data
        model_name = data_name if data_name == 'anilist_likes' else '%s_%s' % (data_name, key[:6])
        result_file = os.path.join(CHECK_POINT, model_name + '_result.json')
        result = read_json(result_file)
        if result and result.get('settings', {}).get('data_key') == key:
            return result
        code = subprocess.call([sys.executable, MAIN_PY, '--data_name', data_name, '--data_dir', folder,
                                '--device', args.device, '--lr', args.lr, '--log_reg', args.log_reg,
                                '--reg_weight', args.reg_weight, '--decay', '0', '--batch_size', args.batch_size,
                                '--model_name', model_name, '--resume', '--save_minutes', args.save_minutes], cwd=ROOT)
        result = read_json(result_file)
        if code != 0 or not result or result.get('settings', {}).get('data_key') != key:
            return None
        return result

    def run_step(self, step, sample, percents):
        """Plain BIPN and every cut-off of percents on the given share of users. Returns the labels that failed."""
        tag = '' if sample >= 1.0 else '_s%g' % (100 * sample)   # a sample gets folders and names of its own
        users = 'all users' if sample >= 1.0 else 'sample'
        failed = []
        runs = [None] + list(percents)
        for number, percent in enumerate(runs, 1):
            print('\n=== %s %d of %d: %s ===' % (users, number, len(runs), self.label(percent)), flush=True)
            name = 'likes' if percent is None else '%s_%d' % (self.column, percent)
            folder = os.path.join(DATA, 'AniList_%s%s' % (name, tag))
            self.convert(folder, percent, sample)
            result = self.train('anilist_%s%s' % (name, tag), folder)
            if result is None:
                failed.append('%s, %s' % (self.label(percent), users))
                print('This run did not finish (see the error above). It is tried again the next time this is started.', flush=True)
                continue
            self.rows.append({'step': step, 'percent': percent, 'result': result})
            self.show()
        return failed

    def of_step(self, step, with_plain=True):
        return [r for r in self.rows if r['step'] == step and (with_plain or r['percent'] is not None)]

    def ranked(self, step):
        """The cut-offs of a step, best first by validation SELECT; equal values by the other measure."""
        return sorted(self.of_step(step, with_plain=False),
                      key=lambda r: (-metric(r['result'], 'validation', SELECT), -metric(r['result'], 'validation', OTHER)))

    def show(self):
        """Prints both steps and writes them to <column>_search_results.csv."""
        with open(self.results_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'rank', 'behaviors', 'cutoff_percent', 'best_epoch'] + ['%s_%s' % k for k in KEYS]
                            + ['validation_%s_minus_plain' % SELECT, 'candidate', 'best'])
            for step, title in (('sample', 'step 1: sample of the users'), ('all', 'step 2: all users')):
                rows = self.of_step(step)
                if not rows:
                    continue
                plain = next((r for r in rows if r['percent'] is None), None)
                base = metric(plain['result'], 'validation', SELECT) if plain else None
                rank = {id(r): n for n, r in enumerate(self.ranked(step), 1)}
                print('\n%s' % title)
                print('%4s %-26s %6s | %-19s | %-19s | %s' % ('rank', 'behaviors', 'epoch', 'validation', 'test', 'validation ' + SELECT))
                print('%4s %-26s %6s | %9s %9s | %9s %9s | %s' % ('', '', '', SELECT, OTHER, SELECT, OTHER, 'against plain BIPN'))
                for row in rows:
                    percent, result = row['percent'], row['result']
                    v = [metric(result, part, key) for part, key in KEYS]
                    gain = '' if base is None or percent is None else '%+.4f' % (v[0] - base)
                    candidate = step == 'sample' and percent in self.candidates
                    best = step == 'all' and percent is not None and percent == self.best
                    note = '(baseline)' if percent is None else gain + ('  <- candidate' if candidate else '') + ('  <- best' if best else '')
                    print('%4s %-26s %6s | %9.4f %9.4f | %9.4f %9.4f | %s'
                          % (rank.get(id(row), ''), self.label(percent), result.get('best_epoch', '?'), v[0], v[1], v[2], v[3], note))
                    writer.writerow([step, rank.get(id(row), ''), self.label(percent), '' if percent is None else percent,
                                     result.get('best_epoch', '')] + ['%.4f' % x for x in v]
                                    + [gain, 'yes' if candidate else '', 'yes' if best else ''])
        print('%d runs finished. Table: %s' % (len(self.rows), self.results_csv), flush=True)

    def summary(self, step):
        """What is kept in <column>_search_best.json for one step."""
        plain = next((r for r in self.of_step(step) if r['percent'] is None), None)
        return {'plain_bipn': plain['result'] if plain else None,
                'cutoffs': {str(r['percent']): {k: r['result'].get(k) for k in ('best_epoch', 'validation', 'test')}
                            for r in self.of_step(step, with_plain=False)}}


def main():
    parser = argparse.ArgumentParser(description='Search the cut-off of a derived behavior: sample first, then the best ones on all users.')
    parser.add_argument('--column', default='recently_read', choices=sorted(COLUMNS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', default='0.001')
    parser.add_argument('--log_reg', default='0.3')
    parser.add_argument('--reg_weight', default='0.001')
    parser.add_argument('--batch_size', default='512')
    parser.add_argument('--save_minutes', default='30')
    parser.add_argument('--sample', type=float, default=0.1, help='share of users of step 1, e.g. 0.1; 1 = no step 1, every cut-off on all users')
    parser.add_argument('--top', type=int, default=3, help='how many cut-offs go on to step 2')
    parser.add_argument('--files_json', help='json file listing the three CSVs (LightGCN/cache/last_files.json)')
    args = parser.parse_args()
    if not 0 < args.sample <= 1:
        sys.exit('ERROR: --sample must be above 0 and at most 1')
    if not os.path.exists(DATA_PROCESS):
        sys.exit('ERROR: %s is missing. Start this with search_recently_read.bat.' % DATA_PROCESS)
    os.makedirs(CHECK_POINT, exist_ok=True)

    search = Search(args)
    percents = COLUMNS[args.column]['percents']
    two_steps = args.sample < 1.0 and not COLUMNS[args.column].get('all_users')
    if two_steps:
        print('%s cut-off search, device %s' % (args.column, args.device))
        print('step 1: plain BIPN and %d cut-offs on %g%% of the users -> the best %d are the candidates'
              % (len(percents), 100 * args.sample, args.top))
        print('step 2: plain BIPN and the candidates on all users -> the best one is kept', flush=True)
        failed = search.run_step('sample', args.sample, percents)
        search.candidates = sorted(r['percent'] for r in search.ranked('sample')[:args.top])
        if not search.candidates:
            sys.exit('\nNo run with %s finished in step 1. Fix the error shown above and start this again.' % args.column)
        search.show()
        print('\nCandidates for step 2 (the best %d of step 1): %s'
              % (len(search.candidates), ', '.join('%d%s' % (p, search.unit) for p in search.candidates)), flush=True)
    else:
        print('%s cut-off search: plain BIPN and %d cut-offs on all users, device %s'
              % (args.column, len(percents), args.device), flush=True)
        failed = []
        search.candidates = list(percents)
    failed += search.run_step('all', 1.0, search.candidates)

    ranked = search.ranked('all')
    plain = next((r for r in search.of_step('all') if r['percent'] is None), None)
    if ranked:
        search.best = ranked[0]['percent']
    search.show()
    print('\n============================================================')
    if not ranked:
        print(' No run with %s finished on all users yet.' % args.column)
    else:
        best = ranked[0]['result']
        value = metric(best, 'validation', SELECT)
        print(' Kept cut-off for %s: %s%d%s  (best on all users of the cut-offs %s)'
              % (args.column, search.sign, search.best, search.unit,
                 ', '.join('%s%d%s' % (search.sign, p, search.unit) for p in search.candidates)))
        if plain:
            base = metric(plain['result'], 'validation', SELECT)
            print(' validation %s %.4f against %.4f for plain BIPN (%+.4f)' % (SELECT, value, base, value - base))
            if value <= base:
                print(' %s did not improve on plain BIPN at any candidate.' % args.column)
        print(' test %s %.4f, %s %.4f' % (SELECT, metric(best, 'test', SELECT), OTHER, metric(best, 'test', OTHER)))
        if not failed:
            with open(search.best_json, 'w', encoding='utf-8') as f:
                json.dump({'column': args.column, 'kept_cutoff_percent': search.best, 'chosen_by': 'validation ' + SELECT,
                           'candidates_percent': search.candidates, 'tried_percent': percents,
                           'sample_share': args.sample if two_steps else None,
                           'improves_on_plain_bipn': bool(plain) and value > metric(plain['result'], 'validation', SELECT),
                           'all_users': search.summary('all'), 'sample': search.summary('sample') if two_steps else None,
                           'files': search.files}, f, indent=1)
            print(' Remembered in %s' % search.best_json)
    print('============================================================')
    if failed:
        print('\n%d run(s) did not finish: %s' % (len(failed), '; '.join(failed)))
        sys.exit(1)


if __name__ == '__main__':
    main()
