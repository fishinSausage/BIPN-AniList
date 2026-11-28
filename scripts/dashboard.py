#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The screen of START.bat: runs the three stages one after another and keeps their results in sight.

    stage 1  LightGCN grid search                      (LightGCN/grid_search.py)
    stage 2  plain BIPN, likes only, one training run  (BIPN/data/AniList/data_process.py, BIPN/codes/main.py)
    stage 3  cut-off search of the derived behavior recently_read (BIPN/codes/column_search.py):
             every cut-off on a sample of the users, then the best few on all users
    stage 4  cut-off search of the derived behavior liked_author: five cut-offs, each on all users

The window is split in two. The top part never scrolls: one row per model that shows
"waiting", then "running" with its progress, then the result, in the same columns for every
model so that they can be compared at a glance. The bottom part shows what the running
stage prints.

All stages read the same train, validation and test csv (chosen once, in stage 1) and
are scored the same way. BIPN's settings, the sample share of stage 3 (SEARCH_SAMPLE) and
the number of cut-offs that go on to all users (SEARCH_TOP) are the ones at the top of
BIPN/scripts/setup_and_train_anilist.bat.

Safe to stop and start again: every stage skips what it has already finished.
"""
import argparse
import codecs
import collections
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata

SEARCH_COLUMNS = ['recently_read', 'liked_author']   # the derived behaviors that stages 3, 4, ... search, in this order

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIGHTGCN = os.path.join(ROOT, 'LightGCN')
BIPN = os.path.join(ROOT, 'BIPN')
SETUP_BAT = os.path.join(BIPN, 'scripts', 'setup_and_train_anilist.bat')
LAST_FILES = os.path.join(LIGHTGCN, 'cache', 'last_files.json')          # the CSVs chosen in stage 1
LIGHTGCN_RESULT = os.path.join(LIGHTGCN, 'results', 'best_result.json')
BIPN_LIKES_DATA = os.path.join(BIPN, 'data', 'AniList_likes')
BIPN_LIKES_RESULT = os.path.join(BIPN, 'check_point', 'anilist_likes_result.json')
COLUMN_SEARCH = os.path.join(BIPN, 'codes', 'column_search.py')

ROW = ' %-30s %-8s %-10s  %-8s %-10s  %s'      # label, validation ndcg, recall, test ndcg, recall, last column
STYLES = {'bold': '1', 'dim': '2', 'yellow': '33', 'green': '32', 'red': '31'}
ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
LOG_PREFIX = re.compile(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \| \w+\s*\| \S+ - ')   # time | level | place - of BIPN's log
BLOCKS = {code: '#' for code in range(0x2580, 0x25A0)}   # the blocks of a progress bar, as plain '#'


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def bat_settings():
    """The settings at the top of BIPN's start file."""
    found = {'LR': '0.001', 'BETA': '0.3', 'GAMMA': '0.001', 'BATCH_SIZE': '512', 'SAVE_MINUTES': '30',
             'SEARCH_SAMPLE': '0.1', 'SEARCH_TOP': '3'}
    try:
        with open(SETUP_BAT, encoding='ascii', errors='replace') as f:
            for line in f:
                match = re.match(r'set "(\w+)=([^"]*)"', line.strip())
                if match and match.group(1) in found:
                    found[match.group(1)] = match.group(2)
    except OSError:
        pass
    return found


def column_configs():
    """COLUMNS of BIPN/codes/column_search.py: the cut-offs of every derived column and how they are searched."""
    sys.path.insert(0, os.path.join(BIPN, 'codes'))
    try:
        import column_search
        return column_search.COLUMNS
    except Exception:
        return {}
    finally:
        sys.path.pop(0)


def search_csv(column):
    return os.path.join(BIPN, '%s_search_results.csv' % column)


def clock(seconds):
    seconds = int(seconds)
    return '%02d:%02d:%02d' % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def f4(value):
    return '%.4f' % float(value)


def metrics(result):
    """A result file's four numbers as text: validation ndcg, recall, test ndcg, recall."""
    return [f4(result[part][key]) for part in ('validation', 'test') for key in ('ndcg@20', 'recall@20')]


# ---- the screen ------------------------------------------------------------------------------
def fit(text, cells):
    """text cut so that it takes at most this many screen cells (Korean and other wide characters take two)."""
    used, kept = 0, []
    for char in text:
        size = 2 if unicodedata.east_asian_width(char) in ('W', 'F', 'A') else 1
        if used + size > cells:
            break
        kept.append(char)
        used += size
    return ''.join(kept)


def enable_vt():
    """True if the console understands cursor movement (Windows 10 and later, after switching it on)."""
    if not sys.stdout.isatty():
        return False
    if os.name != 'nt':
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class Screen:
    """A top part that stays in place and a bottom part that scrolls.

    Without a console that can do this (output sent to a file, very old Windows), everything
    is printed as plain lines, and the top part only when a stage starts or ends.
    """

    def __init__(self, top_wanted, forced_size=None):
        self.top_wanted, self.forced = top_wanted, forced_size
        self.vt = True if forced_size else enable_vt()
        self.out = sys.stdout
        self.history = collections.deque(maxlen=300)
        self.last_top, self.size, self.width = None, None, 116
        if self.vt:
            self.layout()

    def measure(self):
        if self.forced:
            return self.forced
        size = shutil.get_terminal_size((116, 42))
        return size.columns, size.lines

    def layout(self):
        """Clears the window and reserves the top part; called again when the window is resized."""
        self.size = self.measure()
        self.width, self.height = self.size
        self.top = max(1, min(self.top_wanted, self.height - 3))
        self.out.write('\x1b[r\x1b[2J\x1b[%d;%dr\x1b[%d;1H' % (self.top + 1, self.height, self.height))
        for text in list(self.history)[-(self.height - self.top - 1):]:
            self.out.write('\x1b[%d;1H\x1b[2K%s\n' % (self.height, fit(text, self.width - 1)))
        self.last_top = None
        self.out.flush()

    def draw_top(self, lines, plain_too=False):
        """lines: (text, style or None) for every row of the top part."""
        if not self.vt:
            if plain_too:
                self.out.write('\n' + '\n'.join(text for text, _ in lines) + '\n\n')
                self.out.flush()
            return
        if self.measure() != self.size:
            self.layout()
        if lines == self.last_top:
            return
        shown = lines[:self.top - 1] + lines[-1:] if len(lines) > self.top else lines
        parts = []
        for row, (text, style) in enumerate(shown, 1):
            text = fit(text, self.width - 1)
            if style:
                text = '\x1b[%sm%s\x1b[0m' % (STYLES[style], text)
            parts.append('\x1b[%d;1H\x1b[2K%s' % (row, text))
        self.out.write(''.join(parts) + '\x1b[%d;1H' % self.height)
        self.out.flush()
        self.last_top = lines

    def log(self, text):
        """A finished line of the running stage: goes to the bottom part and scrolls it."""
        self.history.append(text)
        if self.vt:
            self.out.write('\x1b[%d;1H\x1b[2K%s\n' % (self.height, fit(text, self.width - 1)))
        else:
            self.out.write(text + '\n')
        self.out.flush()

    def status(self, text):
        """A line that keeps being rewritten (a progress bar): shown on the last line without scrolling."""
        if self.vt:
            self.out.write('\x1b[%d;1H\x1b[2K%s' % (self.height, fit(text, self.width - 1)))
            self.out.flush()

    def wait_for_key(self):
        """Keeps the window open until a key is pressed (only when a person is at the keyboard)."""
        if self.forced or not (self.vt and sys.stdin and sys.stdin.isatty()):
            return
        self.log('Press any key to close this window.')
        try:
            import msvcrt
            msvcrt.getch()
        except ImportError:
            try:
                input()
            except EOFError:
                pass

    def close(self):
        """Gives the whole window back to whatever runs next."""
        if self.vt:
            self.out.write('\x1b[r\x1b[%d;1H' % self.height)
            self.out.flush()


# ---- what is known so far ----------------------------------------------------------------------
class State:
    def __init__(self, n_stages, configs, sample, top):
        self.started, self.n_stages, self.stage = time.time(), n_stages, 0
        self.sample, self.n_candidates = sample, top
        self.finished = self.failed = False
        self.files, self.counts = {}, {}
        # per stage: status = waiting / running / done / failed, text = what a running stage is doing
        self.lightgcn = {'status': 'waiting', 'text': '', 'result': None}
        self.bipn = {'status': 'waiting', 'text': '', 'result': None}
        # one search per derived column, stages 3, 4, ...: the rows of each step ({percent, or None for
        # plain BIPN: row}), the candidates, the kept cut-off and the run going on (step, percent)
        self.columns = [c for c in SEARCH_COLUMNS if c in configs][:max(0, n_stages - 2)]
        self.searches = {}
        for number, column in enumerate(self.columns, 3):
            config = configs[column]
            self.searches[column] = {
                'status': 'waiting', 'text': '', 'sample': {}, 'all': {}, 'candidates': [], 'best': None,
                'current': None, 'stage': number, 'percents': config['percents'], 'sign': config.get('sign', ''),
                'two_steps': sample < 1.0 and not config.get('all_users')}
        self.run = {'epoch': 0, 'best': None, 'best_epoch': 0}                   # the BIPN run going on


def build_top(state, width):
    """The rows of the top part, as (text, style). Always the same number of rows."""
    wide = max(60, width - 2)
    now = time.time() - state.started
    if state.failed:
        right = 'stopped at stage %d of %d - %s' % (state.stage, state.n_stages, clock(now))
    elif state.finished:
        right = 'all %d stages finished - %s' % (state.n_stages, clock(now))
    else:
        right = 'stage %d of %d - elapsed %s' % (max(state.stage, 1), state.n_stages, clock(now))
    title = ' AniList baselines'
    lines = [(title + ' ' * max(1, wide - len(title) - len(right)) + right, 'bold')]
    if state.files:
        info = ' ' + ' - '.join(os.path.basename(state.files[p]) for p in ('train', 'validation', 'test') if p in state.files)
        if state.counts.get('users'):
            info += '    {:,} users - {:,} titles'.format(state.counts['users'], state.counts['titles'])
        if state.counts.get('likes'):
            info += ' - {:,} train likes'.format(state.counts['likes'])
    else:
        info = ' CSVs: choose the train, validation and test csv in the windows that open'
    lines += [(info, None), (' ' + '-' * wide, 'dim')]
    over = state.finished or state.failed

    def text_row(label, text, style):
        return (' %-30s %s' % (label, text), style)

    def idle(label, text='waiting'):
        return text_row(label, 'not run' if over else text, 'dim')

    def stage_row(label, stage, extra):
        if stage['status'] == 'done' and stage['result']:
            return (ROW % tuple([label] + metrics(stage['result']) + [extra]), None)
        if stage['status'] == 'running':
            return text_row(label, 'running - %s' % (stage['text'] or 'starting'), 'yellow')
        if stage['status'] == 'failed':
            return text_row(label, 'FAILED - see the messages below', 'red')
        return idle(label)

    def search_row(search, label, step, percent, last, style=None):
        """A row of a cut-off search: its numbers when the run has finished, otherwise what it is doing."""
        row = search[step].get(percent)
        if row:
            return (ROW % tuple([label] + row['values'] + [last(row)]), style)
        if search['current'] == (step, percent) and search['status'] == 'running':
            return text_row(label, 'running - %s' % (search['text'] or 'starting'), 'yellow')
        if search['current'] == (step, percent) and search['status'] == 'failed':
            return text_row(label, 'FAILED - see the messages below', 'red')
        return idle(label)

    # ---- all users: LightGCN, plain BIPN and, from stage 3, the candidates -------------------
    lines.append((' %-30s %-19s  %-19s' % ('ALL USERS', 'validation', 'test'), 'bold'))
    lines.append((ROW % ('', 'NDCG@20', 'Recall@20', 'NDCG@20', 'Recall@20', 'best setting'), None))
    lg, bp = state.lightgcn, state.bipn
    extra = ''
    if lg['result']:
        best = lg['result'].get('best_hyperparameters', {})
        extra = 'lambda %g - %s layers - epoch %s' % (best.get('lambda', 0), best.get('layers', '?'), lg['result'].get('best_epoch', '?'))
    lines.append(stage_row('1 LightGCN  grid search', lg, extra))
    lines.append(stage_row('2 BIPN      likes only', bp, 'epoch %s' % (bp['result'] or {}).get('best_epoch', '?')))
    if lg['result'] and bp['result'] and lg['status'] == bp['status'] == 'done':
        diff = ['%+.4f' % (float(b) - float(a)) for a, b in zip(metrics(lg['result']), metrics(bp['result']))]
        lines.append((ROW % tuple(['             vs LightGCN'] + diff + ['']), 'dim'))
    else:
        lines.append(('', None))
    for column in state.columns:
        search = state.searches[column]
        name = '+ %s %s' % (column, search['sign'])
        shown = search['candidates'] if search['two_steps'] else search['percents']
        slots = state.n_candidates if search['two_steps'] else len(search['percents'])
        for slot in range(slots):
            head = '%d BIPN  ' % search['stage'] if slot == 0 else '        '
            if slot < len(shown):
                percent = shown[slot]
                is_best = search['best'] == percent
                lines.append(search_row(search, '%s%s%d%%' % (head, name, percent), 'all', percent,
                                        lambda row, b=is_best: row['gain'] + ' vs likes only' + ('  <- best' if b else ''),
                                        'green' if is_best else None))
            elif search['status'] == 'failed' and not shown and slot == 0:
                lines.append(text_row('%s%s' % (head, name), 'FAILED - see the messages below', 'red'))
            else:
                lines.append(idle('%s%s..%%' % (head, name), 'waiting - one of the best %d of the sample below' % slots))
    lines.append((' ' + '-' * wide, 'dim'))

    # ---- step 1 of a two-step search: every cut-off on a sample of the users ------------------
    for column in state.columns:
        search = state.searches[column]
        if not search['two_steps']:
            continue
        block = '%g%% USER SAMPLE' % (100 * state.sample)
        lines.append((' %-30s %-19s  %-19s  %s' % (block, 'validation', 'test',
                                                 'picks the %d that go on to all users' % state.n_candidates), 'bold'))
        lines.append((ROW % ('', 'NDCG@20', 'Recall@20', 'NDCG@20', 'Recall@20', 'vs likes only'), None))
        lines.append(search_row(search, '        likes only', 'sample', None, lambda row: 'baseline (same users)'))
        for percent in search['percents']:
            is_candidate = percent in search['candidates']
            lines.append(search_row(search, '        + %s %s%d%%' % (column, search['sign'], percent), 'sample', percent,
                                    lambda row, c=is_candidate: row['gain'] + ('  <- candidate' if c else ''),
                                    'green' if is_candidate else None))
    lines.append((' ' + '=' * wide, None))
    return lines


# ---- running a stage ---------------------------------------------------------------------------
def reader(pipe, events):
    """Turns what a stage prints into ('line', text) and ('status', text) events; ('end', None) when it stops."""
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    text, after_cr = '', False
    while True:
        chunk = os.read(pipe.fileno(), 4096)
        if not chunk:
            break
        for char in decoder.decode(chunk):
            if char == '\n':
                events.put(('line', text))
                text, after_cr = '', False
            elif char == '\r':
                after_cr = True
            else:
                if after_cr:                       # a carriage return without a line feed: a progress bar
                    events.put(('status', text))
                    text, after_cr = '', False
                text += char
    if text:
        events.put(('line', text))
    events.put(('end', None))


def run(command, cwd, screen, state, on_line):
    """Runs one step of a stage, feeding its output to the bottom part and to on_line. Returns its exit code."""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    events = queue.Queue()
    threading.Thread(target=reader, args=(process.stdout, events), daemon=True).start()
    while True:
        try:
            kind, text = events.get(timeout=0.5)
        except queue.Empty:
            kind, text = 'tick', None
        if kind == 'end':
            break
        if text is not None:
            text = LOG_PREFIX.sub('', ANSI.sub('', text)).replace('\t', '    ').translate(BLOCKS).rstrip()
        if kind == 'line':
            screen.log(text)
            on_line(text)
        elif kind == 'status' and text:
            screen.status(text)
        screen.draw_top(build_top(state, screen.width))
    return process.wait()


def bipn_progress(state, line):
    """Follows a BIPN training run through what main.py logs. Returns the text for a 'running' row."""
    match = re.search(r'epoch (\d+) [\d.]+s Train loss', line)
    if match:
        state.run['epoch'] = int(match.group(1))
    match = re.search(r"validate (\d+) cost time .*'ndcg@20': ([\d.]+)", line)
    if match and (state.run['best'] is None or float(match.group(2)) > state.run['best']):
        state.run['best'], state.run['best_epoch'] = float(match.group(2)), int(match.group(1))
    if 'Reading ' in line or 'million rows read' in line:
        return 'converting the CSVs'
    if state.run['best'] is not None:
        return 'epoch %d done - best NDCG@20 so far %.4f (epoch %d)' % (state.run['epoch'], state.run['best'], state.run['best_epoch'])
    if state.run['epoch']:
        return 'epoch %d done' % state.run['epoch']
    return None


def stage_lightgcn(args, screen, state, settings):
    stage = state.lightgcn
    stage.update(status='running', text='choose the three CSVs in the windows that open')
    progress = {'done': 0, 'total': 20, 'left': '', 'best': None}

    def describe():
        text = '%d of %d done' % (progress['done'], progress['total'])
        if progress['best']:   # taken from what the search prints; its result file is only read when it has ended
            text += ' - best NDCG@20 so far %s' % progress['best']
        return text + progress['left']

    def on_line(line):
        match = re.match(r'\s+(train|validation|test):\s+(.+\.csv)\s*$', line, re.I)
        if match:
            state.files[match.group(1).lower()] = match.group(2)
        match = re.match(r'\s+users (\d+) \| titles (\d+)', line)
        if match:
            state.counts.update(users=int(match.group(1)), titles=int(match.group(2)))
        match = re.match(r'\s+train: \d+ rows, (\d+) likes', line)
        if match:
            state.counts['likes'] = int(match.group(1))
        if line.startswith('[3/4]'):
            stage['text'] = 'reading the CSVs'
        match = re.match(r'\[4/4\] Grid search: (\d+) combinations', line)
        if match:
            progress['total'] = int(match.group(1))
            stage['text'] = describe()
        match = re.match(r'(\d+) combinations were already finished', line)
        if match:
            progress['done'] = int(match.group(1))
            stage['text'] = describe()
        if re.match(r'=== \d+ of \d+:', line):
            left = re.search(r'about ([\d.]+) hours left', line)
            progress['left'] = ' - about %s h left' % left.group(1) if left else ''
            stage['text'] = describe()
        match = re.match(r'best so far: .* validation ndcg@20 ([\d.]+)', line)
        if match:
            progress['best'] = match.group(1)
            stage['text'] = describe()
        match = re.match(r'(\d+) of (\d+) combinations finished', line)
        if match:
            progress['done'] = int(match.group(1))
            stage['text'] = describe()

    command = [sys.executable, '-u', os.path.join(LIGHTGCN, 'grid_search.py'), '--device', args.device]
    for part in ('train', 'validation', 'test'):
        if getattr(args, part):
            command += ['--' + part, getattr(args, part)]
    if args.max_epochs:
        command += ['--max_epochs', str(args.max_epochs)]
    code = run(command, LIGHTGCN, screen, state, on_line)
    result = read_json(LIGHTGCN_RESULT)
    if code != 0 or not result:
        return False
    stage['result'] = result
    return True


def files_argument():
    return ['--files_json', LAST_FILES] if os.path.exists(LAST_FILES) else []


def stage_bipn(args, screen, state, settings):
    stage = state.bipn
    stage.update(status='running', text='converting the CSVs')
    state.run = {'epoch': 0, 'best': None, 'best_epoch': 0}

    def on_line(line):
        text = bipn_progress(state, line)
        if text:
            stage['text'] = text
        elif 'Already converted' in line or 'written to' in line:
            stage['text'] = 'loading the data'

    data_process = os.path.join(BIPN, 'data', 'AniList', 'data_process.py')
    command = [sys.executable, '-u', data_process, '--behaviors', 'like', '--out', BIPN_LIKES_DATA] + files_argument()
    if run(command, BIPN, screen, state, on_line) != 0:
        return False
    stage['text'] = 'loading the data'
    os.makedirs(os.path.join(BIPN, 'check_point'), exist_ok=True)
    command = [sys.executable, '-u', os.path.join(BIPN, 'codes', 'main.py'), '--data_name', 'anilist_likes',
               '--device', args.device, '--lr', settings['LR'], '--log_reg', settings['BETA'],
               '--reg_weight', settings['GAMMA'], '--decay', '0', '--batch_size', settings['BATCH_SIZE'],
               '--model_name', 'anilist_likes', '--resume', '--save_minutes', settings['SAVE_MINUTES']]
    code = run(command, BIPN, screen, state, on_line)
    result = read_json(BIPN_LIKES_RESULT)
    try:
        with open(os.path.join(BIPN_LIKES_DATA, 'data_key.txt')) as f:
            key = f.read().strip()
    except OSError:
        key = None
    if code != 0 or not result or result.get('settings', {}).get('data_key') != key:
        return False
    stage['result'] = result
    return True


def read_search_table(search, column):
    """Fills the rows of a cut-off search from <column>_search_results.csv."""
    table = {'sample': {}, 'all': {}}
    candidates, best = [], None
    try:
        with open(search_csv(column), encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                percent = int(float(row['cutoff_percent'])) if row['cutoff_percent'] else None
                table[row['step']][percent] = {
                    'values': [f4(row[name]) for name in ('validation_ndcg@20', 'validation_recall@20',
                                                          'test_ndcg@20', 'test_recall@20')],
                    'gain': row.get('validation_ndcg@20_minus_plain', '')}
                if row.get('candidate') == 'yes':
                    candidates.append(percent)
                if row.get('best') == 'yes':
                    best = percent
    except (OSError, ValueError, KeyError):
        return
    search['sample'], search['all'] = table['sample'], table['all']
    if candidates:
        search['candidates'] = sorted(candidates)
    search['best'] = best


def search_stage(column):
    """The stage that searches the cut-off of one derived column."""
    def stage_search(args, screen, state, settings):
        stage = state.searches[column]
        stage.update(status='running', text='starting')

        def on_line(line):
            match = re.match(r'=== (sample|all users) (\d+) of (\d+): (.*) ===', line)
            if match:
                percent = re.search(r'\+ \w+ >?(\d+)%', match.group(4))
                stage['current'] = ('sample' if match.group(1) == 'sample' else 'all',
                                    int(percent.group(1)) if percent else None)
                stage['text'] = 'converting the CSVs'
                state.run = {'epoch': 0, 'best': None, 'best_epoch': 0}
            text = bipn_progress(state, line)
            if text:
                stage['text'] = text
            elif 'Already converted' in line or 'written to' in line:
                stage['text'] = 'loading the data'
            if re.match(r'\d+ runs finished', line) or line.startswith('Candidates for step 2'):
                read_search_table(stage, column)

        command = [sys.executable, '-u', COLUMN_SEARCH, '--column', column, '--device', args.device,
                   '--lr', settings['LR'], '--log_reg', settings['BETA'], '--reg_weight', settings['GAMMA'],
                   '--batch_size', settings['BATCH_SIZE'], '--save_minutes', settings['SAVE_MINUTES'],
                   '--sample', str(state.sample), '--top', str(state.n_candidates)] + files_argument()
        code = run(command, BIPN, screen, state, on_line)
        read_search_table(stage, column)
        if code == 0:
            stage['current'] = None
        return code == 0
    return stage_search


def stages_of(state):
    """(title, the stage's entry in the state, function) for every stage that is run."""
    stages = [('LightGCN grid search', state.lightgcn, stage_lightgcn),
              ('plain BIPN baseline (likes only)', state.bipn, stage_bipn)]
    stages += [('%s cut-off search' % column, state.searches[column], search_stage(column)) for column in state.columns]
    return stages[:state.n_stages]


def main():
    parser = argparse.ArgumentParser(description='Run the LightGCN grid search, plain BIPN and the cut-off searches '
                                                 'one after another, with a summary that stays in sight.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stages', type=int, default=2 + len(SEARCH_COLUMNS), choices=range(1, 3 + len(SEARCH_COLUMNS)),
                        help='how many of the stages to run')
    for part in ('train', 'validation', 'test'):
        parser.add_argument('--' + part, help='%s CSV; an Explorer window asks for it when left out' % part)
    parser.add_argument('--max_epochs', type=int, help='for a quick test: epoch limit of every LightGCN run')
    parser.add_argument('--sample', type=float, help='for a quick test: share of users of a sample step instead of SEARCH_SAMPLE')
    parser.add_argument('--size', help='for testing the layout: draw for a window of this size, e.g. 116x42')
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')   # a character the window cannot show must not stop the run
    except (AttributeError, ValueError):
        pass

    settings = bat_settings()
    sample = args.sample if args.sample else float(settings['SEARCH_SAMPLE'])
    state = State(args.stages, column_configs(), sample, int(settings['SEARCH_TOP']))
    stages = stages_of(state)
    forced = tuple(int(x) for x in args.size.lower().split('x')) if args.size else None
    screen = Screen(len(build_top(state, 116)), forced)

    def refresh(plain_too=False):
        screen.draw_top(build_top(state, screen.width), plain_too)

    try:
        refresh()
        for number, (title, entry, function) in enumerate(stages, 1):
            state.stage = number
            screen.log('')
            screen.log('######## stage %d of %d: %s ########' % (number, len(stages), title))
            refresh()
            ok = function(args, screen, state, settings)
            entry['status'] = 'done' if ok else 'failed'
            if not ok:
                state.failed = True
                refresh(plain_too=True)
                screen.log('')
                screen.log('Stage %d did not finish. The message above says why.' % number)
                screen.log('Fix it and start START.bat again: everything that has finished is skipped.')
                screen.wait_for_key()
                return 3
            refresh(plain_too=True)
        state.finished = True
        refresh(plain_too=True)
        screen.log('')
        screen.log('All %d stages finished. The results are kept in:' % len(stages))
        screen.log('  LightGCN\\results\\best_result.json, best_model.pth, grid_results.csv')
        if args.stages >= 2:
            screen.log('  BIPN\\check_point\\anilist_likes_result.json')
        for column in state.columns:
            screen.log('  BIPN\\%s_search_results.csv and %s_search_best.json' % (column, column))
        screen.wait_for_key()
        return 0
    finally:
        screen.close()


if __name__ == '__main__':
    sys.exit(main())
