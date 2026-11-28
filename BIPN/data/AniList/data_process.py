#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turns the train, validation and test CSV into the files BIPN trains on.

    python data_process.py                       # asks for the three CSVs with Explorer windows
    python data_process.py --train a.csv --validation b.csv --test c.csv
    python data_process.py --sample 0.1 --out ../AniList_sample    # 10% of users, same CSVs
    python data_process.py --behaviors like --out ../AniList_likes  # likes only: the plain BIPN baseline
    python data_process.py --behaviors like --extra recently_read --cutoff 0.3 --out ../AniList_recently_read_30

The CSVs are only read, never changed.

Behaviors, made from the train csv only (each row goes into its status list, plus one
rating list if it is a like or has a score):
    planning, current, paused, completed, repeating, dropped   <- status column
    like      scaled_like = 1
    dislike   has a score, not a like, score_100 <= 40
    neutral   has a score, not a like, score_100 > 40
"like" is the target behavior. With --behaviors like it is the only behavior: BIPN then
learns from exactly the same likes as the LightGCN baseline.

Derived behaviors (--extra NAME --cutoff F): one more behavior, placed before "like", that is
not a column of the CSVs but is worked out from the train csv with one number, the cut-off.
They are listed in DERIVED below; to add one, write its function there. So far:
    recently_read   for every user, the latest share (the cut-off) of the user's train rows
                    by updated_at. updated_at is the time the list entry was last edited,
                    which is not always the time the title was read. Rows without an
                    updated_at are never recently_read.
codes/column_search.py searches the cut-off of a derived behavior.

Evaluation, the same rules as the LightGCN baseline (LightGCN/preprocess.py):
  * candidates: every title that appears in the train csv; for each user, the titles the
    user has any row for in the train csv are left out of the ranking (seen_dict.txt)
  * answers: the (user, title) pairs of the validation / test csv with eval_like = 1.
    scaled_like of those two files is not used, and nothing else of them is read.
  * every user with at least one answer is scored

User and title numbers are renumbered 1..N; user_ids.csv and title_ids.csv map them back
to the AniList ids.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys
import time

STATUS_BEHAVIOR = {
    'PLANNING': 'planning',
    'CURRENT': 'current',
    'PAUSED': 'paused',
    'COMPLETED': 'completed',
    'REPEATING': 'repeating',
    'DROPPED': 'dropped',
}
# The last one is the target. main.py reads the list a conversion used from its behaviors.json.
BEHAVIORS = ['planning', 'current', 'paused', 'completed', 'repeating', 'dropped', 'dislike', 'neutral', 'like']
TARGET = 'like'
BEHAVIOR_SETS = {'all': BEHAVIORS, 'like': [TARGET]}
TRAIN_LIKE_COL = 'scaled_like'   # train csv: 1 = like
EVAL_LIKE_COL = 'eval_like'      # validation and test csv: 1 = an answer
DISLIKE_MAX = 40                 # dislike: has a score, not a like, score_100 <= 40
LIKE = {'1', '1.0', 'true'}                   # values that mean "like"
NOT_LIKE = {'0', '0.0', 'false', '', 'nan'}   # values that mean "not a like"
PARTS = ('train', 'validation', 'test')
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = os.path.join(HERE, 'source_files.json')   # the CSVs of the last conversion


def recently_read(rows, cutoff):
    """The latest share (cutoff, 0..1) of one user's train rows by updated_at; equal times in title order."""
    dated = sorted((t for t in rows if rows[t][2] > 0), key=lambda t: (-rows[t][2], t))
    return dated[:int(math.ceil(cutoff * len(rows)))]


# Derived behaviors: name -> function(rows, cutoff) that returns the titles of one user that get the
# behavior. rows is {title: (status behavior or None, rating behavior or None, updated_at)} of the user's
# train rows; cutoff is the one number that codes/column_search.py searches. Only the train csv may be
# used, never validation or test.
DERIVED = {
    'recently_read': recently_read,
}


def listed_files(path):
    """The CSV paths saved in a json file ({'train': ..., 'validation': ..., 'test': ...}), or {} if not usable."""
    try:
        with open(path, encoding='utf-8') as f:
            listed = json.load(f)
    except (OSError, ValueError, TypeError):
        return {}
    if isinstance(listed, dict) and all(os.path.isfile(str(listed.get(part, ''))) for part in PARTS):
        return {part: listed[part] for part in PARTS}
    return {}


def pick_files(given, reuse, files_json=None):
    """Returns {'train': path, 'validation': path, 'test': path}.

    Order: the paths given on the command line; the CSVs listed in files_json (the ones chosen
    for the LightGCN grid search); with reuse, the CSVs of the last full conversion. Whatever is
    still missing is asked for with Explorer windows.
    """
    paths = {part: given.get(part) for part in PARTS}
    if not any(paths.values()):
        for source, text in ((files_json, 'Using the CSVs chosen for the LightGCN grid search:'),
                             (SOURCE_FILES if reuse else None, 'Using the same CSVs as the last conversion:')):
            listed = listed_files(source) if source else {}
            if listed:
                print(text)
                for part in PARTS:
                    print('      %-11s %s' % (part + ':', listed[part]))
                return listed
    last = listed_files(SOURCE_FILES)
    root = None
    if not all(paths.values()):
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes('-topmost', True)   # keeps the Explorer window in front of the console
        except Exception:
            root = None
            print('The file window could not be opened, so type the paths instead.')
    folder = os.path.dirname(last.get('train', '')) or os.path.expanduser('~')
    for part in PARTS:
        if not paths[part]:
            print('Select the %s csv ...' % part.upper(), flush=True)
            if root is not None:
                paths[part] = filedialog.askopenfilename(
                    parent=root, title='Select the %s csv' % part.upper(), initialdir=folder,
                    filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
            else:
                paths[part] = input('Path of the %s csv: ' % part).strip().strip('"')
        if not paths[part]:
            sys.exit('No %s csv was selected, so nothing was converted.' % part)
        paths[part] = os.path.abspath(paths[part])
        if not os.path.isfile(paths[part]):
            sys.exit('ERROR: %s was not found.' % paths[part])
        folder = os.path.dirname(paths[part])
        print('      %-11s %s' % (part + ':', paths[part]), flush=True)
    if root is not None:
        root.destroy()
    if len({os.path.normcase(p) for p in paths.values()}) < len(PARTS):
        sys.exit('ERROR: the same file was selected more than once.')
    return paths


def to_int(text):
    try:
        return int(text)
    except ValueError:
        number = float(text)   # ids written as 39.0
        if number != int(number):
            raise
        return int(number)


def open_csv(path, needed):
    """Opens a CSV and returns (file, reader, positions of the needed columns, position of media_title or None)."""
    csv.field_size_limit(2 ** 31 - 1)
    f = open(path, 'r', encoding='utf-8-sig', errors='replace', newline='')
    reader = csv.reader(f)
    header = [h.strip() for h in next(reader, [])]
    col = {h: i for i, h in enumerate(header)}
    missing = [c for c in needed if c not in col]
    if missing:
        f.close()
        sys.exit('ERROR: %s has no column named %s.\nIts columns are: %s'
                 % (path, ' or '.join(missing), ', '.join(header) or '(none)'))
    return f, reader, [col[c] for c in needed], col.get('media_title')


def like_value(text, other):
    """True for a like, False for not a like; any other value is counted in other."""
    value = text.strip().lower()
    if value in LIKE:
        return True
    if value not in NOT_LIKE and (len(other) < 1000 or value in other):
        other[value] += 1
    return False


def check(path, column, other, short):
    if other:
        examples = ', '.join('%r in %d rows' % pair for pair in other.most_common(5))
        sys.exit('ERROR: the %s column of %s must hold 1 or 0, but it also holds: %s' % (column, path, examples))
    if short:
        sys.exit('ERROR: %d rows of %s have fewer columns than its header line.' % (short, path))


def read_train(path):
    """Returns {user: {title: (status behavior or None, rating behavior or None, updated_at)}}, titles, notes.

    Every row of the train csv is kept, also one without a behavior: the user has seen that title.
    """
    users, titles = {}, {}
    notes = {'rows': 0, 'unknown_status': 0, 'repeated_pairs': 0}
    other, short, t0 = collections.Counter(), 0, time.time()
    needed = ['user_id', 'media_id', 'status', 'score_100', 'score_available', 'updated_at', TRAIN_LIKE_COL]
    f, reader, (iu, im, ist, isc, ifl, iup, ilk), ititle = open_csv(path, needed)
    need = max(iu, im, ist, isc, ifl, iup, ilk) + 1
    with f:
        for row in reader:
            notes['rows'] += 1
            if notes['rows'] % 1000000 == 0:
                print('  ... %d million rows read, %ds' % (notes['rows'] // 1000000, time.time() - t0), flush=True)
            if len(row) < need:
                short += 1
                continue
            try:
                user, title = to_int(row[iu]), to_int(row[im])
            except ValueError:
                sys.exit('ERROR: row %d of %s has a user_id or media_id that is not a whole number: %r, %r'
                         % (notes['rows'] + 1, path, row[iu], row[im]))
            status = STATUS_BEHAVIOR.get(row[ist].strip().upper())
            if status is None:
                notes['unknown_status'] += 1
            rating = None
            if like_value(row[ilk], other):
                rating = TARGET
            elif row[ifl].strip().lower() in ('1', '1.0', 'true'):
                try:
                    score = float(row[isc])
                except ValueError:
                    score = 0.0
                if score > 0:
                    rating = 'dislike' if score <= DISLIKE_MAX else 'neutral'
            try:
                updated = int(float(row[iup]))
            except ValueError:
                updated = 0
            entries = users.setdefault(user, {})
            if title in entries:                       # the same pair twice: the newer row counts,
                notes['repeated_pairs'] += 1           # and it is a like if either row is
                old = entries[title]
                if old[2] >= updated:
                    status, updated = old[0], old[2]
                    rating = TARGET if TARGET in (rating, old[1]) else old[1]
                elif old[1] == TARGET:
                    rating = TARGET
            entries[title] = (status, rating, updated)
            if title not in titles:
                titles[title] = row[ititle].strip() if ititle is not None else ''
    check(path, TRAIN_LIKE_COL, other, short)
    return users, titles, notes


def read_answers(path):
    """Returns the (user, title) pairs of a validation or test csv with eval_like = 1."""
    pairs, rows = set(), 0
    other, short, t0 = collections.Counter(), 0, time.time()
    f, reader, (iu, im, ilk), _ = open_csv(path, ['user_id', 'media_id', EVAL_LIKE_COL])
    need = max(iu, im, ilk) + 1
    with f:
        for row in reader:
            rows += 1
            if rows % 1000000 == 0:
                print('  ... %d million rows read, %ds' % (rows // 1000000, time.time() - t0), flush=True)
            if len(row) < need:
                short += 1
                continue
            if like_value(row[ilk], other):
                try:
                    pairs.add((to_int(row[iu]), to_int(row[im])))
                except ValueError:
                    sys.exit('ERROR: row %d of %s has a user_id or media_id that is not a whole number: %r, %r'
                             % (rows + 1, path, row[iu], row[im]))
    check(path, EVAL_LIKE_COL, other, short)
    return pairs, rows


def file_key(path):
    info = os.stat(path)
    return [os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns]


def convert(paths, out_dir, sample, seed, behaviors, extra=None, cutoff=None):
    t0 = time.time()
    if extra:
        behaviors = behaviors[:-1] + [extra, behaviors[-1]]   # the target stays the last one
    else:
        cutoff = None
    data_key = hashlib.sha1(json.dumps([[file_key(paths[part]) for part in PARTS], sample, seed, behaviors,
                                        cutoff,
                                        TRAIN_LIKE_COL, EVAL_LIKE_COL, DISLIKE_MAX]).encode('utf-8')).hexdigest()[:12]
    key_file = os.path.join(out_dir, 'data_key.txt')
    if os.path.exists(os.path.join(out_dir, 'split_report.txt')) and os.path.exists(key_file):
        with open(key_file) as f:
            if f.read().strip() == data_key:
                print('Already converted from these CSVs: %s' % out_dir, flush=True)
                return
    print('Reading %s' % paths['train'], flush=True)
    users, titles, notes = read_train(paths['train'])
    report = []

    def say(line=''):
        print(line, flush=True)
        report.append(line)

    rng = random.Random(seed)
    user_ids = sorted(users)
    if sample < 1.0:
        user_ids = sorted(rng.sample(user_ids, max(1, int(round(len(user_ids) * sample)))))
    title_ids = sorted({t for u in user_ids for t in users[u]})
    new_user = {u: i + 1 for i, u in enumerate(user_ids)}
    new_title = {t: i + 1 for i, t in enumerate(title_ids)}

    # ---- the answers: validation / test rows with eval_like = 1 ----------------------
    answers, answer_notes = {}, {}
    for part in ('validation', 'test'):
        print('Reading %s' % paths[part], flush=True)
        pairs, rows = read_answers(paths[part])
        kept, no_user, no_title, in_train = {}, 0, 0, 0
        for u, t in sorted(pairs):
            if u not in users:
                no_user += 1
            elif u not in new_user:
                continue                                   # a user outside the sample
            elif t not in new_title:
                no_title += 1
            elif t in users[u]:
                in_train += 1
            else:
                kept.setdefault(new_user[u], []).append(new_title[t])
        answers[part] = kept
        answer_notes[part] = (rows, len(pairs), no_user, no_title, in_train)

    # ---- the training lists -----------------------------------------------------------
    lists = {b: {} for b in behaviors}                     # behavior -> {new user: [new titles]}
    seen = {}                                              # new user -> every title of the user's train rows
    for u in user_ids:
        nu = new_user[u]
        seen[nu] = [new_title[t] for t in sorted(users[u])]
        if extra:
            chosen = DERIVED[extra](users[u], cutoff)
            if chosen:
                lists[extra][nu] = sorted(new_title[t] for t in chosen)
        for t in sorted(users[u]):
            status, rating, _ = users[u][t]
            for b in (status, rating):
                if b in lists:
                    lists[b].setdefault(nu, []).append(new_title[t])

    # ---- write ---------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    def out(name):
        if name.lower().endswith('.csv') and name not in ('user_ids.csv', 'title_ids.csv'):
            raise RuntimeError('refusing to write to %s' % name)
        return os.path.join(out_dir, name)

    for stale in ('split_report.txt', 'convert_report.txt'):   # marks of a finished (older) conversion
        if os.path.exists(out(stale)):
            os.remove(out(stale))
    all_items = {}
    for b in behaviors:
        with open(out(b + '.txt'), 'w') as f:
            for nu in sorted(lists[b]):
                for nt in lists[b][nu]:
                    f.write('%d %d\n' % (nu, nt))
                all_items.setdefault(nu, set()).update(lists[b][nu])
        with open(out(b + '_dict.txt'), 'w', encoding='utf-8') as f:
            json.dump({str(nu): lists[b][nu] for nu in sorted(lists[b])}, f)
    with open(out('all_dict.txt'), 'w', encoding='utf-8') as f:
        json.dump({str(nu): sorted(all_items[nu]) for nu in sorted(all_items)}, f)
    with open(out('seen_dict.txt'), 'w', encoding='utf-8') as f:
        json.dump({str(nu): seen[nu] for nu in sorted(seen)}, f)
    with open(out('pos_sampling.txt'), 'w') as f:
        for index, b in enumerate(behaviors):
            for nu in sorted(lists[b]):
                for nt in lists[b][nu]:
                    f.write('%d %d %d 1\n' % (nu, nt, index))
    for part in ('validation', 'test'):
        with open(out(part + '.txt'), 'w') as f:
            for nu in sorted(answers[part]):
                for nt in answers[part][nu]:
                    f.write('%d %d\n' % (nu, nt))
        with open(out(part + '_dict.txt'), 'w', encoding='utf-8') as f:
            json.dump({str(nu): answers[part][nu] for nu in sorted(answers[part])}, f)
    with open(out('count.txt'), 'w') as f:
        json.dump({'user': len(user_ids), 'item': len(title_ids)}, f)
    with open(out('behaviors.json'), 'w') as f:
        json.dump(behaviors, f)
    with open(out('user_ids.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'anilist_user_id'])
        w.writerows((new_user[u], u) for u in user_ids)
    with open(out('title_ids.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['title', 'anilist_media_id', 'media_title'])
        w.writerows((new_title[t], t, titles[t]) for t in title_ids)
    with open(out('data_key.txt'), 'w') as f:
        f.write(data_key + '\n')

    # ---- report --------------------------------------------------------------------
    say('AniList -> BIPN conversion (data key %s)' % data_key)
    for part in PARTS:
        say('%-11s %s' % (part + ':', paths[part]))
    say('train rows %d | rows with an unknown status %d | repeated user-title pairs %d'
        % (notes['rows'], notes['unknown_status'], notes['repeated_pairs']))
    if sample < 1.0:
        say('SAMPLE: %d of %d users (%.0f%%), seed %d' % (len(user_ids), len(users), 100 * sample, seed))
    say('users %d | titles %d (all users and titles of the train csv)' % (len(user_ids), len(title_ids)))
    say('train rows of these users: %d' % sum(len(v) for v in seen.values()))
    say()
    say('%-14s %12s' % ('behavior', 'training'))
    for b in behaviors:
        note = ''
        if b == TARGET:
            note = '   <- target (%s = 1)' % TRAIN_LIKE_COL
        elif b == extra:
            note = '   <- derived from the train csv, cut-off %g' % cutoff
        say('%-14s %12d%s' % (b, sum(len(v) for v in lists[b].values()), note))
    say('%-14s %12d' % ('TOTAL', sum(len(v) for b in behaviors for v in lists[b].values())))
    say()
    liked = set(lists[TARGET])
    for part in ('validation', 'test'):
        rows, found, no_user, no_title, in_train = answer_notes[part]
        kept = answers[part]
        say('%s: %d answers (%s = 1) of %d users, %d of them without a train like'
            % (part, sum(len(v) for v in kept.values()), EVAL_LIKE_COL, len(kept), sum(1 for nu in kept if nu not in liked)))
        if no_user or no_title or in_train:
            say('  left out: %d answers of users not in train, %d on titles not in train, %d that are train rows'
                % (no_user, no_title, in_train))
    say('written to %s in %ds' % (out_dir, time.time() - t0))
    with open(out('split_report.txt'), 'w', encoding='utf-8') as f:     # written last: marks a finished conversion
        f.write('\n'.join(report) + '\n')
    with open(SOURCE_FILES, 'w', encoding='utf-8') as f:                # later conversions can use the same CSVs
        json.dump(paths, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the AniList train, validation and test CSV for BIPN.')
    for name in PARTS:
        parser.add_argument('--' + name, help='%s CSV; an Explorer window asks for it when left out' % name)
    parser.add_argument('--files_json', help='json file listing the three CSVs (LightGCN/cache/last_files.json); '
                                             'used when the CSVs are not given, so that both models read the same files')
    parser.add_argument('--out', default=HERE, help='folder for the converted files')
    parser.add_argument('--behaviors', choices=sorted(BEHAVIOR_SETS), default='all',
                        help='all = 6 status + 3 rating behaviors; like = likes only (the plain BIPN baseline)')
    parser.add_argument('--extra', choices=sorted(DERIVED),
                        help='one more behavior, derived from the train csv with --cutoff')
    parser.add_argument('--cutoff', type=float, help='with --extra: its cut-off, above 0 and at most 1')
    parser.add_argument('--sample', type=float, default=1.0, help='fraction of users to keep, e.g. 0.1')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    if not 0 < args.sample <= 1:
        sys.exit('ERROR: --sample must be between 0 and 1')
    if args.extra and (args.cutoff is None or not 0 < args.cutoff <= 1):
        sys.exit('ERROR: --extra needs --cutoff, above 0 and at most 1')
    files = pick_files({name: getattr(args, name) for name in PARTS}, reuse=args.sample < 1.0, files_json=args.files_json)
    convert(files, os.path.abspath(args.out), args.sample, args.seed, list(BEHAVIOR_SETS[args.behaviors]),
            args.extra, args.cutoff)
