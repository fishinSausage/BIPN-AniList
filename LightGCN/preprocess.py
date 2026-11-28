#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File pickers and the CSV preprocessor of the LightGCN baseline.

Three CSVs are used: train, validation and test. Only these columns are read:

    user_id      the AniList user
    media_id     the AniList title
    scaled_like  train csv only:        1 = like, an edge of LightGCN's graph
    eval_like    validation / test csv: 1 = a title the model is expected to recommend

How the data is used (the same rules as BIPN's converter, BIPN/data/AniList/data_process.py):
  * training: the (user, title) pairs of the train csv with scaled_like = 1
  * candidates: every title that appears in the train csv; for each user, the titles the
    user has any row for in the train csv are left out of the ranking
  * answers: the pairs of the validation / test csv with eval_like = 1. scaled_like of
    those two files is not used, and nothing else of them is read.

The result is saved in the cache folder next to this file, so the CSVs are read only once.
"""
import collections
import csv
import hashlib
import json
import os
import sys
import time
from array import array

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
LAST_FILES = os.path.join(CACHE, 'last_files.json')

USER_COL, ITEM_COL = 'user_id', 'media_id'
TRAIN_LIKE_COL = 'scaled_like'   # train csv
EVAL_LIKE_COL = 'eval_like'      # validation and test csv
PARTS = ('train', 'validation', 'test')
LIKE = {'1', '1.0', 'true'}                   # values that mean "like"
NOT_LIKE = {'0', '0.0', 'false', '', 'nan'}   # values that mean "not a like"


def pick_files(given):
    """Returns {'train': path, 'validation': path, 'test': path}, asking with Explorer windows for the ones not given."""
    last = {}
    if os.path.exists(LAST_FILES):
        try:
            with open(LAST_FILES, encoding='utf-8') as f:
                last = json.load(f)
        except (OSError, ValueError):
            last = {}
    paths = {part: given.get(part) for part in PARTS}
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
            sys.exit('No %s csv was selected, so nothing was started.' % part)
        paths[part] = os.path.abspath(paths[part])
        if not os.path.isfile(paths[part]):
            sys.exit('ERROR: %s was not found.' % paths[part])
        folder = os.path.dirname(paths[part])
        print('      %-11s %s' % (part + ':', paths[part]), flush=True)
    if root is not None:
        root.destroy()
    if len({os.path.normcase(p) for p in paths.values()}) < len(PARTS):
        sys.exit('ERROR: the same file was selected more than once.')
    os.makedirs(CACHE, exist_ok=True)
    with open(LAST_FILES, 'w', encoding='utf-8') as f:
        json.dump(paths, f, indent=1)
    return paths


def to_int(text):
    try:
        return int(text)
    except ValueError:
        number = float(text)   # ids written as 39.0
        if number != int(number):
            raise
        return int(number)


def read_csv(path, like_col, every_row):
    """Reads one CSV. Returns the (user ids, title ids) of its rows with like_col = 1 and, if
    every_row is set, also the (user ids, title ids) of all its rows."""
    csv.field_size_limit(2 ** 31 - 1)
    like_u, like_i, all_u, all_i = array('q'), array('q'), array('q'), array('q')
    rows = short = 0
    other = collections.Counter()
    start = time.time()
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader, [])]
        missing = [name for name in (USER_COL, ITEM_COL, like_col) if name not in header]
        if missing:
            sys.exit('ERROR: %s has no column named %s.\nIts columns are: %s'
                     % (path, ' or '.join(missing), ', '.join(header) or '(none)'))
        iu, ii, il = header.index(USER_COL), header.index(ITEM_COL), header.index(like_col)
        need = max(iu, ii, il) + 1
        for row in reader:
            rows += 1
            if rows % 1000000 == 0:
                print('      %d million rows read ...' % (rows // 1000000), flush=True)
            if len(row) < need:
                short += 1
                continue
            value = row[il].strip().lower()
            like = value in LIKE
            if not like and value not in NOT_LIKE:
                if len(other) < 1000 or value in other:
                    other[value] += 1
                continue
            if like or every_row:
                try:
                    user, item = to_int(row[iu]), to_int(row[ii])
                except ValueError:
                    sys.exit('ERROR: row %d of %s has a %s or %s that is not a whole number: %r, %r'
                             % (rows + 1, path, USER_COL, ITEM_COL, row[iu], row[ii]))
                if every_row:
                    all_u.append(user)
                    all_i.append(item)
                if like:
                    like_u.append(user)
                    like_i.append(item)
    if other:
        examples = ', '.join('%r in %d rows' % pair for pair in other.most_common(5))
        sys.exit('ERROR: the %s column of %s must hold 1 or 0, but it also holds: %s' % (like_col, path, examples))
    if short:
        sys.exit('ERROR: %d rows of %s have fewer columns than its header line.' % (short, path))
    print('      %s: %d rows, %d with %s = 1 (%.0f s)'
          % (os.path.basename(path), rows, len(like_u), like_col, time.time() - start), flush=True)
    as_array = lambda a: np.array(a, dtype=np.int64)
    return as_array(like_u), as_array(like_i), as_array(all_u), as_array(all_i)


def file_key(path):
    info = os.stat(path)
    return [os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns]


def number(ids, values):
    """Position of every value in the sorted array ids, and whether the value is in ids at all."""
    place = np.minimum(np.searchsorted(ids, values), len(ids) - 1)
    return place, ids[place] == values


def load(paths):
    """Returns the three CSVs as arrays of consecutive user and title numbers.

    Keys of the returned dict:
      train_u, train_i        the train likes (scaled_like = 1), no duplicates
      seen_u, seen_i          every (user, title) pair of the train csv: not a candidate for that user
      valid_u, valid_i        the answers of the validation csv (eval_like = 1)
      test_u, test_i          the answers of the test csv (eval_like = 1)
      user_ids, item_ids      the AniList id of every user number / title number
      n_users, n_items, key (changes when a CSV changes), files
    """
    key = hashlib.sha1(json.dumps([[file_key(paths[part]) for part in PARTS],
                                   USER_COL, ITEM_COL, TRAIN_LIKE_COL, EVAL_LIKE_COL, 'v2'])
                       .encode('utf-8')).hexdigest()[:16]
    cache_file = os.path.join(CACHE, 'data_%s.npz' % key)
    if os.path.exists(cache_file):
        print('      Using the preprocessed data saved earlier: %s' % cache_file)
        with np.load(cache_file) as saved:
            data = {name: saved[name] for name in saved.files}
    else:
        like_u, like_i, all_u, all_i = read_csv(paths['train'], TRAIN_LIKE_COL, every_row=True)
        if len(like_u) == 0:
            sys.exit('ERROR: the train csv has no row with %s = 1.' % TRAIN_LIKE_COL)
        user_ids, item_ids = np.unique(all_u), np.unique(all_i)
        n_items = len(item_ids)
        seen = np.unique(np.searchsorted(user_ids, all_u) * n_items + np.searchsorted(item_ids, all_i))
        train = np.unique(np.searchsorted(user_ids, like_u) * n_items + np.searchsorted(item_ids, like_i))
        data = {'train_u': train // n_items, 'train_i': train % n_items,
                'seen_u': seen // n_items, 'seen_i': seen % n_items, 'user_ids': user_ids, 'item_ids': item_ids}
        left_out = []
        for part, name in (('validation', 'valid'), ('test', 'test')):
            u, i, _, _ = read_csv(paths[part], EVAL_LIKE_COL, every_row=False)
            (u, user_ok), (i, item_ok) = number(user_ids, u), number(item_ids, i)
            pairs = np.unique(u[user_ok & item_ok] * n_items + i[user_ok & item_ok])
            in_train = np.isin(pairs, seen, assume_unique=True)
            data[name + '_u'], data[name + '_i'] = pairs[~in_train] // n_items, pairs[~in_train] % n_items
            left_out += [int((~user_ok).sum()), int((user_ok & ~item_ok).sum()), int(in_train.sum())]
            if len(data[name + '_u']) == 0:
                sys.exit('ERROR: the %s csv has no usable row with %s = 1.' % (part, EVAL_LIKE_COL))
        data['left_out'] = np.array(left_out)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache_file + '.tmp.npz', **data)
        os.replace(cache_file + '.tmp.npz', cache_file)
    data.update(n_users=len(data['user_ids']), n_items=len(data['item_ids']), key=key,
                files={part: os.path.abspath(paths[part]) for part in PARTS})
    describe(data)
    return data


def describe(data):
    liked = np.zeros(data['n_users'], dtype=bool)
    liked[data['train_u']] = True
    print('      users %d | titles %d (all users and titles of the train csv)' % (data['n_users'], data['n_items']))
    print('      train: %d rows, %d likes (%s = 1) by %d users'
          % (len(data['seen_u']), len(data['train_u']), TRAIN_LIKE_COL, int(liked.sum())))
    for index, (part, name) in enumerate((('validation', 'valid'), ('test', 'test'))):
        users = np.unique(data[name + '_u'])
        no_user, no_title, in_train = (int(x) for x in data['left_out'][3 * index:3 * index + 3])
        print('      %s: %d answers (%s = 1) of %d users, %d of them without a train like'
              % (part, len(data[name + '_u']), EVAL_LIKE_COL, len(users), int((~liked[users]).sum())))
        if no_user or no_title or in_train:
            print('        left out: %d answers of users not in train, %d on titles not in train, %d that are train rows'
                  % (no_user, no_title, in_train))
    sys.stdout.flush()
