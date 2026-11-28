#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File pickers and the CSV preprocessor of the LightGCN baseline.

Only three columns of the train and valid CSV are read:

    user_id      the AniList user
    media_id     the AniList title
    scaled_like  1 = the user liked the title, 0 or empty = not a like

Every (user, title) pair with scaled_like = 1 becomes one edge of LightGCN's graph. All
other rows and all other columns are ignored. The result is saved in the cache folder next
to this file, so the CSVs are read only once.
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

USER_COL, ITEM_COL, LIKE_COL = 'user_id', 'media_id', 'scaled_like'
LIKE = {'1', '1.0', 'true'}                   # values of scaled_like that mean "like"
NOT_LIKE = {'0', '0.0', 'false', '', 'nan'}   # values that mean "not a like"


def pick_files(train=None, valid=None):
    """Returns the paths of the train and the valid CSV, asking with Explorer windows for the ones not given."""
    last = {}
    if os.path.exists(LAST_FILES):
        try:
            with open(LAST_FILES, encoding='utf-8') as f:
                last = json.load(f)
        except (OSError, ValueError):
            last = {}
    paths = {'train': train, 'valid': valid}
    root = None
    if not (train and valid):
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
    for key in ('train', 'valid'):
        if not paths[key]:
            print('Select the %s csv ...' % key.upper(), flush=True)
            if root is not None:
                paths[key] = filedialog.askopenfilename(
                    parent=root, title='Select the %s csv' % key.upper(), initialdir=folder,
                    filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
            else:
                paths[key] = input('Path of the %s csv: ' % key).strip().strip('"')
        if not paths[key]:
            sys.exit('No %s csv was selected, so nothing was started.' % key)
        paths[key] = os.path.abspath(paths[key])
        if not os.path.isfile(paths[key]):
            sys.exit('ERROR: %s was not found.' % paths[key])
        folder = os.path.dirname(paths[key])
        print('      %s: %s' % (key, paths[key]), flush=True)
    if root is not None:
        root.destroy()
    if os.path.normcase(paths['train']) == os.path.normcase(paths['valid']):
        sys.exit('ERROR: the same file was selected as train and as valid.')
    os.makedirs(CACHE, exist_ok=True)
    with open(LAST_FILES, 'w', encoding='utf-8') as f:
        json.dump(paths, f, indent=1)
    return paths['train'], paths['valid']


def to_int(text):
    try:
        return int(text)
    except ValueError:
        number = float(text)   # ids written as 39.0
        if number != int(number):
            raise
        return int(number)


def read_likes(path):
    """Reads one CSV and returns the user ids and title ids of its rows with scaled_like = 1."""
    csv.field_size_limit(2 ** 31 - 1)
    users, items = array('q'), array('q')
    rows = short = 0
    other = collections.Counter()
    start = time.time()
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader, [])]
        missing = [name for name in (USER_COL, ITEM_COL, LIKE_COL) if name not in header]
        if missing:
            sys.exit('ERROR: %s has no column named %s.\nIts columns are: %s'
                     % (path, ' or '.join(missing), ', '.join(header) or '(none)'))
        iu, ii, il = header.index(USER_COL), header.index(ITEM_COL), header.index(LIKE_COL)
        need = max(iu, ii, il) + 1
        for row in reader:
            rows += 1
            if len(row) < need:
                short += 1
                continue
            value = row[il].strip().lower()
            if value in LIKE:
                try:
                    users.append(to_int(row[iu]))
                    items.append(to_int(row[ii]))
                except ValueError:
                    sys.exit('ERROR: row %d of %s has a %s or %s that is not a whole number: %r, %r'
                             % (rows + 1, path, USER_COL, ITEM_COL, row[iu], row[ii]))
            elif value not in NOT_LIKE:
                if len(other) < 1000 or value in other:
                    other[value] += 1
            if rows % 1000000 == 0:
                print('      %d million rows read ...' % (rows // 1000000), flush=True)
    if other:
        examples = ', '.join('%r in %d rows' % pair for pair in other.most_common(5))
        sys.exit('ERROR: the %s column of %s must hold 1 or 0, but it also holds: %s' % (LIKE_COL, path, examples))
    if short:
        sys.exit('ERROR: %d rows of %s have fewer columns than its header line.' % (short, path))
    print('      %s: %d rows, %d with %s = 1 (%.0f s)'
          % (os.path.basename(path), rows, len(users), LIKE_COL, time.time() - start), flush=True)
    return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64)


def file_key(path):
    info = os.stat(path)
    return [os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns]


def load(train_path, valid_path):
    """Returns the likes of both CSVs as arrays of consecutive user and title numbers.

    Keys of the returned dict:
      train_u, train_i    the train likes (user number, title number), no duplicates
      valid_u, valid_i    the valid likes that are not in train, no duplicates
      user_ids, item_ids  the AniList id of every user number / title number
      n_users, n_items, key (changes when either CSV changes), train_file, valid_file
    """
    key = hashlib.sha1(json.dumps([file_key(train_path), file_key(valid_path), USER_COL, ITEM_COL, LIKE_COL])
                       .encode('utf-8')).hexdigest()[:16]
    cache_file = os.path.join(CACHE, 'likes_%s.npz' % key)
    if os.path.exists(cache_file):
        print('      Using the preprocessed data saved earlier: %s' % cache_file)
        with np.load(cache_file) as saved:
            data = {name: saved[name] for name in saved.files}
    else:
        tu, ti = read_likes(train_path)
        vu, vi = read_likes(valid_path)
        if len(tu) == 0:
            sys.exit('ERROR: the train csv has no row with %s = 1.' % LIKE_COL)
        if len(vu) == 0:
            sys.exit('ERROR: the valid csv has no row with %s = 1.' % LIKE_COL)
        user_ids = np.unique(np.concatenate([tu, vu]))
        item_ids = np.unique(np.concatenate([ti, vi]))
        n_items = len(item_ids)
        train = np.unique(np.searchsorted(user_ids, tu) * n_items + np.searchsorted(item_ids, ti))
        valid = np.unique(np.searchsorted(user_ids, vu) * n_items + np.searchsorted(item_ids, vi))
        in_train = np.isin(valid, train, assume_unique=True)
        data = {'train_u': train // n_items, 'train_i': train % n_items,
                'valid_u': valid[~in_train] // n_items, 'valid_i': valid[~in_train] % n_items,
                'user_ids': user_ids, 'item_ids': item_ids,
                'left_out': np.array([len(tu) - len(train), len(vu) - len(valid), int(in_train.sum())])}
        if len(data['valid_u']) == 0:
            sys.exit('ERROR: every like of the valid csv is also in the train csv.')
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache_file + '.tmp.npz', **data)
        os.replace(cache_file + '.tmp.npz', cache_file)
    data.update(n_users=len(data['user_ids']), n_items=len(data['item_ids']), key=key,
                train_file=os.path.abspath(train_path), valid_file=os.path.abspath(valid_path))
    describe(data)
    return data


def describe(data):
    n_users, n_items = data['n_users'], data['n_items']
    train_users = np.zeros(n_users, dtype=bool)
    train_users[data['train_u']] = True
    train_items = np.zeros(n_items, dtype=bool)
    train_items[data['train_i']] = True
    valid_users = np.unique(data['valid_u'])
    dup_train, dup_valid, both = (int(x) for x in data['left_out'])
    print('      users %d | titles %d | train likes %d | valid likes %d'
          % (n_users, n_items, len(data['train_u']), len(data['valid_u'])))
    print('      users with a valid like: %d (%d of them have no train like)'
          % (len(valid_users), int((~train_users[valid_users]).sum())))
    print('      valid likes on titles nobody liked in train: %d' % int((~train_items[data['valid_i']]).sum()))
    if dup_train or dup_valid or both:
        print('      left out: %d repeated train likes, %d repeated valid likes, %d valid likes that are also in train'
              % (dup_train, dup_valid, both))
    sys.stdout.flush()
