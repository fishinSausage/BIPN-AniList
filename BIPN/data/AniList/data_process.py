#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turns interactions.csv into the files BIPN trains on.

    python data_process.py                      # full data, written next to this file
    python data_process.py --sample 0.1 --out ../AniList_sample   # 10% of users

interactions.csv is only read, never changed.

Behaviors (each row goes into its status list, plus a rating list if it has a score):
    planning, current, paused, completed, repeating, dropped   <- status column
    dislike (score <= 40), neutral (41-79), like (>= 80)       <- score_100 column
"like" is the target behavior.

Split: for every user with at least 3 likes, the most recently updated like
(updated_at) is the test title and the one before it is the validation title.
Likes with the same timestamp are ordered at random (fixed seed). A held-out
title is removed from all of that user's lists, so the answer is not in the
training data.

User and title numbers are renumbered 1..N; user_ids.csv and title_ids.csv map
them back to the AniList ids.
"""
import argparse
import csv
import json
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
# Same order as the 'anilist' entry in main.py. The last one is the target.
BEHAVIORS = ['planning', 'current', 'paused', 'completed', 'repeating', 'dropped', 'dislike', 'neutral', 'like']
TARGET = 'like'
LIKE_MIN = 80       # like:    score_100 >= 80  (4 stars or more out of 5)
DISLIKE_MAX = 40    # dislike: score_100 <= 40  (2 stars or fewer)
MIN_LIKES_TO_TEST = 3
HERE = os.path.dirname(os.path.abspath(__file__))


def rating_behavior(score):
    if score >= LIKE_MIN:
        return 'like'
    if score <= DISLIKE_MAX:
        return 'dislike'
    return 'neutral'


def read_csv(path):
    """Returns {user: {title: (status behavior or None, rating behavior or None, updated_at)}}, titles, notes."""
    csv.field_size_limit(2 ** 31 - 1)
    users, titles = {}, {}
    notes = {'rows': 0, 'malformed': 0, 'unknown_status': 0, 'repeated_pairs': 0}
    t0 = time.time()
    with open(path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        col = {h: i for i, h in enumerate(header)}
        needed = ['user_id', 'media_id', 'status', 'score_100', 'score_available', 'updated_at']
        missing = [c for c in needed if c not in col]
        if missing:
            sys.exit('ERROR: interactions.csv has no column(s) %s. Its columns are: %s' % (missing, header))
        iu, im, ist, isc, ifl, iup = [col[c] for c in needed]
        ititle = col.get('media_title')
        ncol = len(header)
        for row in reader:
            notes['rows'] += 1
            if notes['rows'] % 1000000 == 0:
                print('  ... %d million rows read, %ds' % (notes['rows'] // 1000000, time.time() - t0), flush=True)
            if len(row) != ncol:
                notes['malformed'] += 1
                continue
            try:
                user, title = int(row[iu]), int(row[im])
            except ValueError:
                notes['malformed'] += 1
                continue
            status = STATUS_BEHAVIOR.get(row[ist].strip().upper())
            if status is None:
                notes['unknown_status'] += 1
            rating = None
            if row[ifl].strip().lower() in ('1', '1.0', 'true'):
                try:
                    score = float(row[isc])
                except ValueError:
                    score = 0.0
                if score > 0:
                    rating = rating_behavior(score)
            if status is None and rating is None:
                continue
            try:
                updated = int(float(row[iup]))
            except ValueError:
                updated = 0
            entries = users.setdefault(user, {})
            if title in entries:
                notes['repeated_pairs'] += 1
                if entries[title][2] >= updated:
                    continue
            entries[title] = (status, rating, updated)
            if title not in titles:
                titles[title] = row[ititle].strip() if ititle is not None else ''
    return users, titles, notes


def convert(csv_path, out_dir, sample, seed):
    t0 = time.time()
    print('Reading %s' % csv_path, flush=True)
    users, titles, notes = read_csv(csv_path)
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

    # ---- pick the validation and test title of each user ----------------------
    validation, test = {}, {}
    test_ties = validation_ties = 0
    for u in user_ids:
        likes = sorted(t for t, e in users[u].items() if e[1] == TARGET)
        if len(likes) < MIN_LIKES_TO_TEST:
            continue
        rng.shuffle(likes)                                # random order among equal timestamps
        likes.sort(key=lambda t: users[u][t][2])           # stable: oldest first
        stamps = [users[u][t][2] for t in likes]
        test[u], validation[u] = likes[-1], likes[-2]
        if stamps[-1] == stamps[-2]:
            test_ties += 1
        if stamps[-2] == stamps[-3] or stamps[-1] == stamps[-2]:
            validation_ties += 1

    # ---- build the training lists ------------------------------------------------
    lists = {b: {} for b in BEHAVIORS}                     # behavior -> {new user: [new titles]}
    before = dict.fromkeys(BEHAVIORS, 0)
    for u in user_ids:
        nu = new_user[u]
        held_out = {test.get(u), validation.get(u)}
        for t in sorted(users[u]):
            status, rating, _ = users[u][t]
            for b in (status, rating):
                if b is None:
                    continue
                before[b] += 1
                if t not in held_out:
                    lists[b].setdefault(nu, []).append(new_title[t])

    # ---- write ---------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    def out(name):
        if name.lower() == 'interactions.csv':
            raise RuntimeError('refusing to write to interactions.csv')
        return os.path.join(out_dir, name)

    all_items = {}
    for b in BEHAVIORS:
        with open(out(b + '.txt'), 'w') as f:
            for nu in sorted(lists[b]):
                for nt in lists[b][nu]:
                    f.write('%d %d\n' % (nu, nt))
                all_items.setdefault(nu, set()).update(lists[b][nu])
        with open(out(b + '_dict.txt'), 'w', encoding='utf-8') as f:
            json.dump({str(nu): lists[b][nu] for nu in sorted(lists[b])}, f)
    with open(out('all_dict.txt'), 'w', encoding='utf-8') as f:
        json.dump({str(nu): sorted(all_items[nu]) for nu in sorted(all_items)}, f)
    with open(out('pos_sampling.txt'), 'w') as f:
        for index, b in enumerate(BEHAVIORS):
            for nu in sorted(lists[b]):
                for nt in lists[b][nu]:
                    f.write('%d %d %d 1\n' % (nu, nt, index))
    for name, held in (('validation', validation), ('test', test)):
        pairs = sorted((new_user[u], new_title[t]) for u, t in held.items())
        with open(out(name + '.txt'), 'w') as f:
            for nu, nt in pairs:
                f.write('%d %d\n' % (nu, nt))
        with open(out(name + '_dict.txt'), 'w', encoding='utf-8') as f:
            json.dump({str(nu): [nt] for nu, nt in pairs}, f)
    with open(out('count.txt'), 'w') as f:
        json.dump({'user': len(user_ids), 'item': len(title_ids)}, f)
    with open(out('user_ids.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'anilist_user_id'])
        w.writerows((new_user[u], u) for u in user_ids)
    with open(out('title_ids.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['title', 'anilist_media_id', 'media_title'])
        w.writerows((new_title[t], t, titles[t]) for t in title_ids)

    # ---- report --------------------------------------------------------------------
    tested = len(test)
    say('AniList -> BIPN conversion')
    say('source rows %d | malformed rows skipped %d | rows with an unknown status %d | repeated user-title pairs %d'
        % (notes['rows'], notes['malformed'], notes['unknown_status'], notes['repeated_pairs']))
    if sample < 1.0:
        say('SAMPLE: %d of %d users (%.0f%%), seed %d' % (len(user_ids), len(users), 100 * sample, seed))
    say('users %d | titles %d' % (len(user_ids), len(title_ids)))
    say()
    say('%-10s %12s %12s' % ('behavior', 'all rows', 'training'))
    for b in BEHAVIORS:
        kept = sum(len(v) for v in lists[b].values())
        say('%-10s %12d %12d%s' % (b, before[b], kept, '   <- target' if b == TARGET else ''))
    say('%-10s %12d %12d' % ('TOTAL', sum(before.values()), sum(len(v) for b in BEHAVIORS for v in lists[b].values())))
    say()
    say('users with %d+ likes, each giving 1 test and 1 validation title: %d (%.1f%% of users)'
        % (MIN_LIKES_TO_TEST, tested, 100.0 * tested / max(1, len(user_ids))))
    if tested:
        say('test title decided by a coin flip (latest timestamp shared with another like): %d users (%.1f%%)'
            % (test_ties, 100.0 * test_ties / tested))
        say('validation title decided by a coin flip: %d users (%.1f%%)' % (validation_ties, 100.0 * validation_ties / tested))
    say('written to %s in %ds' % (out_dir, time.time() - t0))
    with open(out('convert_report.txt'), 'w', encoding='utf-8') as f:     # written last: marks a finished conversion
        f.write('\n'.join(report) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the AniList CSV for BIPN.')
    parser.add_argument('--csv', default=os.path.join(HERE, 'interactions.csv'))
    parser.add_argument('--out', default=HERE, help='folder for the converted files')
    parser.add_argument('--sample', type=float, default=1.0, help='fraction of users to keep, e.g. 0.1')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    if not 0 < args.sample <= 1:
        sys.exit('ERROR: --sample must be between 0 and 1')
    if not os.path.exists(args.csv):
        sys.exit('ERROR: %s not found' % args.csv)
    convert(args.csv, os.path.abspath(args.out), args.sample, args.seed)
