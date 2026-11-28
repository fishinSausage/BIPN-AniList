#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2023/9/23 15:25
# @Desc  :
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from data_set import DataSet
from model import BIPN

from trainer import Trainer
from trainer_2GPU import Trainer as trainer_2GPU

seed = 2021
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--log_reg', type=float, default=0.5, help='')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)
    parser.add_argument('--omega', type=float, default=1)

    parser.add_argument('--data_name', type=str, default='tmall', help='')
    parser.add_argument('--data_dir', type=str, default='',
                        help='AniList only: folder of converted data, instead of the one that belongs to --data_name')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=4)

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--gpu_no', type=int, default=1, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='tmall', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--resume', action='store_true',
                        help='save progress while training and continue from it on the next start')
    parser.add_argument('--save_minutes', type=float, default=30, help='with --resume: minutes between saves')

    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'yelp':
        args.data_path = './data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'ml':
        args.data_path = './data/ML10M'
        args.behaviors = ['neutral', 'neg', 'pos']
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view', 'cart', 'buy']
    elif args.data_name in ('anilist', 'anilist_sample', 'anilist_likes') or (
            args.data_name.startswith('anilist') and args.data_dir):
        # made by data/AniList/data_process.py: anilist = all behaviors, anilist_sample = the 10% user
        # sample for grid search, anilist_likes = likes only (the plain BIPN baseline)
        args.data_path = args.data_dir or {'anilist': './data/AniList', 'anilist_sample': './data/AniList_sample',
                                           'anilist_likes': './data/AniList_likes'}[args.data_name]
        args.behaviors = ['planning', 'current', 'paused', 'completed', 'repeating', 'dropped',
                          'dislike', 'neutral', 'like']
        # Same evaluation as the LightGCN baseline: recall@20 and ndcg@20 on the validation / test
        # rows with eval_like = 1, best epoch chosen by validation ndcg@20 (SELECT in LightGCN/lightgcn.py).
        args.topk = [20]
        args.metrics = ['recall', 'ndcg']
        args.select_metric = 'ndcg@20'
        key_file = os.path.join(args.data_path, 'data_key.txt')
        if not os.path.exists(key_file):
            sys.exit('ERROR: %s was not made from the train, validation and test CSV. Run '
                     'setup_and_train_anilist.bat (or data/AniList/data_process.py) to convert them.' % args.data_path)
        with open(key_file) as f:
            args.data_key = f.read().strip()  # saved progress made from other CSVs is not continued
        behaviors_file = os.path.join(args.data_path, 'behaviors.json')
        if os.path.exists(behaviors_file):  # the behaviors of this conversion; the last one is the target
            with open(behaviors_file) as f:
                args.behaviors = json.load(f)
    else:
        raise Exception('data_name cannot be None')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    # args.train_writer = SummaryWriter('./log/train/' + logfile)
    # args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = BIPN(args, dataset).to(args.device)
    if args.gpu_no == 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
        trainer = trainer_2GPU(model, dataset, args)
    else:
        trainer = Trainer(model, dataset, args)

    logger.info(args.__str__())
    logger.info(model)

    trainer.train_model()
    # trainer.evaluate(0, 5, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



