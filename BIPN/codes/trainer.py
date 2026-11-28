#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author:
# @Date  : 2021/11/1 15:45
# @Desc  :
import copy
import json
import sys
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data_set import DataSet
from metrics import metrics_dict


class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        # the best epoch is the one with the highest value of this on validation
        self.select_metric = getattr(args, 'select_metric', 'hit@20')
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name

        self.device = args.device
        self.TIME = args.TIME

        # Resume support, only active when main.py is started with --resume:
        # progress is saved every --save_minutes and after every epoch, and the
        # next start with the same --model_name continues from the last save.
        self.resume = getattr(args, 'resume', False)
        self.save_seconds = 60 * getattr(args, 'save_minutes', 30)
        self.resume_path = os.path.join(self.model_path, self.model_name + '_resume.pt')
        self.result_path = os.path.join(self.model_path, self.model_name + '_result.json')
        self.settings = {k: str(getattr(args, k, None)) for k in (
            'data_name', 'behaviors', 'embedding_size', 'layers', 'lr', 'decay', 'reg_weight', 'log_reg',
            'batch_size', 'neg_count', 'node_dropout', 'message_dropout', 'loss_type', 'data_key')}
        self.last_save = time.time()
        self.best_result = 0
        self.best_dict = {}
        self.best_epoch = 0
        self.best_model = None
        self.final_test = None

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

    def clear_parameter(self, model):
        # for device in model.device_ids:
        #     model.module.storage_user_embeddings = None
        #     model.module.storage_item_embeddings = None
        model.storage_user_embeddings = None
        model.storage_item_embeddings = None

    @staticmethod
    def _plain(metric_dict):
        # numpy numbers -> plain floats, so they can be saved and written as json
        return None if metric_dict is None else {k: float(v) for k, v in metric_dict.items()}

    def save_progress(self, epoch, batch_no, loss_sum, finished=False):
        """Saves everything needed to continue: epoch = epoch in progress, batch_no = batches of it already done."""
        state = {
            'settings': self.settings,
            'epoch': epoch,
            'batch_no': batch_no,
            'loss_sum': float(loss_sum),
            'finished': finished,
            'best_result': float(self.best_result),
            'best_epoch': self.best_epoch,
            'best_dict': self._plain(self.best_dict),
            'final_test': self._plain(self.final_test),
            'model': None if finished else self.model.state_dict(),
            'optimizer': None if finished else self.optimizer.state_dict(),
            'best_model': None if finished or self.best_model is None else self.best_model.state_dict(),
        }
        tmp_path = self.resume_path + '.tmp'
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.resume_path)  # a stop in the middle of saving cannot damage the last save
        self.last_save = time.time()

    def load_progress(self):
        if not os.path.exists(self.resume_path):
            return None
        state = torch.load(self.resume_path, map_location=self.device)
        if state['settings'] != self.settings:
            changed = ['%s: %s -> %s' % (k, state['settings'].get(k), v)
                       for k, v in self.settings.items() if state['settings'].get(k) != v]
            # sys.exit (not an exception) so that @logger.catch does not swallow it and the caller sees a failure
            sys.exit('ERROR: the saved progress in %s was made with different settings (%s). Delete that file '
                     'to start over, or use another --model_name.' % (self.resume_path, '; '.join(changed)))
        return state

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        start_epoch, start_batch, loss_sum = 0, 0, 0.0
        if self.resume:
            state = self.load_progress()
            if state is None:
                if os.path.exists(self.result_path):
                    os.remove(self.result_path)  # left over from an earlier run with this name
            else:
                self.best_result = state['best_result']
                self.best_epoch = state['best_epoch']
                self.best_dict = state['best_dict']
                self.final_test = state['final_test']
                if state['finished']:
                    logger.info('this run has already finished, nothing left to train (%s)' % self.resume_path)
                    logger.info(f"training end, best iteration %d, results: %s" %
                                (self.best_epoch + 1, self.best_dict.__str__()))
                    logger.info(f"final test result is:  %s" % self.final_test.__str__())
                    return
                self.model.load_state_dict(state['model'])
                self.optimizer.load_state_dict(state['optimizer'])
                if state['best_model'] is not None:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_model.load_state_dict(state['best_model'])
                start_epoch, start_batch, loss_sum = state['epoch'], state['batch_no'], state['loss_sum']
                logger.info('continuing from saved progress: epoch %d, %d of %d batches done' %
                            (start_epoch + 1, start_batch, len(train_dataset_loader)))

        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        for epoch in range(start_epoch, self.epochs):

            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(train_dataset_loader, epoch,
                                                                           start_batch, loss_sum)
            start_batch, loss_sum = 0, 0.0

            stop = False
            if validate_metric_dict is not None:
                result = validate_metric_dict[self.select_metric]
                # early stop
                if result - self.best_result > 0:
                    self.final_test = test_metric_dict
                    self.best_result = result
                    self.best_dict = validate_metric_dict
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch
                if epoch - self.best_epoch > 4:
                    stop = True
            if self.resume:
                self.save_progress(epoch + 1, 0, 0.0)
            if stop:
                break
        # save the best model
        if self.best_model is None and self.resume:
            logger.info('no epoch scored above 0 on validation %s, so there is no best model to save' % self.select_metric)
        else:
            self.save_model(self.best_model)
        logger.info(f"training end, best iteration %d, results: %s" %
                    (self.best_epoch + 1, self.best_dict.__str__()))

        logger.info(f"final test result is:  %s" % self.final_test.__str__())
        if self.resume:
            with open(self.result_path, 'w', encoding='utf-8') as f:
                json.dump({'model_name': self.model_name, 'settings': self.settings,
                           'best_epoch': self.best_epoch + 1, 'validation': self._plain(self.best_dict),
                           'test': self._plain(self.final_test)}, f, indent=1)
            self.save_progress(self.epochs, 0, 0.0, finished=True)

    def _train_one_epoch(self, behavior_dataset_loader, epoch, start_batch=0, loss_sum=0.0):
        start_time = time.time()
        total_batches = len(behavior_dataset_loader)
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=total_batches - start_batch,
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )
        # start_batch > 0 only right after a resume: the rest of the interrupted epoch is trained
        total_loss = loss_sum
        batch_no = start_batch
        for batch_index, batch_data in behavior_dataset_iter:
            if batch_no >= total_batches:
                break
            start = time.time()
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            # loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            batch_no += 1
            total_loss += loss.item()
            if self.resume and time.time() - self.last_save >= self.save_seconds:
                self.save_progress(epoch, batch_no, total_loss)

        if self.resume:
            self.save_progress(epoch, batch_no, total_loss)  # so a stop during evaluation loses no training
        total_loss = total_loss / max(batch_no, 1)

        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        self.clear_parameter(self.model)
        # validate
        start_time = time.time()
        validate_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.validate_dataset(),
                                             self.dataset.validation_interacts, self.dataset.validation_gt_length)
        epoch_time = time.time() - start_time
        logger.info(
            f"validate %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, validate_metric_dict.__str__()))

        # test
        start_time = time.time()
        test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_interacts, self.dataset.test_gt_length)
        epoch_time = time.time() - start_time
        logger.info(
            f"test %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

        return test_metric_dict, validate_metric_dict

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        # Titles left out of each user's ranking: everything the user has a train row for when the
        # data has seen_dict.txt (AniList), otherwise the user's training items of the target behavior.
        train_items = getattr(self.dataset, 'seen_dict', None) or self.dataset.train_behavior_dict[self.behaviors[-1]]
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            start = time.time()
            # scores = self.model.module.full_predict(batch_data)
            scores = self.model.full_predict(batch_data)

            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                user_score[0] = -np.inf  # number 0 is padding, not a title
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name + self.TIME + '.pth'))
