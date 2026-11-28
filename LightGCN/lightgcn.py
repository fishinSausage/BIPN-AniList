#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LightGCN: model, triple sampler, training loop and evaluation.

Follows "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
(He et al., SIGIR 2020, arXiv 2002.02126):

  * propagation  E(k+1) = D^-1/2 A D^-1/2 E(k): no self-loops, no weight matrices, no
    activation function (Eq. 3 and 7)
  * final embedding = mean of the layers 0..K, i.e. alpha_k = 1/(K+1) (Eq. 4)
  * score = inner product of the user and the title embedding (Eq. 5)
  * BPR loss with L2 regularization on the layer-0 embeddings only, no dropout (Eq. 15)
  * the fixed settings below (Section 4.1.2)
  * recall@20 and ndcg@20 with the all-ranking protocol (Section 4.1)
  * early stopping as in NGCF, which the paper refers to: stop when the validation result
    has not increased for 50 epochs

One deliberate difference: NGCF watches recall@20 for the best epoch and the early stop. This
project watches ndcg@20 (SELECT below), the measure BIPN's best epoch is chosen by as well.

Where the paper gives no detail, the authors' code is followed
(github.com/gusye1234/LightGCN-PyTorch): how the triples are sampled, the exact form of the
loss and the recall and NDCG formulas.

Evaluation rules of this project (the same as BIPN's, see preprocess.py): the graph is made
of the train likes (scaled_like = 1); the candidates of a user are all titles of the train
csv except the ones the user has a train row for; the answers are the validation / test
rows with eval_like = 1.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Fixed settings (Section 4.1.2 of the paper) ------------------------------
EMBEDDING_SIZE = 64      # "the embedding size is fixed to 64"
LEARNING_RATE = 0.001    # Adam with "the default learning rate of 0.001"
BATCH_SIZE = 1024        # "default mini-batch size of 1024"
MAX_EPOCHS = 1000        # "typically, 1000 epochs are sufficient for LightGCN to converge"
PATIENCE_EPOCHS = 50     # NGCF: stop if the validation result does not increase for 50 epochs
TOP_K = 20               # recall@20 and ndcg@20
# ---- Not from the paper: these do not change the model ------------------------
EVAL_EVERY = 5           # validation is computed every 5 epochs
SEED = 2020
SELECT = 'ndcg@20'       # the best epoch and the best combination are chosen by this, on validation
#                          (NGCF and the LightGCN paper use recall@20; BIPN's main.py uses the same as here)
EVAL_BATCH = 1024        # users scored at once during evaluation
# -------------------------------------------------------------------------------
METRICS = ('recall@%d' % TOP_K, 'ndcg@%d' % TOP_K)


def settings():
    return {'embedding_size': EMBEDDING_SIZE, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS, 'patience_epochs': PATIENCE_EPOCHS, 'eval_every': EVAL_EVERY,
            'seed': SEED, 'top_k': TOP_K, 'select': SELECT}


def normalized_adjacency(train_u, train_i, n_users, n_items, device):
    """D^-1/2 A D^-1/2 for A = [[0, R], [R^T, 0]] as a sparse tensor; users first, then titles."""
    n = n_users + n_items
    rows = np.concatenate([train_u, train_i + n_users])
    cols = np.concatenate([train_i + n_users, train_u])
    a = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    degree = np.asarray(a.sum(axis=1)).flatten()
    d = np.zeros(n, dtype=np.float32)
    d[degree > 0] = np.power(degree[degree > 0], -0.5)
    a = (sp.diags(d) @ a @ sp.diags(d)).tocoo()
    index = torch.from_numpy(np.vstack([a.row, a.col]).astype(np.int64))
    value = torch.from_numpy(a.data.astype(np.float32))
    return torch.sparse_coo_tensor(index, value, (n, n)).coalesce().to(device)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, n_layers, adjacency):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.user_emb = nn.Embedding(n_users, EMBEDDING_SIZE)
        self.item_emb = nn.Embedding(n_items, EMBEDDING_SIZE)
        # "the embedding parameters are initialized with the Xavier method"
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.adjacency = adjacency

    def propagate(self):
        """Returns the final user and title embeddings: the mean of the layers 0..K."""
        e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        total = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.adjacency, e)
            total = total + e
        return torch.split(total / (self.n_layers + 1), [self.n_users, self.n_items])

    def bpr_loss(self, users, pos, neg, lam):
        """BPR loss of a batch of (user, liked title, not-liked title) plus lam * L2 of their layer-0 embeddings."""
        user_all, item_all = self.propagate()
        u = user_all[users]
        pos_score = (u * item_all[pos]).sum(dim=-1)
        neg_score = (u * item_all[neg]).sum(dim=-1)
        loss = F.softplus(neg_score - pos_score).mean()   # = mean of -ln sigmoid(pos - neg)
        reg = 0.5 * (self.user_emb(users).pow(2).sum() + self.item_emb(pos).pow(2).sum()
                     + self.item_emb(neg).pow(2).sum()) / len(users)
        return loss + lam * reg


class Sampler:
    """Draws the training triples of one epoch on the device, the way the authors' code does.

    As many draws as there are train likes: a user uniformly at random (users without a like
    are skipped), one of that user's liked titles uniformly, and one title the user has not
    liked uniformly.
    """

    def __init__(self, train_u, train_i, n_users, n_items, device):
        self.n_users, self.n_items, self.n_draws, self.device = n_users, n_items, len(train_u), device
        order = np.lexsort((train_i, train_u))
        counts = np.bincount(train_u, minlength=n_users)
        self.liked = torch.from_numpy(train_i[order]).to(device)                       # grouped by user
        self.start = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)[:-1]])).to(device)
        self.count = torch.from_numpy(counts).to(device)
        self.keys = torch.from_numpy(np.sort(train_u * n_items + train_i)).to(device)  # for "has the user liked it"

    def sample(self):
        users = torch.randint(0, self.n_users, (self.n_draws,), device=self.device)
        users = users[self.count[users] > 0]
        count = self.count[users]
        offset = (torch.rand(len(users), device=self.device) * count).long()
        pos = self.liked[self.start[users] + torch.minimum(offset, count - 1)]
        neg = torch.randint(0, self.n_items, (len(users),), device=self.device)
        while True:
            keys = users * self.n_items + neg
            place = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
            clash = self.keys[place] == keys
            n_clash = int(clash.sum())
            if n_clash == 0:
                return users, pos, neg
            neg[clash] = torch.randint(0, self.n_items, (n_clash,), device=self.device)


class Evaluator:
    """recall@20 and ndcg@20 of one answer set ('valid' or 'test') with the all-ranking protocol.

    Every user with at least one answer is scored. The candidates are all titles except the
    ones the user has a row for in the train csv.
    """

    def __init__(self, data, part, device):
        shape = (data['n_users'], data['n_items'])
        answer_u, answer_i = data[part + '_u'], data[part + '_i']
        seen = sp.csr_matrix((np.ones(len(data['seen_u']), dtype=np.int8), (data['seen_u'], data['seen_i'])), shape=shape)
        answers = sp.csr_matrix((np.ones(len(answer_u), dtype=np.int8), (answer_u, answer_i)), shape=shape)
        users = np.unique(answer_u)
        self.n_eval, self.top = len(users), min(TOP_K, data['n_items'])
        self.batches = []
        for s in range(0, len(users), EVAL_BATCH):
            batch = users[s:s + EVAL_BATCH]
            se, an = seen[batch].tocoo(), answers[batch].tocoo()
            self.batches.append([torch.from_numpy(np.asarray(x)).long().to(device) for x in
                                 (batch, se.row, se.col, an.row, an.col, answers[batch].getnnz(axis=1))])
        discount = 1.0 / np.log2(np.arange(TOP_K) + 2.0)
        self.discount = torch.from_numpy(discount).float().to(device)
        self.ideal = torch.from_numpy(np.cumsum(discount)).float().to(device)   # ideal[n-1] = best DCG with n hits

    @torch.no_grad()
    def __call__(self, model):
        model.eval()
        user_all, item_all = model.propagate()
        recall = ndcg = 0.0
        for users, seen_row, seen_col, an_row, an_col, n_answers in self.batches:
            scores = user_all[users] @ item_all.t()
            scores[seen_row, seen_col] = -float('inf')    # titles the user has a train row for are not candidates
            top = scores.topk(self.top, dim=1)[1]
            truth = torch.zeros_like(scores, dtype=torch.bool)
            truth[an_row, an_col] = True
            hits = truth.gather(1, top).float()           # 1 where the title at that rank is an answer
            dcg = (hits * self.discount[:hits.shape[1]]).sum(dim=1)
            ideal = self.ideal[torch.clamp(n_answers, max=TOP_K) - 1]
            recall += float((hits.sum(dim=1) / n_answers).sum())
            ndcg += float((dcg / ideal).sum())
        model.train()
        return {METRICS[0]: round(recall / self.n_eval, 6), METRICS[1]: round(ndcg / self.n_eval, 6)}


def train(data, n_layers, lam, device, max_epochs=MAX_EPOCHS, log=print):
    """Trains one LightGCN and returns its best validation epoch.

    Returned dict: best_epoch, epochs (how many were run), validation and test (recall@20 and
    ndcg@20 at the best validation epoch), seconds, seconds_per_epoch and state (the layer-0
    and the final embeddings at the best epoch).
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adjacency = normalized_adjacency(data['train_u'], data['train_i'], data['n_users'], data['n_items'], device)
    model = LightGCN(data['n_users'], data['n_items'], n_layers, adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = Sampler(data['train_u'], data['train_i'], data['n_users'], data['n_items'], device)
    validate, test = Evaluator(data, 'valid', device), Evaluator(data, 'test', device)

    best, best_test, best_epoch, best_state, epoch, start = None, None, 0, None, 0, time.time()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        users, pos, neg = sampler.sample()
        total, batches = torch.zeros((), device=device), 0
        for s in range(0, len(users), BATCH_SIZE):
            loss = model.bpr_loss(users[s:s + BATCH_SIZE], pos[s:s + BATCH_SIZE], neg[s:s + BATCH_SIZE], lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            batches += 1
        mean_loss = float(total) / max(batches, 1)
        if not np.isfinite(mean_loss):
            raise RuntimeError('the loss is not a number any more (epoch %d)' % epoch)
        if epoch == 1:
            log('  epoch 1 took %.1f s (%d batches)' % (time.time() - epoch_start, batches))
        if epoch % EVAL_EVERY and epoch != max_epochs:
            continue
        metrics = validate(model)
        if best is None or metrics[SELECT] > best[SELECT]:
            best, best_epoch, best_test = metrics, epoch, test(model)   # test is only reported, never used to choose
            with torch.no_grad():
                user_final, item_final = model.propagate()
            best_state = {'user_embedding': model.user_emb.weight.detach().cpu().clone(),
                          'item_embedding': model.item_emb.weight.detach().cpu().clone(),
                          'user_final': user_final.cpu().clone(), 'item_final': item_final.cpu().clone()}
        log('  epoch %4d | loss %.4f | validation recall@20 %.4f  ndcg@20 %.4f | best %s %.4f at epoch %d | %.1f min'
            % (epoch, mean_loss, metrics[METRICS[0]], metrics[METRICS[1]], SELECT, best[SELECT], best_epoch,
               (time.time() - start) / 60))
        if epoch - best_epoch >= PATIENCE_EPOCHS:
            log('  stopped: validation %s has not increased for %d epochs' % (SELECT, PATIENCE_EPOCHS))
            break
    seconds = time.time() - start
    return {'best_epoch': best_epoch, 'epochs': epoch, 'validation': best, 'test': best_test,
            'seconds': round(seconds, 1), 'seconds_per_epoch': round(seconds / max(epoch, 1), 2), 'state': best_state}
