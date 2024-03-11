import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import argparse

from load_data import DataLoader
from models import NBFNet
from util import *
from utils import cal_performance, cal_ranks_v2

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser(
        description='EL-GNN'
    )
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--grad_clip', type=float, default=np.inf)
parser.add_argument('--dataset', type=str, default='data/C3')
parser.add_argument('--decay_rate', type=float, default=0.9949514652701398)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--act', type=str, default='tanh')
parser.add_argument('--n_tbatch', type=int, default=50)
parser.add_argument('--sum', type=str, default='min')
parser.add_argument('--mul', type=str, default='add')
parser.add_argument('--degree', type=int, default=100000)

args = parser.parse_args()


loss_list = []

def train_one_epoch(model, dataloader, optimizer, scheduler, args):
    epoch_loss = 0.
    
    batch_size = args.batch_size
    n_batch = dataloader.n_train // batch_size + (dataloader.n_train % batch_size > 0)

    model.train()
    ranking = []
    for i in range(n_batch):
        start = i*batch_size
        end = min(dataloader.n_train, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        triple = dataloader.get_batch(batch_idx)
        neg_triple = dataloader.get_neg_batch(batch_idx)

        model.zero_grad()

        scores = model(triple[:,0], triple[:,1])
        pos_scores = scores[[torch.arange(len(scores)).to(args.device),torch.LongTensor(triple[:,2]).to(args.device)]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 

        loss_all = loss

        loss_all.backward()
        optimizer.step()

        filters = []
        for ii in range(len(triple)):
            filt = dataloader.filters[(triple[ii, 0], triple[ii, 1])]
            filt_1hot = np.zeros((dataloader.n_ent, ))
            if len(filt) > 0:
                filt_1hot[np.array(filt)] = 1
            filt_1hot[triple[ii,2]] = 0
            filters.append(filt_1hot)

        filters = np.array(filters)
        objs = np.zeros_like(filters)
        objs[np.arange(len(scores)), triple[:, 2].reshape(-1)] = 1
        ranks = cal_ranks_v2(scores.data.cpu().numpy(), objs, filters)
        ranking += ranks
        epoch_loss += loss.item()
    loss_list.append(epoch_loss / n_batch)
    ranking = np.array(ranking)
    t_mrr, t_h1, t_h10 = cal_performance(ranking)
    print('[TRAIN] MRR:%.4f H@1:%.4f H@10:%.4f\n'%(t_mrr, t_h1, t_h10))
    scheduler.step()

    valid_mrr, out_str = evaluate(model, dataloader, args, mode='Valid')
    dataloader.shuffle_train()
    return valid_mrr, out_str


def evaluate(model, dataloader, args, mode='Valid'):
    batch_size = args.n_tbatch

    n_data = dataloader.n_valid
    n_batch = n_data // batch_size + (n_data % batch_size > 0)
    ranking = []

    model.eval()
    for i in range(n_batch):
        start = i*batch_size
        end = min(n_data, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        triple = dataloader.get_batch(batch_idx, data='valid')
        scores = model(triple[:, 0], triple[:, 1]).data.cpu().numpy()
        filters = []
        for ii in range(len(triple)):
            filt = dataloader.filters[(triple[ii, 0], triple[ii, 1])]
            filt_1hot = np.zeros((dataloader.n_ent, ))
            if len(filt) > 0:
                filt_1hot[np.array(filt)] = 1
            filt_1hot[triple[ii,2]] = 0
            filters.append(filt_1hot)

        filters = np.array(filters)
        objs = np.zeros_like(filters)
        objs[np.arange(len(scores)), triple[:, 2].reshape(-1)] = 1
        ranks = cal_ranks_v2(scores, objs, filters)
        ranking += ranks
    ranking = np.array(ranking)
    v_mrr, v_h1, v_h10 = cal_performance(ranking)


    n_data = dataloader.n_test
    n_batch = n_data // batch_size + (n_data % batch_size > 0)
    ranking = []
    model.eval()
    for i in range(n_batch):
        start = i*batch_size
        end = min(n_data, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        triple = dataloader.get_batch(batch_idx, data='test')
        scores = model(triple[:, 0], triple[:, 1]).data.cpu().numpy()
        filters = []
        for ii in range(len(triple)):
            filt = dataloader.filters[(triple[ii, 0], triple[ii, 1])]
            filt_1hot = np.zeros((dataloader.n_ent, ))
            if len(filt) > 0:
                filt_1hot[np.array(filt)] = 1
            filt_1hot[triple[ii,2]] = 0
            filters.append(filt_1hot)

        filters = np.array(filters)
        objs = np.zeros_like(filters)
        objs[np.arange(len(scores)), triple[:, 2].reshape(-1)] = 1
        ranks = cal_ranks_v2(scores, objs, filters)
        ranking += ranks
    ranking = np.array(ranking)
    t_mrr, t_h1, t_h10 = cal_performance(ranking)

    out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10)
    return v_mrr, out_str


def main(args):

    dataloader = DataLoader(args.dataset)


    # Configure model, optimizer, scheduler
    model = NBFNet(args, dataloader).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay_rate)

    best_mrr = 0
    for epoch in range(args.epoch):
        if epoch == args.epoch - 1:
            print('ok')
        mrr, out_str = train_one_epoch(model, dataloader, optimizer, scheduler, args)
        if mrr >= best_mrr:
            best_mrr = mrr
            best_str = out_str
        print(str(epoch) + '\t' + out_str)
    print(best_str)


if __name__ == "__main__":
    main(args)
