import os
import torch
import numpy as np
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
import argparse

from load_data import DataLoader
from models import NBFNet
from utils import cal_performance, cal_ranks_v2
from util import *

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
DATASET = None

class Options(object):
    pass

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

        model.zero_grad()
        scores = model(triple[:,0], triple[:,1])

        pos_scores = scores[[torch.arange(len(scores)).to(args.device),torch.LongTensor(triple[:,2]).to(args.device)]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 
        loss.backward()
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
    ranking = np.array(ranking)
    t_mrr, t_h1, t_h10 = cal_performance(ranking)
    scheduler.step()

    valid_h1, out_str, test_h1 = evaluate(model, dataloader, args, mode='Valid')
    dataloader.shuffle_train()
    return valid_h1, out_str, test_h1
    


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
    return v_h1, out_str, t_h1



def objective(opt):
    dataloader = DataLoader(DATASET)

    gpu_id = ray.get_gpu_ids()
    print(gpu_id)
    device = 'cuda'
    print(device)

    args = Options

    args.batch_size = opt['batch_size']
    args.lr = 10**opt['lr']
    args.decay_rate = opt['decay_rate']
    args.weight_decay = 10**opt['weight_decay']
    args.hidden_dim = opt['hidden_dim']
    args.n_layer = 5
    args.dropout = opt['dropout']
    args.act = opt['act']
    args.n_tbatch = 100
    args.device = device
    args.epoch = 50
    args.sum = opt['sum']
    args.mul = opt['mul']
    args.degree = opt['degree']

    # Configure model, optimizer, scheduler
    model = NBFNet(args, dataloader).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay_rate)

    best_h1 = 0
    for epoch in range(args.epoch):
        valid_h1, out_str, t_h1 = train_one_epoch(model, dataloader, optimizer, scheduler, args)
        if valid_h1 > best_h1:
            best_h1 = valid_h1
            best_str = out_str
        session.report({"h1": valid_h1, "test_h1": t_h1})
    print(best_str)

space = {
    "lr": tune.uniform(-4, -2),
    "decay_rate": tune.uniform(0.99, 1.0),
    'hidden_dim': tune.choice([16, 32, 64, 128]),
    'dropout': tune.uniform(0, 0.3),
    'act': tune.choice(['tanh', 'idd', 'sig', 'relu']),
    'batch_size': tune.choice([10, 20, 30, 40, 50]),
    'weight_decay': tune.choice([0, 1e-1, 1e-2, 1e-3, 1e-4]),
    'sum': tune.choice(['add', 'min', 'max']),
    'mul': tune.choice(['add', 'mul']),
    'degree': tune.choice([1, 3, 5, 7, 9]),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EL-GNN'
    )
    parser.add_argument('--dataset', type=str, default="U")
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET = f'{current_dir}/data/{args.dataset}'

    ray.init(num_gpus=1, num_cpus=4)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(
            objective,
            resources={"cpu": 4, "gpu": 0.25}
        ),
        tune_config=tune.TuneConfig(
            metric="h1",
            mode="max",
            scheduler=sched,
            num_samples=4,
        ),
        run_config=air.RunConfig(
            name="exp",
            stop={"training_iteration": 100},
        ),
        param_space=space,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    print(results.get_best_result())
