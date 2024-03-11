import torch
from torch import nn, autograd
import torch.nn.functional as F
import numpy as np
from filelock import FileLock
import os
with FileLock(os.path.expanduser("~/.data.lock")):
    from rspmm.rspmm import generalized_rspmm, sparse_coo_tensor

class NBFNet(nn.Module):
    def __init__(self, params, dataloader):
        super(NBFNet, self).__init__()
        self.n_rel = 2 * dataloader.n_rel + 1
        self.in_device = params.device
        self.n_ent = dataloader.n_ent
        self.dataset = dataloader.KG.astype(np.int64)
        self.hidden_dim = params.hidden_dim
        self.n_layer = params.n_layer
        self.sum = params.sum
        self.mul = params.mul
        self.rela_embed_layer = nn.ModuleList([nn.Embedding(self.n_rel, self.hidden_dim) for _ in range(self.n_layer)])
        self.ent_embedding = nn.Embedding(self.n_ent, self.hidden_dim)
        self.query_embedding = nn.Embedding(1, self.hidden_dim)
        self.linear_classifier = nn.Linear(self.hidden_dim, 1)

        self.linear = nn.ModuleList([nn.Linear(1*self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])

        self.layer_norm = nn.ModuleList([nn.LayerNorm([self.hidden_dim]) for _ in range(self.n_layer)])

        self.linear_ent = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])

        self.select_degree = params.degree
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        if params.act == 'tanh':
            self.act = torch.tanh
        elif params.act == 'sig':
            self.act = torch.sigmoid
        elif params.act == 'relu':
            self.act = torch.relu
        elif params.act == 'idd':
            self.act = lambda x : x
        elif params.act == 'softplus':
            self.act = F.softplus
        elif params.act == 'glu':
            self.act = F.glu
        
        print(f'dropout is {params.dropout}')

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, head, rel):
        batch_size = head.shape[0]
        query_rel = torch.from_numpy(rel).to(self.in_device)
        entity = torch.from_numpy(head).to(self.in_device)
        dataset = torch.as_tensor(self.dataset, dtype=torch.long)
        cur_entity = torch.column_stack([torch.arange(batch_size, device=self.in_device), entity])
        # hidden_embedding = torch.zeros((self.n_ent, batch_size, self.hidden_dim), device=self.in_device)
        hidden_embedding = torch.ones((self.n_ent, batch_size, self.hidden_dim), device=self.in_device) / np.sqrt(self.hidden_dim) - 0.5

        adj_mat = sparse_coo_tensor(
            indices=torch.stack([dataset[:, 0], dataset[:, 2], dataset[:, 1]]).to(self.in_device),
            values=torch.ones(dataset.shape[0], dtype=torch.float32, device=self.in_device),
            size=(self.n_ent, self.n_ent, self.n_rel)
        ).transpose(0, 1)

        if True:
            constant_idx = self.get_indx_constants(adj_mat, self.select_degree)

            cord_x, cord_y = torch.meshgrid(constant_idx, torch.arange(batch_size, device=self.in_device))

            hidden_embedding[cord_x.reshape(-1), cord_y.reshape(-1), :] = self.ent_embedding(cord_x.reshape(-1))

        hidden_embedding[entity, np.arange(batch_size), :] = self.query_embedding(query_rel)
        boundary = hidden_embedding
        layer_input = hidden_embedding

        for i in range(self.n_layer):
            # handle relation
            relation = self.rela_embed_layer[i].weight.unsqueeze(1).repeat(1, batch_size, 1).flatten(1)
            
            layer_input = self.linear_ent[i](layer_input)

            update = generalized_rspmm(adj_mat, relation, layer_input.flatten(1), sum=self.sum, mul=self.mul).view(self.n_ent, batch_size, self.hidden_dim)

            layer_input = self.linear[i](torch.cat([update], dim=-1)) + layer_input

            layer_input = self.act(self.dropout(self.layer_norm[i](layer_input)))

        result = self.linear_classifier(layer_input).squeeze().permute(1, 0)
        return result
    
    def get_indx_constants(self, adj_mat, select_degree=3):
        adj_mat = adj_mat.to_dense()
        degree = torch.sum(adj_mat[:,:,1:(self.n_rel-1) // 2], dim=(1, 2))
        constant_idx = torch.where(degree >= select_degree)[0]
        return constant_idx