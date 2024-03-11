import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        self.filters = defaultdict(lambda:set())
        self.valid_filters = defaultdict(lambda:set())
        self.test_filters = defaultdict(lambda:set())

        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        # self.valid_triple = self.read_no_fact_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        
        self.valid_filters = self.combine_dict(self.valid_filters, self.filters)
        self.test_filters = self.combine_dict(self.test_filters, self.filters)
    
        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.train_triple)
        self.valid_data = self.valid_triple
        self.test_data  = self.test_triple

        self.load_graph(self.fact_data)
        # self.load_test_graph(self.fact_data)
        # self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple))
        # self.load_test_graph(self.double_triple(self.train_triple))


        # self.valid_q, self.valid_a = self.load_query(self.valid_data)
        # self.test_q,  self.test_a  = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test  = len(self.test_data)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                # self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        # idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
        # self.KG = np.concatenate([np.array(triples), idd], 0)

        self.KG = np.array(triples)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))


    # def load_test_graph(self, triples):
    #     idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

    #     self.tKG = np.concatenate([np.array(triples), idd], 0)

    #     # self.tKG = np.array(triples)
    #     self.tn_fact = len(self.tKG)
    #     self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            return np.array(self.valid_data)[batch_idx]
        if data=='test':
            return np.array(self.test_data)[batch_idx]
    
    def get_neg_batch(self, batch_idx):
        chosen_data = np.array(self.train_data)[batch_idx]
        all_entity = np.arange(self.n_ent)
        neg_batch = []
        for i in range(len(batch_idx)):
            head_rel = chosen_data[i][:2]
            pos_tail = self.filters[(head_rel[0], head_rel[1])]
            neg_tail = np.random.choice(np.setdiff1d(all_entity, pos_tail))
            neg_batch.append([head_rel[0], head_rel[1], neg_tail])
        return np.array(neg_batch)

        # subs = []
        # rels = []
        # objs = []
        
        # subs = query[batch_idx, 0]
        # rels = query[batch_idx, 1]
        # objs = np.zeros((len(batch_idx), self.n_ent))
        # for i in range(len(batch_idx)):
        #     objs[i][answer[batch_idx[i]]] = 1
        # return subs, rels, objs

    
    # def get_batch(self, batch_idx, steps=2, data='train'):
    #     if data=='train':
    #         return np.array(self.train_data)[batch_idx]
    #     if data=='valid':
    #         query, answer = np.array(self.valid_q), np.array(self.valid_a)
    #     if data=='test':
    #         query, answer = np.array(self.test_q), np.array(self.test_a)

    #     subs = []
    #     rels = []
    #     objs = []
        
    #     subs = query[batch_idx, 0]
    #     rels = query[batch_idx, 1]
    #     objs = np.zeros((len(batch_idx), self.n_ent))
    #     for i in range(len(batch_idx)):
    #         objs[i][answer[batch_idx[i]]] = 1
    #     return subs, rels, objs
    
    def shuffle_train(self,):
        # fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        n_train = len(train_triple)
        rand_idx = np.random.permutation(n_train)
        train_triple = train_triple[rand_idx]

        # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        # self.fact_data = self.double_triple(fact_triple)
        self.train_data = np.array(train_triple)
        self.n_train = len(self.train_data)
        # self.load_graph(self.fact_data)
    
    def combine_dict(self, target, source):
        for key in source.keys():
            if key not in target.keys():
                target[key] = source[key]
            else:
                target[key] = target[key].union(source[key])
        return target

    # def shuffle_train(self,):
    #     fact_triple = np.array(self.fact_triple)
    #     train_triple = np.array(self.train_triple)
    #     all_triple = np.concatenate([fact_triple, train_triple], axis=0)
    #     n_all = len(all_triple)
    #     rand_idx = np.random.permutation(n_all)
    #     all_triple = all_triple[rand_idx]

    #     # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
    #     self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist())
    #     self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
    #     self.n_train = len(self.train_data)
    #     self.load_graph(self.fact_data)

