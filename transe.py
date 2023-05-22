import argparse
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from kgraph import Data
from kgraph import FB15k237, FB13, WN11
from kgraph import FB15k, WN18, WN18RR
from kgraph import DataIter
from kgraph import Predict
from kgraph.loss import MarginLoss

from torch.utils import data as torch_data

from kgraph.log import set_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())

logger = set_logger(f'TransE_{run_time}')

best_model_saved_path = f'./save_models/TransE_{run_time}.pth'
best_selftraining_model_saved_path = f'./save_models/TransE_{run_time}_self_training.pth'

def set_device(gpu):
    global device
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

def cal_rate(samples, source_samples):
    index = samples[:, 3]
    
    length = np.max(index) + 1
    
    
    first_sample = samples[index == 0, :3]
    if length > 1:
        second_sample = samples[index == 1, :3]
    end_index = np.where(index == 0)[0][-1]
    
    samples = samples[:end_index, :3]
    pos_set = {(h, r, t) for h, r, t in source_samples}
    first_set = {(h, r, t) for h, r, t in first_sample}
    if length > 1:
        second_set = {(h, r, t) for h, r, t in second_sample}
    new_set = {(h, r, t) for h, r, t in samples}
    
    first_num = len(first_sample) * 1.0
    if length > 1:
        second_num = len(second_sample) * 1.0
    else: 
        second_num = 0.
    new_num = len(samples) * 1.0
    
    first_percentage = len(list(first_set & pos_set)) / first_num
    if length > 1:
        second_percentage = len(list(second_set & pos_set)) / second_num
    else:
        second_percentage = 0.
    new_percentage = len(list(new_set & pos_set)) / new_num
    return samples, first_num, second_num, first_percentage, second_percentage, new_percentage

def get_graph(train_set, num_rel):
    pairs = set()
    graph = defaultdict(list)
    
    for triple in train_set:
        graph[(triple[0], triple[1])].append(triple[2])
        graph[(triple[2]), triple[1] + num_rel].append(triple[0])
        
        pairs.add((triple[0], triple[1]))
        pairs.add((triple[2], triple[1] + num_rel))
    
    return graph, pairs

class TempData(torch_data.Dataset):
    
    def __init__(self, data, orign_graph, orign_pairs, num_ent, num_rel):
        super(TempData, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        
        self.orign_graph = orign_graph
        self.orign_pairs = orign_pairs
        self.pairs = self.reset_pair(data)
    
    def reset_pair(self, data):
        _, pairs = get_graph(data, self.num_rel)
        return list(pairs)
    
    def __len__(self):
        len_ = len(self.pairs)
        assert len_ > 0, f'pairs <= 0, {len_}'
        return len_
    
    def __getitem__(self, index):
        pair = self.pairs[index]
        if pair in self.orign_pairs:
            pair_objs = self.orign_graph[pair]
        else: 
            pair_objs = []
        pair_label = torch.zeros(self.num_ent, dtype=torch.float)
        
        pair_label[pair_objs] = 1.
        
        pair = torch.LongTensor(list(pair))
        return pair, pair_label


class TransE(nn.Module):
    
    def __init__(self, num_ent: int, num_rel: int, embedding_dim: int, p: int=1, norm_flag=True, margin: float=5.0):
        super(TransE, self).__init__()
        
        self.p = 1
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim = embedding_dim
        self.norm_flag = norm_flag
        
        self.ent_embeddings = nn.Embedding(num_ent, embedding_dim)
        self.rel_embeddings = nn.Embedding(num_rel, embedding_dim)
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.criterion = MarginLoss(margin=margin)
    
    def embed_lookup(self, data):
        head = self.ent_embeddings(data[:, 0])
        rel = self.rel_embeddings(data[:, 1])
        tail = self.ent_embeddings(data[:, 2])
        return head, rel, tail
    
    def _calc(self, head, rel, tail):
        if self.norm_flag:
            head = F.normalize(head, p=2, dim=-1)
            rel = F.normalize(rel, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)
        score = head + rel - tail
        
        score = torch.norm(score, p=self.p, dim=-1).flatten()
        return score
    
    def regul(self):
        ent_weight = torch.norm(self.ent_embeddings.weight, p=self.p, dim=-1)
        rel_weight = torch.norm(self.rel_embeddings.weight, p=self.p, dim=-1)
        return (ent_weight + rel_weight) / 2
    
    def forward(self, data):
        head, rel, tail = self.embed_lookup(data)
        score = self._calc(head, rel, tail)
        return score
    
    def loss(self, data, label):
        data = torch.from_numpy(data).to(self.ent_embeddings.weight.data.device)
        label = torch.from_numpy(label).to(self.ent_embeddings.weight.data.device)
        score = self.forward(data)
        loss = self.criterion(score, label)
        return loss
    
    @torch.no_grad()
    def predict(self, data):
        data = torch.from_numpy(data).to(self.ent_embeddings.weight.data.device)
        # score = self.forward(data)
        # return score.cpu().numpy()
        mask = torch.arange(data.size(0))
        mask_tail = mask[data[:, 1] < self.num_rel]
        mask_head = mask[data[:, 1] >= self.num_rel]
        
        h = self.ent_embeddings(data[:, 0])
        r = self.rel_embeddings(data[:, 1] % self.num_rel)
        t = self.ent_embeddings.weight.data
        
        if self.norm_flag:
            h = F.normalize(h, p=2, dim=-1)
            r = F.normalize(r, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)
        
        h = h.view(-1, 1, self.dim)
        r = r.view(-1, 1, self.dim)
        t = t.view(1, -1, self.dim)
        
        score = h + r - t
        score[mask_head, :, :] = t + r[mask_head, :, :] - h[mask_head, :, :]
        
        score = torch.norm(score, self.p, -1)
        return score.cpu().numpy()
        
def load_data_from_KGraph(data_name):
    if data_name == 'FB15k237':
        return FB15k237()
    elif data_name == 'WN18RR':
        return WN18RR()
    elif data_name == 'wn18':
        return WN18()
    elif data_name == 'fb13':
        return FB13()
    elif data_name == 'wn11':
        return WN11()
    elif data_name == 'fb15k':
        return FB15k()
    else:
        raise ValueError('data name not found')

def get_dataiter(data, batch_size, num_negative, bern_flag=1):
    return DataIter(data, batch_size=batch_size, num_threads=16,
                    num_neg=num_negative, bern_flag=bern_flag,
                    element_type='triple')

def save_model(model, save_path='./chechpoints'):
    state = {
        'state_dict': model.cpu().state_dict()
    }
    
    torch.save(state, save_path)
    model.to(device)

def load_model(model, load_path):
    state = torch.load(load_path)
    state_dict = state['state_dict']
    model.cpu()
    model.load_state_dict(state_dict)
    model.to(device)


def generate_pseudo_samples(samples, model, orign_graph, orign_pairs, num_ent, num_rel, length_n=3):
    
    def calculate_score(data):
        with torch.no_grad():
            return 0 - model.predict(data.numpy())
    
    
    def filter_pseudo_label(pred, label, length=3):
        b_range = np.arange(pred.shape[0])
        label = np.around(label)
        pred = np.where(label.astype(np.bool_), -np.ones_like(pred) * 100000000, pred)

        obj_col, obj_row = np.nonzero(label)
        
        obj_index = np.argsort(-pred, axis=1)
        
        all_score = np.array([x[obj_index[i][:length+1]] for i, x in enumerate(pred)])
        all_score[:, :length] -= all_score[:, length].reshape(-1, 1)
        
        index = []
        replace_obj = []
        score = []
        
        for i in range(length):
            index.append(np.zeros_like(b_range) + i)
            replace_obj.append(obj_index[b_range, index[i]])
            score.append(all_score[b_range, index[i]])
        
        # print(index)
        index = np.concatenate(index, axis=0)
        replace_obj = np.concatenate(replace_obj, axis=0)
        score = np.concatenate(score, axis=0)

        return replace_obj, index, score
    
    temp_data = TempData(samples, orign_graph, orign_pairs, num_ent, num_rel)
    temp_data_iter = torch_data.DataLoader(temp_data, 10, shuffle=False, num_workers=16)
    model.eval()
    
    length_m = 1 if length_n <= 1 else length_n + 1
    
    new_labeled_samples = []
    triple_score = []
    new_pairs = []
    
    for batch_data, batch_label in temp_data_iter:
        new_triples = []
        pred_score = calculate_score(batch_data)
        pseudo_label, index, score = filter_pseudo_label(pred_score, batch_label.numpy(), length_n)
        
        
        batch_size = batch_data.shape[0]
        for i, idx in enumerate(index):
            j = i % batch_size
            h, r = batch_data[j].cpu().numpy()
            new_pairs.append((h, r))
            
            if r < num_rel:
                new_triples.append([h,r, pseudo_label[i], idx])
            else: 
                new_triples.append([pseudo_label[i], r - num_rel, h, idx])
        
        new_labeled_samples.append(np.asarray(new_triples))
        triple_score.append(score)
    
    new_labeled_samples = np.concatenate(new_labeled_samples, axis=0).astype(np.int32)
    triple_score = np.concatenate(triple_score, axis=0)
    
    triple_sort = np.argsort(-triple_score)
    triple_change_index = triple_sort[:len(triple_sort) // length_m]
    new_labeled_samples = new_labeled_samples[triple_change_index, :]
    new_pairs = list(set([new_pairs[i] for i in triple_change_index]))
    
    new_labeled_samples, fn, sn, fp, sp, nnp = cal_rate(new_labeled_samples, samples)
    
    logger.info(f'{fp:.4f}% of new samples {int(fn)} at first rank are pseudo-labeled samples')
    logger.info(f'{sp:.4f}% of new samples {int(sn)} at second rank are pseudo-labeled samples')
    logger.info(f'{nnp:.4f}% of new samples {len(new_labeled_samples)} are pseudo-labeled samples')
    
    return new_labeled_samples, new_pairs


def run(args):
    global best_model_saved_path
    logger.info(vars(args))
    data = load_data_from_KGraph(args.data)
    dataiter = get_dataiter(data, args.batch_size, args.neg, args.bern)
    model = TransE(data.num_ent, data.num_rel, args.dim, args.p, args.norm, args.margin).to(device)
    logger.info(model)
    logger.info(f'The best model save in: {best_model_saved_path}')
    opt = optim.SGD(model.parameters(), lr=args.lr)
    
    predict = Predict(data, element_type='pair')
    
    best_mrr = 0.0
    n_stop = 10
    
    if args.saved_path != 'null':
        args.epoch = 0
        best_model_saved_path = args.saved_path
    
    
    for epoch in trange(args.epoch):
        avg_loss = 0.0
        num_j = 0
        for batch_data, batch_label in dataiter.generate_triple_with_negative():
            
            opt.zero_grad()
            l = model.loss(batch_data, batch_label)
            l.backward()
            opt.step()
            l_value = l.item()
            avg_loss += l_value
            num_j += 1
        
        for batch_data, batch_label in dataiter.generate_triple_with_negative_on_random():
            # print(batch_data)
            opt.zero_grad()
            l = model.loss(batch_data, batch_label)
            l.backward()
            opt.step()
            l_value = l.item()
            avg_loss += l_value
            num_j += 1
        
        # print(f"After {epoch} epoch training, the loss is {l_value:.4f}, and the average loss is {(avg_loss / (epoch + 1)):.4f}")
        logger.info(f"After {epoch} epoch training, the loss is {l_value:.4f}, and the average loss is {(avg_loss / num_j):.4f}")
        
        # Early stop
        if epoch > 20:
            results = predict.predict_test(model.predict, 10)
            mrr = results[1]['avg_filtered']['mrr']
            logger.info(f"The results at {epoch} epoch:\n{results[0]}")
            if mrr < best_mrr:
                n_stop -= 1
                logger.info(f'n stop: {n_stop}.')
                if n_stop == 0:
                    break
                continue
            
            # print(f"The results at {epoch} epoch:")
            # print(results[0])
            n_stop = 30
            best_mrr = mrr
            save_model(model, best_model_saved_path)
            logger.info('Save the best model')
    
    # print(best_model_saved_path)
    
    load_model(model, best_model_saved_path)

    results = predict.predict_test(model.predict, 10)
    
    logger.info(f'The best results:\n {results[0]}')
    logger.info(f'\n{predict.predict_N2N(model.predict, 10)}')
    
def self_train(args):
    global best_selftraining_model_saved_path
    logger.info(vars(args))
    data = load_data_from_KGraph(args.data)
    
    orign_graph, orign_pairs = get_graph(data.train, data.num_rel)
    temp_data = Data(num_ent=data.num_ent, num_rel=data.num_rel)
    temp_data.valid = data.valid
    temp_data.test = data.test
    
    
    model = TransE(data.num_ent, data.num_rel, args.dim, args.p, args.norm, args.margin)
    logger.info(model)
    logger.info(f'The best model save in: {best_selftraining_model_saved_path}')
    
    if args.saved_path != 'null':
        load_model(model, args.saved_path)
        args.epoch = 1000
    else:
        load_model(model, best_model_saved_path)
    
    model.to(device)
    
    opt = optim.SGD(model.parameters(), lr=args.lr)
    
    predict = Predict(data, element_type='pair')
    
    best_mrr = 0.0
    n_stop = 30
    
    # if args.saved_path != 'null':
    #     args.epoch = 0
    #     best_selftraining_model_saved_path = args.saved_path
    
    # train the model by the algorithm 3
    for epoch in range(args.epoch):
        
        # generate the pseudo-labeled samples by algorithm 2
        new_pseudo_labeled_samples, new_pairs = generate_pseudo_samples(data.test, model, orign_graph, orign_pairs, data.num_ent, data.num_rel, args.length_n)
        
        temp_data.train = np.concatenate([data.train, new_pseudo_labeled_samples], axis=0)
        temp_data.update()
        dataiter = get_dataiter(temp_data, args.batch_size, args.neg, args.bern)


        avg_loss = 0.0
        num_j = 0
        for batch_data, batch_label in dataiter.generate_triple_with_negative():
            
            opt.zero_grad()
            l = model.loss(batch_data, batch_label)
            l.backward()
            opt.step()
            l_value = l.item()
            avg_loss += l_value
            num_j += 1
        
        for batch_data, batch_label in dataiter.generate_triple_with_negative_on_random():
            # print(batch_data)
            opt.zero_grad()
            l = model.loss(batch_data, batch_label)
            l.backward()
            opt.step()
            l_value = l.item()
            avg_loss += l_value
            num_j += 1
        
        # print(f"After {epoch} epoch training, the loss is {l_value:.4f}, and the average loss is {(avg_loss / (epoch + 1)):.4f}")
        logger.info(f"After {epoch} epoch training, the loss is {l_value:.4f}, and the average loss is {(avg_loss / num_j):.4f}")
        
        if epoch >= 0:
        
            results = predict.predict_test(model.predict, 10)
            logger.info(f"The results at {epoch} epoch:\n{results[0]}")
            mrr = results[1]['avg_filtered']['mrr']
            if mrr < best_mrr:
                n_stop -= 1
                if n_stop == 0:
                    break
                continue
            
            # print(f"The results at {epoch} epoch:")
            # print(results[0])
            n_stop = 30
            best_mrr = mrr
            save_model(model, best_selftraining_model_saved_path)
            logger.info('Save the best model')
    
    # print(best_selftraining_model_saved_path)
    
    load_model(model, best_selftraining_model_saved_path)

    results = predict.predict_test(model.predict, 10)
    
    logger.info(f'The best results:\n {results[0]}')
    logger.info(f'\n{predict.predict_N2N(model.predict, 10)}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data', type=str, default='FB15k237', help='dataset')
    parser.add_argument('--dim', type=int, default=200, help='embedding dimension')
    parser.add_argument('--p', type=int, default=1, help='norm')
    parser.add_argument('--margin', type=float, default=5.0, help='margin')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--neg', type=int, default=25, help='negative sample')
    parser.add_argument('--bern', type=int, default=1, help='bern flag')
    parser.add_argument('--norm', type=int, default=1, help='norm flag')
    parser.add_argument('--save', type=str, default='transe', help='save model')
    parser.add_argument('--load', type=str, default=None, help='load model')
    parser.add_argument('--test', type=int, default=0, help='test flag')
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
    parser.add_argument('--length_n', type=int, default=3, help='the length n')
    # parser.add_argument('--predict', action='')
    parser.add_argument('--saved_path', type=str, default='null', help='The save path of the best model.')
    args = parser.parse_args()
    
    run(args)
    self_train(args)
    











