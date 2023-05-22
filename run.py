import apex
import os, sys
import time
import argparse
import numpy as np
import torch

from scipy import sparse
from collections import defaultdict
from yacs.config import CfgNode as CN
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import *
from kgraph import FB15k237, WN18RR, Data
from kgraph import DataIter, Predict
from kgraph.log import set_logger

from torch.utils.data import Dataset, DataLoader


def update(params):
    
    C = CN()                    ##
    C.DATA = CN()
    C.DATA.NAME = 'FB15k237'
    C.DATA.SMOOTH = 0.05

    C.TRAIN = CN()
    C.TRAIN.BATCH_SIZE = 32
    C.TRAIN.NUM_WORKERS = 10
    C.TRAIN.MAX_EPOCH = 1000
    C.TRAIN.LR = 0.0005
    C.TRAIN.WEIGHT_DECAY = 0.0
    C.TRAIN.SEED = 41504
    C.TRAIN.GPU = 0
    C.TRAIN.DEBUG = False
    C.TRAIN.LOG_STEP = 100
    C.TRAIN.EARLY_STOP_CNT = 30
    C.TRAIN.USE_APEX = True
    C.TRAIN.LENGTH = 3

    C.ENCODER = CN()
    C.ENCODER.NAME = 'CompGCN'
    C.ENCODER.EMBED_DIM = 200
    C.ENCODER.NUM_BASES = -1
    C.ENCODER.ENCODER_DROP1 = 0.3
    C.ENCODER.ENCODER_DROP2 = 0.3
    C.ENCODER.ENCODER_GCN_DROP = 0.1
    C.ENCODER.ENCODER_GCN_BIAS = False
    C.ENCODER.GCN_HIDDEN_CHANNELS = 200
    C.ENCODER.GCN_IN_CHANNELS = 200
    C.ENCODER.GCN_OUT_CHANNELS = 200
    C.ENCODER.GCN_NUM_LAYERS = 2
    C.ENCODER.GCN_OPN = 'sub'
    C.ENCODER.BIAS = False

    C.DECODER = CN()
    C.DECODER.NAME = 'ConvE'
    C.DECODER.GAMMA = 40
    C.DECODER.K_W = 10
    C.DECODER.K_H = 10
    C.DECODER.KER_SZ = 7
    C.DECODER.DECODER_FEAT_DROP = 0.3
    C.DECODER.DECODER_HID_DROP = 0.3
    C.DECODER.NUM_FILTERS = 200
    C.DECODER.CONVE_BIAS = False
    
    cfg = C.clone()
    params = vars(params)
    cfg.defrost()
    
    if params['pretrain_path'] != '':
        path = os.path.join(os.getcwd(), 'checkpoints', params['pretrain_path'])
        name = params['pretrain_path'].split('_')[:2]
        if not params['use_magic']:
            name = '_'.join(name + [params['data']]) + '.yaml'
        else: 
            name = '_'.join(name + [params['data']] + ['MAGIC']) + '.yaml'
        
        cfg.merge_from_file(os.path.join(path, name))
    else:
        for k in list(cfg):
            temp = params.get(k.lower(), None)
            next_cfg = getattr(cfg, k)
            if temp is not None and temp != getattr(next_cfg, 'NAME'):
                setattr(next_cfg, 'NAME', temp)
            for i in list(next_cfg):
                tmp = params.get(i.lower(), None)
                if tmp is not None and tmp != getattr(next_cfg, i):
                    setattr(next_cfg, i, tmp)
    cfg.freeze()
    return cfg

def save_config(cfg, path):                         
    path = os.path.join(os.getcwd(), path)
    with open(path, 'w') as f:
        f.write(cfg.dump())

def set_gpu(gpu):               
    
    """
    Sets the GPU to be used for the run.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def set_seed(seed):
    """
    Sets the seed.
    """
    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True 


def set_device(gpu):
    '''                                                             
    Sets the device to be used for the run.
    '''
    
    if gpu != -1 and torch.cuda.is_available():
        set_gpu(gpu)
        return torch.device('cuda:' + str(gpu))
    else:
        return torch.device('cpu')


def load_data(data_name, data_path=None):
    '''
    Sets the data to be used for the run.                   
    '''
    data_name = data_name.lower()
    assert data_name in ['fb15k237', 'wn18rr'], 'Invalid data name.'
    
    if data_name == 'fb15k237':
        data = FB15k237(data_path)
    else:
        data = WN18RR(data_path)
    return data


def generate_matrix(data, data_num_ent, data_num_rel):              
    
    ent_pairs = set()
    ent_pair_id = {}
    
    graph_matrix = []
    attributes = []
    
    tmp_id = 0
    for h, r, t in data:
        if (h, t) not in ent_pairs:
            graph_matrix.append([h, t, tmp_id])
            ent_pair_id[(h, t)] = tmp_id
            ent_pairs.add((h, t))
            attributes.append([tmp_id, r])
            tmp_id += 1
        else:
            attributes.append([ent_pair_id[(h, t)], r])
    
    for i in range(data_num_ent):
        graph_matrix.append([i, i, tmp_id+i])
        attributes.append([tmp_id+i, 2*data_num_rel+i])
        
    
    graph_matrix = np.array(graph_matrix)
    
    matrix = {'indices': graph_matrix[:, [0, 1]].T,
              'values': graph_matrix[:, 2]}
    
    # return matrix, np.array(attributes)
    return {'indices': torch.LongTensor(matrix["indices"]),
            'values': torch.LongTensor(matrix["values"])}, torch.LongTensor(attributes)


def show_params(params):
    
    msg = '\nParameters:\n'
    
    for key, value in params.items():
        msg += '\t{}: {}\n'.format(key, value)
    return msg

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

def generate_new_tail(pred, label, length=3):                                       
    """ 
    :param pred: predicted entity
    :param label: label
    :return: entity
    """
    
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
    
    index = np.concatenate(index, axis=0)
    replace_obj = np.concatenate(replace_obj, axis=0)
    score = np.concatenate(score, axis=0)

    return replace_obj, index, score

def generate_graph(data, num_rel):                  
    pairs = set()
    graph = defaultdict(list)
    
    for triple in data:
        graph[(triple[0], triple[1])].append(triple[2])
        graph[(triple[2]), triple[1] + num_rel].append(triple[0])
        
        pairs.add((triple[0], triple[1]))
        pairs.add((triple[2], triple[1] + num_rel))
    
    return graph, pairs

class TempData(Dataset):                                
    
    def __init__(self, data, orign_graph, orign_pairs, num_ent, num_rel):
        super(TempData, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        
        self.orign_graph = orign_graph
        self.orign_pairs = orign_pairs
        self.pairs = self.reset_pair(data)
    
    def reset_pair(self, data):                                 
        _, pairs = generate_graph(data, self.num_rel)
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


class SemiData(Dataset):                        
    
    def __init__(self, new_triples, new_pairs, orign_triples, orign_pairs, num_ent, num_rel, smooth=0.05):
        super(SemiData, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.smooth = smooth
        
        data = np.concatenate([new_triples, orign_triples], axis=0)
        
        self.graph, _ = generate_graph(data, self.num_rel)
        self.pairs = list(orign_pairs | set(new_pairs))
    
    def __len__(self):
        len_ = len(self.pairs)
        assert len_ > 0, f'pairs <= 0, {len_}'
        return len_
    
    def __getitem__(self, index):
        pair = self.pairs[index]
        pair_objs = self.graph[pair]
        pair_label = torch.zeros(self.num_ent, dtype=torch.float)
        pair_label += 1.0 / self.num_ent
        
        pair_label[pair_objs] = 1. - self.smooth + 1.0 / self.num_ent
        
        pair = torch.LongTensor(list(pair))
        return pair, pair_label


class Run(object):
    
    def __init__(self, params):             
        '''
        Constructor of the runner class.
        '''
        
        params.embed_dim = params.encoder_conve_k_w * \
            params.encoder_conve_k_h if params.embed_dim is None else params.embed_dim     
        
        self.p = update(params)
        # self.use_magic = params.use_magic
        self.pre_train_step = params.pre_train_step
        run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())
        name = f'{self.p.ENCODER.NAME}_{self.p.DECODER.NAME}_{run_time}'
        self.logger = set_logger(name)
        self.gamma = self.p.DECODER.GAMMA 
        
        self.restore = False
        if params.pretrain_path != '':
            path = os.path.join(os.getcwd(), 'checkpoints', params.pretrain_path)
            name = params.pretrain_path.split('_')[:2]
            name.append(self.p.DATA.NAME)
            if params.use_magic:
                self.save_path = os.path.join(path, '_'.join(name) + '_MAGIC' + '.pth')
            else:
                self.save_path = os.path.join(path, '_'.join(name) + '.pth')
            self.restore = True
        
        set_seed(self.p.TRAIN.SEED)
        self.device = set_device(self.p.TRAIN.GPU)
        
        self.load_data()
        
        self.model = self.add_model(self.p.ENCODER.NAME, self.p.DECODER.NAME)
        self.optimizer = self.add_optimizer(self.model.parameters(), self.p.TRAIN.LR, self.p.TRAIN.WEIGHT_DECAY)
        
        self.use_apex = params.use_apex
        if params.use_apex:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level="O1")
        
        self.logger.info('\n' + str(self.p))
    
    
    def load_data(self):
        '''                                             
        Loads data.
        '''
        self.data = load_data(self.p.DATA.NAME, None)
        self.kgraph_predict = Predict(self.data, element_type='pair')
        self.num_ent = self.data.num_ent
        self.num_rel = self.data.num_rel
        self.dataiter = DataIter(self.data, self.p.TRAIN.BATCH_SIZE,
                                 num_threads=self.p.TRAIN.NUM_WORKERS,
                                 smooth_lambda=self.p.DATA.SMOOTH,
                                 element_type='pair')
        
        self.edge_index, self.edge_type, self.matrix, self.attributes = self.construct_adj()
        
        train_data = self.data.train
        rel_head = train_data[:, [1, 0]]
        rel_head[:, 0] += self.data.num_rel
        rel_tail = train_data[:, [1, 2]]
        rel_tails = np.unique(np.concatenate((rel_head, rel_tail), axis=0), axis=0).T
        
        row = rel_tails[0]
        col = rel_tails[1]
        data = np.ones_like(col).astype(np.float32)
        
        rel_tails = sparse.coo_matrix((data, (row, col)), shape=(2*self.data.num_rel, self.data.num_ent)).todense()
        self.tmp_rel_tails = np.asarray(rel_tails)
        
        self.rel_tails = None
        self.set_semi_supervised()
        self.set_temp_data()
    
    def set_semi_supervised(self):              
        self.semi_supervised_data = None
    
    def reset_ss_train_data(self, new_triples, new_pairs, smooth_lambda=0.0):                       
        self.semi_supervised_data = SemiData(new_triples, new_pairs, self.data.train, self.orign_pairs,
                                             self.num_ent, self.num_rel, smooth_lambda)
        # print(len(self.semi_supervised_data))
    
    def get_batch_from_ss_data(self):                                   
        return DataLoader(dataset=self.semi_supervised_data, batch_size=self.p.TRAIN.BATCH_SIZE, shuffle=True, num_workers=self.p.TRAIN.NUM_WORKERS)

    def set_temp_data(self):            
        self.temp_data = self.data
        self.orign_graph, self.orign_pairs = generate_graph(self.data.train, self.num_rel)
    
    def reset_temp_pair(self, data):                        
        self.temp_data = TempData(data, self.orign_graph, self.orign_pairs, self.num_ent, self.num_rel)
    
    def get_batch_from_temp(self):                                                 
        return DataLoader(dataset=self.temp_data, batch_size=self.p.TRAIN.BATCH_SIZE, shuffle=False, num_workers=self.p.TRAIN.NUM_WORKERS)

    def generate_new_pos_samples(self, pos_samples):                            
        def function(data):
            with torch.no_grad():
                pred = self.model.predict(data.to(self.device)).cpu().numpy()
                # print('pred:\n', pred)
                return pred
        
        begin_generate_sample = time.time()
        
        # self.reset_ss_train_data(pos_samples, 0.0)
        self.reset_temp_pair(pos_samples)
        
        self.model.eval()
        
        length_m = 1 if self.p.TRAIN.LENGTH else self.p.TRAIN.LENGTH + 1
        
        # new_pos_samples = [pos_samples]
        new_pos_samples = []
        triple_score = []
        new_pairs = []
        
        print(self.p.TRAIN.LENGTH)
        
        for batch_data, batch_label in self.get_batch_from_temp():
            new_triples = []
            pred = function(batch_data)
            new_tails, index, score = generate_new_tail(pred, batch_label.numpy(), self.p.TRAIN.LENGTH)
            

            batch_size = batch_data.shape[0]
            for i, idx in enumerate(index):
                j = i % batch_size
                h, r = batch_data[j].cpu().numpy()
                new_pairs.append((h, r))
                if r < self.num_rel:
                    new_triples.append([h, r, new_tails[i], idx])
                else:
                    new_triples.append([new_tails[i], r - self.num_rel, h, idx])

            new_pos_samples.append(np.asarray(new_triples))
            triple_score.append(score)
        new_pos_samples = np.concatenate(new_pos_samples, axis=0).astype(np.int32)
        triple_score = np.concatenate(triple_score, axis=0)
        
        triple_sort = np.argsort(-triple_score)
        triple_change_index = triple_sort[:len(triple_sort) // length_m]
        new_pos_samples = new_pos_samples[triple_change_index, :]
        new_pairs = list(set([new_pairs[i] for i in triple_change_index]))
        end_generate_sample = time.time()
        
        new_pos_samples, fn, sn, fp, sp, nnp = cal_rate(new_pos_samples, pos_samples)
        
        self.logger.info(f'Generate {len(new_pos_samples)} new samples by {(end_generate_sample - begin_generate_sample):.2f}s')
        self.logger.info(f'{fp:.4f}% of new samples {int(fn)} at first rank are positive samples')
        self.logger.info(f'{sp:.4f}% of new samples {int(sn)} at second rank are positive samples')
        self.logger.info(f'{nnp:.4f}% of new samples {len(new_pos_samples)} are positive samples')
        return new_pos_samples, new_pairs

    def construct_adj(self):
        '''                                                                 
        calculates the adjacency matrix.
        '''
        train_data = self.data.train
        inv_train_data = train_data[:, [2, 1, 0]]
        inv_train_data[:, 1] += self.num_rel
        train_data = np.concatenate([train_data, inv_train_data], axis=0)
        
        matrix, attributes = generate_matrix(train_data, self.num_ent, self.num_rel)
        
        edge_index = torch.from_numpy(train_data[:, [0, 2]].T).long().to(self.device)
        edge_type = torch.from_numpy(train_data[:, 1]).long().to(self.device)
        
        return edge_index, edge_type, matrix, attributes
    
    def add_model(self, encoder, decoder):
        '''                                                 
        Creates the model.
        '''
        
        model_name = '{}_{}'.format(encoder, decoder)
        self.model_name = model_name
        model_name = model_name.lower()
        
        if model_name == 'embed_transe':
            model = Embed_TransE(self.p, self.num_ent, self.num_rel, self.edge_index, self.edge_type, self.device)
        elif model_name == 'embed_distmult':
            model = Embed_DistMult(self.p, self.num_ent, self.num_rel, self.edge_index, self.edge_type, self.device)
        elif model_name == 'embed_conve':
            model = Embed_ConvE(self.p, self.num_ent, self.num_rel, self.edge_index, self.edge_type, self.device)
        else:
            raise NotImplementedError('Invalid model name.')
        
        self.logger.info(model)
        return model.to(self.device)
    
    def add_optimizer(self, parameters, lr, weight_decay):      
        '''
        Creates the optimizer for training the parameters of the model.
        '''
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    
    def save_model(self, save_path):                
        device = self.model.device
        state = {
            'state_dict': self.model.cpu().state_dict(),
            'best_valid': self.best_valid,
            'best_epoch': self.best_epoch,
        }
        torch.save(state, save_path)
        self.model.to(device)

    def load_model(self, load_path):                    
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_valid = state['best_valid']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.optimizer = self.add_optimizer(self.model.parameters(), self.p.TRAIN.LR, self.p.TRAIN.WEIGHT_DECAY)
    
    def evaluate(self, data_type, epoch=None):                  
        def function(data):
            with torch.no_grad():
                pred = self.model.predict(data).cpu().numpy()
                # return 2 - pred - self.rel_tails[data[:, 1], :]
                # return - self.rel_tails[data[:, 1], :]
                if self.rel_tails is None:
                    return 1 - pred
                return - (pred * self.rel_tails[data[:, 1], :])
                # return 1 - pred
        self.model.eval()
        
        if data_type == 'valid':
            table, results = self.kgraph_predict.predict_valid(function, 512)
        else:
            table, results = self.kgraph_predict.predict_test(function, 512)
        
        if epoch is not None:
            self.logger.info('Epoch: {}'.format(epoch))
        self.logger.info('\n' + table + '\n')
        
        return results['avg_filtered']
    

    def per_epoch(self, epoch):                         
        
        self.model.train()
        losses = []
        
        per_epoch_time = time.time()
        
        for step, (batch, label) in enumerate(self.dataiter.generate_pair()):
            forward_time_start = time.time()
            self.optimizer.zero_grad()
            pred= self.model(batch)
            loss = self.model.loss(pred, label)
            
            forward_time_end = time.time()
            forward_time = forward_time_end - forward_time_start
            backward_time_start = forward_time_end
            
            if self.use_apex:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())
            backward_time_end = time.time()
            backward_time = backward_time_end - backward_time_start
            
            if step == 0:
                self.logger.info('Epoch: {} Step: {} [Forward: {:.1f}s; Backward: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time,
                                    backward_time,
                                    np.mean(losses)))
            
            elif step > 0 and step % self.p.TRAIN.LOG_STEP == 0:
                self.logger.info('Epoch: {} Step: {} [Forward: {:.1f}s; Backward: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time * self.p.TRAIN.LOG_STEP,
                                    backward_time * self.p.TRAIN.LOG_STEP,
                                    np.mean(losses)))
            else:
                continue
        
        per_epoch_time = time.time() - per_epoch_time
        per_epoch_time = time.strftime("%M:%S", time.localtime(per_epoch_time))
        self.logger.info('Epoch: {} [Time: {}] Loss: {:.5f}'.format(epoch, per_epoch_time, np.mean(losses)))
        return np.mean(losses)
    
    def semi_supervised_per_train(self, epoch):                 
        losses = []
        
        new_pos_data, new_pairs = self.generate_new_pos_samples(self.data.test)
        self.reset_ss_train_data(new_pos_data, new_pairs, self.p.DATA.SMOOTH)
        
        # exit(1)
        
        self.model.train()
        
        per_epoch_time = time.time()
        for step, (batch, label) in enumerate(self.get_batch_from_ss_data()):
            forward_time_start = time.time()
            
            batch = batch.to(self.device)
            label = label.to(self.device)
            if batch.size(0) <= 1:
                continue
            self.optimizer.zero_grad()
            pred= self.model(batch)
            loss = self.model.loss(pred, label)
            
            forward_time_end = time.time()
            forward_time = forward_time_end - forward_time_start
            backward_time_start = forward_time_end
            
            if self.use_apex:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())
            backward_time_end = time.time()
            backward_time = backward_time_end - backward_time_start
            
            
            if step == 0:
                self.logger.info('Epoch: {} Step: {} semi [Forward: {:.1f}s; Backward: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time,
                                    backward_time,
                                    np.mean(losses)))
            
            elif step > 0 and step % self.p.TRAIN.LOG_STEP == 0:
                self.logger.info('Epoch: {} Step: {} semi [Forward: {:.1f}s; Backward: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time * self.p.TRAIN.LOG_STEP,
                                    backward_time * self.p.TRAIN.LOG_STEP,
                                    np.mean(losses)))
            else:
                continue
        
        per_epoch_time = time.time() - per_epoch_time
        per_epoch_time = time.strftime("%M:%S", time.localtime(per_epoch_time))
        self.logger.info('Epoch: {} semi [Time: {}] Loss: {:.5f}'.format(epoch, per_epoch_time, np.mean(losses)))
        return np.mean(losses)
        
    
    def train(self):              
        
        self.best_valid, self.best_epoch = 0, 0
        self.best_hits = 0.
        run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())
        dir_path = os.path.join('./checkpoints', self.model_name + run_time)
        os.makedirs(name=dir_path, exist_ok=True)
        self.save_path = os.path.join(dir_path, self.model_name+ '_' + self.p.DATA.NAME + '.pth')
        
        if self.restore:
            self.load_model(self.save_path)
            self.logger.info('Successfully loaded previous model.')
        
        epochs = self.p.TRAIN.MAX_EPOCH
        if self.p.TRAIN.DEBUG:
            epochs = 5
        
        kill_cnt = 0
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=30, eta_min=0.00005, last_epoch=-1)
        
        for epoch in range(epochs):
            # print('epoch:', epoch)
            # if self.use_magic and epoch % (1 + np.around(self.pre_train_step / (epoch + 1))) == 0:
            #     train_loss = self.semi_supervised_per_train(epoch)
            # else: 
            #     train_loss = self.per_epoch(epoch)
            train_loss = self.per_epoch(epoch)
            scheduler.step()
            
            # torch.cuda.empty_cache()
            self.model.save_embeddings()
            val_results = self.evaluate('test', epoch)
            
            if val_results['mrr'] > self.best_valid or val_results['hits@10'] > self.best_hits:
                if val_results['mrr'] > self.best_valid:
                    self.best_valid = val_results['mrr']
                else: 
                    self.best_hits = val_results['hits@10']
                self.best_epoch = epoch
                self.save_model(self.save_path)
                kill_cnt = 0
                self.logger.info('The best model of epoch {} have save in:\n {}.'.format(epoch, self.save_path))
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.gamma> 5:
                    self.gamma-= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.gamma))
                
                if kill_cnt > self.p.TRAIN.EARLY_STOP_CNT:
                    self.logger.info('Early stopping!!')
                    break
        
        self.logger.info('Loading best model from the epoch {}..., Evaluating on Test Data!'.format(self.best_epoch))
        self.logger.info('Best model is saved in:\n {}'.format(self.save_path))
        save_config(self.p, os.path.join(dir_path, self.model_name+ '_' + self.p.DATA.NAME + '.yaml'))
    
    def self_train(self):                                    
        
        self.best_valid, self.best_epoch = 0, 0
        self.best_hits = 0.
        # run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())
        # dir_path = os.path.join('./checkpoints', self.model_name + run_time)
        # os.makedirs(name=dir_path, exist_ok=True)
        # self.save_path = os.path.join(dir_path, self.model_name+ '_' + self.p.DATA.NAME + '.pth')
        
        if self.save_path[-9:-4] == 'MAGIC':
            self.load_model(self.save_path[:-10] + '.pth')
        else: 
            self.load_model(self.save_path)
        self.logger.info(f'Successfully loaded previous model from:\n {self.save_path}.')
        
        dir_path = os.path.join(*self.save_path.split('/')[-3:-1])
        self.save_path = os.path.join(dir_path, self.model_name+ '_' + self.p.DATA.NAME + '_MAGIC' + '.pth')
        
        # self.save_model(self.save_path)
        
        epochs = self.p.TRAIN.MAX_EPOCH
        if self.p.TRAIN.DEBUG:
            epochs = 5
        
        kill_cnt = 0

        scheduler = CosineAnnealingLR(self.optimizer, T_max=30, eta_min=0.00005, last_epoch=-1)
       
        for epoch in range(epochs):
            train_loss = self.semi_supervised_per_train(epoch)
            scheduler.step()

            self.model.save_embeddings()
            val_results = self.evaluate('test', epoch)
            
            if val_results['mrr'] > self.best_valid or val_results['hits@10'] > self.best_hits:
                if val_results['mrr'] > self.best_valid:
                    self.best_valid = val_results['mrr']
                else: 
                    self.best_hits = val_results['hits@10']
                self.best_epoch = epoch
                self.save_model(self.save_path)
                kill_cnt = 0
                self.logger.info('The best model of epoch {} have save in:\n {}.'.format(epoch, self.save_path))
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.gamma> 5:
                    self.gamma-= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.gamma))
                
                if kill_cnt > self.p.TRAIN.EARLY_STOP_CNT:
                    self.logger.info('Early stopping!!')
                    break
        
        self.logger.info('Loading best model from the epoch {}..., Evaluating on Test Data!'.format(self.best_epoch))
        self.logger.info('Best model is saved in:\n {}'.format(self.save_path))
        save_config(self.p, os.path.join(dir_path, self.model_name+ '_' + self.p.DATA.NAME+ '_MAGIC' + '.yaml'))
    
    def predict(self, save_path=None):
        
        def set_rel_tails(alpha):
            self.rel_tails = self.tmp_rel_tails * alpha + (1- self.tmp_rel_tails) * (1 - alpha)
        
        def find_best_parameters():
            alpha = -0.05
            best = {'alpha': alpha + 0.5, 'mrr': 0.}
            for i in range(10):
                alpha += 0.01
                set_rel_tails(alpha + 0.5)
                results = self.evaluate('test')
                if results['mrr'] > best['mrr']:
                    best['alpha'] = alpha + 0.5
                    best['mrr'] = results['mrr']
            return best['alpha']
        
        if save_path is None:
            save_path = self.save_path
        
        self.load_model(save_path)
        self.model.eval()
        set_rel_tails(find_best_parameters())
        self.evaluate('test')
        self.logger.info('Prediction finished!')
        self.logger.info('The prediction results are saved in:\n {}'.format(save_path))
        self.save_model(save_path)
        
        table = self.evaluateN2N()
        self.logger.info('The results of N2N:\n' + table)
        
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run the models on the benchmark datasets.")
    
    parser.add_argument('-data', type=str, default='FB15k237', help='Name of the dataset.', choices=['FB15k237', 'WN18RR'])
    parser.add_argument('-use_apex', action='store_true')
    parser.add_argument('-predict', action='store_true')
    parser.add_argument('-pretrain_path', type=str, default='', help='the save path of the pretrained model.')
    parser.add_argument('-pre_train_step', type=int, default=100, help='')
    
    # ########################################################################################################################
    # Hyperparameters
    # ########################################################################################################################
    parser.add_argument('-use_magic', action='store_true', help='Use the magic semi-superised for training')
    parser.add_argument('-length', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('-max_epoch', type=int, default=1, help='Number of epochs.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('-seed', type=int, default=41504, help='Seed.')
    parser.add_argument('-gpu', type=int, default=0, help='GPU to use.')
    parser.add_argument('-log_step', type=int, default=100, help='Log step.')
    parser.add_argument('-debug', action='store_true', help='Debug mode.')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('-early_stop_cnt', type=int, default=30, help='Early stopping count.')
    
    parser.add_argument('-num_ent', type=int, default=0, help='Number of entities.')
    parser.add_argument('-num_rel', type=int, default=0, help='Number of relations.')
    
    
    
    # ########################################################################################################################
    # Encoder parameters
    # ########################################################################################################################
    parser.add_argument('-encoder', type=str, default='Embed', help='Name of the encoder.')
    parser.add_argument('-embed_dim', type=int, default=200, help='Embedding dimension.')
    parser.add_argument('-num_bases', type=int, default=-1, help='Number of bases.')
    
    parser.add_argument('-encoder_drop1', type=float, default=0.3, help='Dropout rate for the first layer of the encoder.')
    parser.add_argument('-encoder_drop2', type=float, default=0.3, help='Dropout rate for the second layer of the encoder.')
    parser.add_argument('-encoder_gcn_drop', type=float, default=0.1, help='Dropout rate for the GCN layer of the encoder.')
    parser.add_argument('-encoder_gcn_bias', action='store_true', help='Whether to add bias to the GCN layer of the encoder.')
    
    # parser.add_argument('-gcn_hidden_channels', type=int, default=200, help='Hidden channels of the GCN layer.')
    # parser.add_argument('-gcn_in_channels', type=int, default=200, help='Input channels of the GCN layer.')
    # parser.add_argument('-gcn_out_channels', type=int, default=200, help='Output channels of the GCN layer.')
    # parser.add_argument('-gcn_num_layers', type=int, default=2, help='Number of GCN layers.')
    # parser.add_argument('-gcn_opn', type=str, default='sub', help='The function in aggregation')
    
    
    
    
    
    
    
    # ########################################################################################################################
    # Decoder parameters
    # ########################################################################################################################
    parser.add_argument('-decoder', type=str, default='ConvE', help='Name of the decoder.', choices=['TransE', 'DistMult', 'ConvE'])
    
    # TransE parameters
    parser.add_argument('-gamma', type=int, default=40, help='Gamma value for TransE.')
    
    
    # ConvE parameters
    parser.add_argument('-k_w', type=int, default=10, help='Convolution width.')
    parser.add_argument('-k_h', type=int, default=20, help='Convolution height.')
    parser.add_argument('-ker_sz', type=str, default='7', help='Convolution kernel size.')
    parser.add_argument('-decoder_feat_drop', type=float, default=0.3, help='Dropout rate for the decoder.')
    parser.add_argument('-decoder_hid_drop', type=float, default=0.3, help='Dropout rate for the decoder.')
    parser.add_argument('-num_filters', type=int, default=200, help='Number of filters.')
    parser.add_argument('-smooth', type=float, default=5e-2, help='Smooth factor.')
    parser.add_argument('-conve_bias', action='store_true', help='Use bias in the ConvE layer.')
    
    

    
    
    
    
    args = parser.parse_args()
    
    # print(args)
    
    runner = Run(args)
    
    if args.predict:
        assert args.pretrain_path != '', 'The pretrained model is needed for the magic semi-supervised setting.'
        runner.predict()
    else:
        if args.use_magic:
            assert args.pretrain_path != '', 'The pretrained model is needed for the magic semi-supervised setting.'
            runner.self_train()
        else:
            runner.train()
            runner.self_train()
        