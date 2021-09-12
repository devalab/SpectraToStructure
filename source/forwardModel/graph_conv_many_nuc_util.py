import torch

import netdataio
import pickle
import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
class ModelTrial():
    def __init__(self, training_loader, 
                 validation_loader, hparams):
        self.learning_rate = hparams['learning_rate']
        self.my_batch_size = hparams['batch_size']

        # something


    def batch_size(self):
        return self.my_batch_size
    
    def build_model(self, hparams):
        """
        Return the model 
        """
        net = nets.GraphVertExtraLinModel( g_feature_n,[INT_D] * LAYER_N, resnet=USE_RESNET, 
                                           noise=INIT_NOISE, agg_func=AGG_FUNC, GS=GS, 
                                           OUT_DIM = len(tgt_nucs))
        #data_feat = ['vect']

        


    def optimizer(self, model):
        """
        return the optimizer
        """
        
        return torch.optim.Adam(model.parameters(),
                                lr=self.learning_rate)
    def loss(self):
        """
        Return a loss function that takes in two tensors
        """

        
    def training_metrics(self):
        """
        """
        return {
            'classification_Error' : function, 
            'nll_loss' : torch.nn.functional.nll_loss
        
        }

    def validation_metrics(self):
        """
        """
        

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]

### Create datasets and data loaders

default_feat_vect_args = dict(feat_atomicno_onehot=default_atomicno, 
                              feat_pos=False, feat_atomicno=True,
                              feat_valence=True, aromatic=True, hybridization=True, 
                              partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
                              r_covalent=False,
                              total_valence_onehot=True, 
                              
                              r_vanderwals=False, default_valence=True, rings=True)

default_feat_mat_args = dict(feat_distances = False, 
                             feat_r_pow = None)

default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(edge_weighted=False, 
                        norm_adj=True, add_identity=True, 
                        split_weights=default_split_weights)


DEFAULT_DATA_HPARAMS = {'feat_vect_args' : default_feat_vect_args, 
                       'feat_mat_args' : default_feat_mat_args, 
                       'adj_args' : default_adj_args}

def dict_combine(d1, d2):
    d1 = copy.deepcopy(d1)
    d1.update(d2)
    return d1


def make_datasets(exp_config, hparams, train_sample=0):
    """
    """


    d = pickle.load(open(exp_config['filename'], 'rb'))
    train_df = d['train_df']
    if train_sample > 0:
        print("WARNING, subsampling training data to ", train_sample)
        train_df = train_df.sample(train_sample, random_state=0) 

    tgt_nucs = d['tgt_nucs']
    test_df = d['test_df'] #.sample(10000, random_state=10)

    MAX_N = d['MAX_N']

    datasets = {}

    for phase, df in [('train', train_df), 
                      ('test', test_df)]:
                       

        ds = netdataio.MoleculeDatasetMulti(df.rdmol.tolist(), 
                                            df.value.tolist(),  
                                            MAX_N, len(tgt_nucs), 
                                            hparams['feat_vect_args'], 
                                            hparams['feat_mat_args'], 
                                            hparams['adj_args'])        
        datasets[phase] = ds

        

    return datasets['train'], datasets['test']


def create_checkpoint_func(every_n, filename_str):
    def checkpoint(epoch_i, net, optimizer):
        if epoch_i % every_n > 0:
            return {}
        checkpoint_filename = filename_str.format(epoch_i = 0)
        t1 = time.time()
        torch.save(net.state_dict(), checkpoint_filename + ".state")
        torch.save(net, checkpoint_filename + ".model")
        t2 = time.time()
        return {'savetime' : (t2-t1)}
    return checkpoint

def run_epoch(net, optimizer, criterion, dl, 
               pred_only = False, USE_CUDA=True,
               return_pred = False, desc="train", 
              print_shapes=False, progress_bar=True):
    t1_total= time.time()

    accum_pred = []
    running_loss = 0.0
    total_points = 0
    total_compute_time = 0.0
    if progress_bar:
        iterator =  tqdm(enumerate(dl), total=len(dl), desc=desc)
    else:
        iterator = enumerate(dl)

    for i_batch, (adj, vect_feat, mat_feat, vals, mask) in \
            iterator:
        t1 = time.time()
        if print_shapes:
            print("adj.shape=", adj.shape, 
                  "vect_feat.shape=", vect_feat.shape, 
                  "mat_feat.shape=", mat_feat.shape, 
                  "vals.shape=", vals.shape, 
                  "mask.shape=", mask.shape)

        if not pred_only:
            optimizer.zero_grad()

        adj_t = move(adj, USE_CUDA)
        vect_feat_t = move(vect_feat, USE_CUDA)
        mat_feat_t = None
        mask_t = move(mask, USE_CUDA)
        vals_t = move(vals, USE_CUDA)


        res = net((adj_t, vect_feat_t)) # , mat_feat_t))


        if return_pred:
            accum_pred_val = {}
            if isinstance(res, dict):
                for k, v in res.items():
                    accum_pred_val[k] = res[k].cpu().detach().numpy()
            else:
                accum_pred_val['res'] = res.cpu().detach().numpy()
            accum_pred_val['mask'] = mask.cpu().detach().numpy()
            accum_pred_val['truth'] = vals.cpu().detach().numpy()

            accum_pred.append(accum_pred_val)

        if criterion is None:
            loss = 0.0
        else:
            loss = criterion(res, vals_t, mask_t)

        if not pred_only:
            loss.backward()
            optimizer.step()

        obs_points = mask.sum()
        if criterion is not None:
            running_loss += loss.item() * obs_points
        total_points +=  obs_points


        t2 = time.time()
        total_compute_time += (t2-t1)
        #ksjndask
    t2_total = time.time()
    


    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'total_points' : total_points, 
            'time' : t2_total-t1_total, 
            'compute_time' : total_compute_time, 
            }
    if return_pred:
        keys = accum_pred[0].keys()
        for k in keys:
            accum_pred_v = np.vstack([a[k] for a in accum_pred])
            res[f'pred_{k}'] = accum_pred_v
            
    return res


def generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, 
                   MAX_EPOCHS=1000, 
                   USE_CUDA=True, use_std=False, 
                   writer=None, validate_func = None, 
                   checkpoint_func = None):


    # loss_scale = torch.Tensor(loss_scale)
    # std_scale = torch.Tensor(std_scale)


    for epoch_i in tqdm(range(MAX_EPOCHS)):
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.0
        total_compute_time = 0.0
        t1_total = time.time()

        net.train()
        train_res = run_epoch(net, optimizer, criterion, dl_train, 
                               pred_only = False, USE_CUDA=USE_CUDA, desc='train')

        # print("{} {:3.3f} compute={:3.1f}s total={:3.1f}s".format(epoch_i, 
        #                                                           running_loss/total_points, 
        #                                                          total_compute_time, t2_total-t1_total))
        if writer is not None:
            writer.add_scalar("train_loss", train_res['running_loss']/train_res['total_points'], 
                              epoch_i)

        if epoch_i % 5 == 0:
            net.eval()
            test_res = run_epoch(net, optimizer, criterion, dl_test, 
                                 pred_only = True, USE_CUDA=USE_CUDA, 
                                 return_pred=True, desc='validate')
            for metric_name, metric_val in validate_func(test_res).items():
                writer.add_scalar(metric_name, metric_val, epoch_i)
            
            
        if checkpoint_func is not None:
            checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer)


def move(tensor, cuda=False):
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()


class PredModel(object):
    def __init__(self, meta_filename, checkpoint_filename, USE_CUDA=False):

        meta = pickle.load(open(meta_filename, 'rb'))


        self.meta = meta 


        self.USE_CUDA = USE_CUDA

        if self. USE_CUDA:
            net = torch.load(checkpoint_filename)
        else:
            net = torch.load(checkpoint_filename, 
                             map_location=lambda storage, loc: storage)

        self.net = net
        self.net.eval()


    def pred(self, rdmols, values, BATCH_SIZE = 32, 
             debug=False, prog_bar= False):

        dataset_hparams = self.meta['dataset_hparams']
        MAX_N = self.meta.get('max_n', 32)

        USE_CUDA = self.USE_CUDA

        COMBINE_MAT_VECT='row'


        feat_vect_args = dataset_hparams['feat_vect_args']
        feat_mat_args = dataset_hparams['feat_mat_args']
        adj_args = dataset_hparams['adj_args']

        ds = netdataio.MoleculeDatasetMulti(rdmols, values, 
                                            MAX_N, len(self.meta['tgt_nucs']), feat_vect_args, 
                                            feat_mat_args, adj_args,  
                                            combine_mat_vect=COMBINE_MAT_VECT)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


        allres = []
        alltrue = []
        results_df = []
        m_pos = 0
        
        res = run_epoch(self.net, None, None, dl, 
                        pred_only = True, USE_CUDA=self.USE_CUDA,
                        return_pred = True, print_shapes=debug, desc='predict',
                        progress_bar=prog_bar)
        

        for rd_mol_i, (rdmol, true_val) in enumerate(zip(rdmols, values)):
            for nuc_i, nuc in enumerate(self.meta['tgt_nucs']):
                true_nuc_spect = true_val[nuc_i]
                for atom_idx, true_shift in true_nuc_spect.items():
                    atom_res = {}
                    for pred_key in ['pred_mu', 'pred_std']:
                        atom_res[pred_key] = res[pred_key][rd_mol_i, atom_idx, nuc_i]
                    atom_res['nuc_i'] = nuc_i
                    atom_res['nuc'] = nuc
                    atom_res['atom_idx'] = atom_idx
                    atom_res['m_pos'] = rd_mol_i
                    atom_res['value'] = true_shift
                    
                    results_df.append(atom_res)
                        
        results_df = pd.DataFrame(results_df)
        return results_df
    
    
  

def rand_dict(d):
    p = {}
    for k, v in d.items():
        if isinstance(v, list):
            p[k] = v[np.random.randint(len(v))]
        else:
            p[k] = v
    return p


def create_validate_func(tgt_nucs):
    def val_func(res): # val, mask, truth):
        val = res['pred_res']
        mask = res['pred_mask']
        truth = res['pred_truth']
        res = {}
        for ni, n in enumerate(tgt_nucs):
            delta = (val[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            res[f"{n}/test_std_err"] = np.std(delta)
            res[f"{n}/test_max_error"] = np.max(np.abs(delta))
            res[f"{n}/test_mean_abs_err"] = np.mean(np.abs(delta))
            res[f"{n}/test_abs_err_90"] = np.percentile(np.abs(delta), 90)
        return res
    return val_func

def create_uncertain_validate_func(tgt_nucs):
    def val_func(res): # val, mask, truth):
        mu = res['pred_mu']
        std = res['pred_std'] 
        mask = res['pred_mask']
        truth = res['pred_truth']
        res = {}
        for ni, n in enumerate(tgt_nucs):
            delta = (mu[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            masked_std = (std[:, :, ni])[mask[:, :, ni] > 0].flatten()
            res[f"{n}/test_std_err"] = np.std(delta)
            res[f"{n}/test_max_error"] = np.max(np.abs(delta))
            res[f"{n}/test_mean_abs_err"] = np.mean(np.abs(delta))
            res[f"{n}/test_abs_err_90"] = np.percentile(np.abs(delta), 90)
            res[f"{n}/std/mean"] = np.mean(masked_std)
            res[f"{n}/std/min"] = np.min(masked_std)
            res[f"{n}/std/max"] = np.max(masked_std)

        return res
    return val_func


