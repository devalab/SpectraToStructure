import numpy as np
import pickle
import pandas as pd
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob

import time
from . import util
from .util import move
from . import nets

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
from . import netdataio
import itertools

class Model(object):
    def __init__(self, meta_filename, checkpoint_filename, USE_CUDA=False):

        meta = pickle.load(open(meta_filename, 'rb'))


        self.meta = meta 


        self.USE_CUDA = USE_CUDA

#        if self. USE_CUDA:
#            net = torch.load(checkpoint_filename)
#        else:
#            net = torch.load(checkpoint_filename, 
#                             map_location=lambda storage, loc: storage)
        self.net_params =  {
                "init_noise": 0.01,
                "resnet": True,
                "int_d": 2048,
                "layer_n": 10,
                "GS": 4,
                "agg_func": "goodmax",
                "force_lin_init": True,
                "g_feature_n": 37,
                "resnet_out": True,
                "out_std": True,
                "graph_dropout": 0.0,
                "resnet_d": 128,
                "OUT_DIM": 1
            }
        modelState = checkpoint_filename 
        self.model = nets.GraphVertModel(**self.net_params)
        #self.model.load_state_dict(torch.load(modelState))
#        self.model.eval()
        if self.USE_CUDA:
            self.model.load_state_dict(torch.load(modelState), strict=False)
        else:
            self.model.load_state_dict(torch.load(modelState, map_location=torch.device("cpu")), strict=False)
        self.net = self.model
        self.net.eval()


    def pred(self, rdmols, values, BATCH_SIZE = 32, debug=False):

        data_feat = self.meta.get('data_feat', ["vect"])
        MAX_N = self.meta.get('max_n', 32)

        USE_CUDA = self.USE_CUDA

        COMBINE_MAT_VECT='row'

        self.dat_meta = self.meta["dataset_hparams"]
        feat_vect_args = self.dat_meta['feat_vect_args']
        feat_mat_args = self.dat_meta['feat_mat_args']
        adj_args = self.dat_meta['adj_args']

        ds = netdataio.MoleculeDatasetMulti(rdmols, values, 
                                            MAX_N, len(self.meta['tgt_nucs']), feat_vect_args, 
                                            feat_mat_args, adj_args,  
                                            combine_mat_vect=COMBINE_MAT_VECT)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


        allres = []
        alltrue = []
        results_df = []
        m_pos = 0
        #removing tqdm :
        for i_batch, (adj, vect_feat, mat_feat, vals, mask) in enumerate(dl): #enumerate(tqdm(dl))

            if debug:
                print(adj.shape, vect_feat.shape, mat_feat.shape, vals.shape, mask.shape)
            adj_t = move(adj, USE_CUDA)

            args = [adj_t]

            if 'vect' in data_feat:
                vect_feat_t = move(vect_feat, USE_CUDA)
                args.append(vect_feat_t)

            if 'mat' in data_feat:
                mat_feat_t = move(mat_feat, USE_CUDA)
                args.append(mat_feat_t)

            mask_t = move(mask, USE_CUDA)
            vals_t = move(vals, USE_CUDA)

            res = self.net(args)

            del adj_t
            del vect_feat_t
            del mask_t
            del vals_t

            res = (res["mu"], res["std"])
            std_out = False
            if isinstance(res, tuple):
                std_out = True
                
            if std_out:
                y_est = res[0].detach().cpu().numpy()
                std_est = res[1].detach().cpu().numpy()
            else:
                y_est = res.detach().cpu().numpy()
                std_est = np.zeros_like(y_est)
            for m, v, v_est, std_est in zip(mask.numpy(), vals.numpy(), y_est, std_est):
                for nuc_i, nuc in enumerate(self.meta['tgt_nucs']):
                    m_n = m[:, nuc_i]
                    v_n = v[:, nuc_i]
                    v_est_n = v_est[:, nuc_i]
                    std_est_n = std_est[:, nuc_i]
                    for i in np.argwhere(m_n > 0).flatten():
                        results_df.append({
                            'i_batch' : i_batch, 'atom_idx' : i, 
                            'm_pos' : m_pos,
                            'nuc_i' : nuc_i, 
                            'nuc' : nuc, 
                            'std_out' : std_out, 
                            'value' : v_n[i], 
                            'est' : float(v_est_n[i]), 
                            'std' : float(std_est_n[i]), 
                            
                        })
                m_pos += 1

        results_df = pd.DataFrame(results_df)
        return results_df
