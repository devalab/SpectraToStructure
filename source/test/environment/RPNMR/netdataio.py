import numpy as np
import torch
import pandas as pd
from . import atom_features
from . import molecule_features
import torch.utils.data

class MoleculeDatasetNew(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vals , MAX_N, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {}, combine_mat_vect = None, 
                 mask_zeroout_prob=0.0):
        self.mols = mols
        self.pred_vals = pred_vals
        self.MAX_N = MAX_N
        self.cache = {}
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        self.mask_zeroout_prob = mask_zeroout_prob

    def __len__(self):
        return len(self.mols)
    
    def mask_sel(self, v):
        if self.mask_zeroout_prob == 0.0: # not self.single_value:
            return v
        else:
            mask = v[4].copy()
            a = np.sum(mask)
            mask[np.random.rand(*mask.shape) < self.mask_zeroout_prob] = 0.0
            b = np.sum(mask)
            out = list(v)
            out[4] = mask
            return out

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        pred_val = self.pred_vals[idx]

        conf_idx = np.random.randint(mol.GetNumConformers())

        if self.cache_key(idx, conf_idx) in self.cache:
            return self.mask_sel(self.cache[self.cache_key(idx, conf_idx)])
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect

        f_mat = molecule_features.feat_tensor_mol(mol, conf_idx=conf_idx,
                                                  **self.feat_edge_args) 

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]

        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)
        # do the padding
        mat_feat[:DATA_N, :DATA_N, :f_mat.shape[2]] = f_mat  
        
        if self.combine_mat_vect == 'row':
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2]:] = f_vect
        elif self.combine_mat_vect == 'col':
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2]:] = f_vect

        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad
                        
        # create mask and preds 
        
        mask = np.zeros(self.MAX_N, dtype=np.float32)
        vals = np.zeros(self.MAX_N, dtype=np.float32)
        for k, v in pred_val.items():
            mask[k] = 1.0
            vals[k] = v

        v = (adj, vect_feat, mat_feat, 
            vals, 
            mask,)
        
        
        self.cache[self.cache_key(idx, conf_idx)] = v
        return self.mask_sel(v)


class MoleculeDatasetMulti(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vals, MAX_N, 
                 PRED_N = 1, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {}, combine_mat_vect = None, 
        ):
        self.mols = mols
        self.pred_vals = pred_vals
        self.MAX_N = MAX_N
        self.cache = {}
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        #self.mask_zeroout_prob = mask_zeroout_prob
        self.PRED_N = PRED_N

    def __len__(self):
        return len(self.mols)
    
    def mask_sel(self, v):
        return v

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        pred_val = self.pred_vals[idx]

        conf_idx = np.random.randint(mol.GetNumConformers())

        if self.cache_key(idx, conf_idx) in self.cache:
            return self.mask_sel(self.cache[self.cache_key(idx, conf_idx)])
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect

        f_mat = molecule_features.feat_tensor_mol(mol, conf_idx=conf_idx,
                                                  **self.feat_edge_args) 

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]
        if MAT_CHAN == 0: # Dataloader can't handle tensors with empty dimensions
            MAT_CHAN = 1
        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)
        # do the padding
        mat_feat[:DATA_N, :DATA_N, :f_mat.shape[2]] = f_mat  
        
        if self.combine_mat_vect == 'row':
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2]:] = f_vect
        elif self.combine_mat_vect == 'col':
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2]:] = f_vect

        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad
                        
        # create mask and preds 
        
        mask = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        vals = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        #print(self.PRED_N, pred_val)
        for pn in range(self.PRED_N):
            for k, v in pred_val[pn].items():
                mask[int(k), pn] = 1.0
                vals[int(k), pn] = v

        v = (adj, vect_feat, mat_feat, 
            vals, 
            mask,)
        
        
        self.cache[self.cache_key(idx, conf_idx)] = v

        return self.mask_sel(v)
