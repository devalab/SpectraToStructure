import pandas as pd
import numpy as np
import sklearn.metrics
import torch
#from numba import jit
import scipy.spatial
from rdkit import Chem
from .util import get_nos_coords


def feat_tensor_mol(mol, feat_distances=True, feat_r_pow = None, 
                    MAX_POW_M = 2.0, conf_idx = 0):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    if feat_distances:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)
        c = np.swapaxes(b, 2, 1)
        res_mats.append(c)
    if feat_r_pow is not None:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)**2
        c = np.swapaxes(b, 2, 1)
        d = np.sqrt(np.sum(c, axis=2))
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis]

                       
        for p in feat_r_pow:
            e_pow = e**p
            if (e_pow > MAX_POW_M).any():
                print("WARNING: max(M) = {:3.1f}".format(np.max(e_pow)))
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)
    else: # Empty matrix
        M = np.zeros((ATOM_N, ATOM_N, 0), dtype=np.float32)

    return M 

def mol_to_nums_adj(m, MAX_ATOM_N=None):# , kekulize=False):
    """
    molecule to symmetric adjacency matrix
    """

    m = Chem.Mol(m)

    # m.UpdatePropertyCache()
    # Chem.SetAromaticity(m)
    # if kekulize:
    #     Chem.rdmolops.Kekulize(m)

    ATOM_N = m.GetNumAtoms()
    if MAX_ATOM_N is None:
        MAX_ATOM_N = ATOM_N

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N))
    atomic_nums = np.zeros(MAX_ATOM_N)

    assert ATOM_N <= MAX_ATOM_N

    for i in range(ATOM_N):
        a = m.GetAtomWithIdx(i)
        atomic_nums[i] = a.GetAtomicNum()

    for b in m.GetBonds():
        head = b.GetBeginAtomIdx()
        tail = b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        adj[head, tail] = order
        adj[tail, head] = order
    return atomic_nums, adj



def feat_mol_adj(mol, edge_weighted=True, add_identity=False, 

                 norm_adj=False, split_weights = None):
    """
    Compute the adjacency matrix for this molecule

    If split-weights == [1, 2, 3] then we create separate adj matrices for those
    edge weights

    NOTE: We do not kekulize the molecule, we assume that has already been done

    """
    
    atomic_nos, adj = mol_to_nums_adj(mol)
    ADJ_N = adj.shape[0]
    adj = torch.Tensor(adj)
    
    if edge_weighted and split_weights is not None:
        raise ValueError("can' have both weighted edies and split the weights")
    
    if split_weights is None:
        adj = adj.unsqueeze(0)
        if edge_weighted:
            pass # already weighted
        else:
            adj[adj > 0] = 1.0
    else:
        split_adj = torch.zeros((len(split_weights), ADJ_N, ADJ_N ))
        for i in range(len(split_weights)):
            split_adj[i] = (adj == split_weights[i])
        adj = split_adj
        
    if norm_adj and not add_identity:
        raise ValueError()
        
    if add_identity:
        adj = adj + torch.eye(ADJ_N)

    if norm_adj:
        res = []
        for i in range(adj.shape[0]):
            a = adj[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))

            s1 = D_12.reshape(ADJ_N, 1)
            s2 = D_12.reshape(1, ADJ_N)
            adj_i = s1 * a * s2 
            res.append(adj_i)
        adj = torch.stack(res)
    return adj

