"""
Per-atom features : Featurizations that return one per atom
[in contrast to whole-molecule featurizations]
"""

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
#from numba import jit
import scipy.spatial
from rdkit import Chem
from .util import get_nos_coords



def atom_adj_mat(mol, conformer_i, **kwargs):
    """
    OUTPUT IS ATOM_N x (adj_mat, tgt_atom, atomic_nos, dists )
    
    This is really inefficient given that we explicitly return the same adj
    matrix for each atom, and index into it
    
    Adj mat is valence number * 2
    
    
    """
    
    MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 64)
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    ATOM_N = len(atomic_nos)

    atomic_nos_pad, adj = mol_to_nums_adj(mol, MAX_ATOM_N)
    

    features = np.zeros((ATOM_N,), 
                    dtype=[('adj', np.uint8, (MAX_ATOM_N, MAX_ATOM_N)), 
                           ('my_idx', np.int), 
                           ('atomicno', np.uint8, MAX_ATOM_N), 
                           ('pos', np.float32, (MAX_ATOM_N, 3,))])

    
    
    for atom_i in range(ATOM_N):
        vects = coords - coords[atom_i]
        features[atom_i]['adj'] = adj*2
        features[atom_i]['my_idx'] =  atom_i
        features[atom_i]['atomicno'] = atomic_nos_pad
        features[atom_i]['pos'][:ATOM_N] = vects
    return features

def advanced_atom_props(mol, conformer_i, **kwargs):
    import rdkit.Chem.rdPartialCharges
    pt = Chem.GetPeriodicTable()
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                     catchErrors=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    ATOM_N = len(atomic_nos)
    out = np.zeros(ATOM_N, 
                   dtype=[('total_valence', np.int),  
                          ('aromatic', np.bool), 
                          ('hybridization', np.int), 
                          ('partial_charge', np.float32),
                          ('formal_charge', np.float32), 
                          ('atomicno', np.int), 
                          ('r_covalent', np.float32),
                          ('r_vanderwals', np.float32),
                          ('default_valence', np.int),
                          ('rings', np.bool, 5), 
                          ('pos', np.float32, 3)])
    
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        out[i]['total_valence'] = a.GetTotalValence()
        out[i]['aromatic'] = a.GetIsAromatic()
        out[i]['hybridization'] = a.GetHybridization()
        out[i]['partial_charge'] = a.GetProp('_GasteigerCharge')
        out[i]['formal_charge'] = a.GetFormalCharge()
        out[i]['atomicno'] = atomic_nos[i]
        out[i]['r_covalent'] =pt.GetRcovalent(atomic_num)
        out[i]['r_vanderwals'] =  pt.GetRvdw(atomic_num)
        out[i]['default_valence'] = pt.GetDefaultValence(atomic_num)
        out[i]['rings'] = [a.IsInRingSize(r) for r in range(3, 8)]
        out[i]['pos'] = coords[i]
                          
    return out

HYBRIDIZATIONS = [Chem.HybridizationType.S, 
                  Chem.HybridizationType.SP, 
                  Chem.HybridizationType.SP2, 
                  Chem.HybridizationType.SP3, 
                  Chem.HybridizationType.SP3D, 
                  Chem.HybridizationType.SP3D2]

def to_onehot(x, vals):
    return [x == v for v in vals]
        
                  
def feat_tensor_atom(mol, 
                     feat_atomicno = True, feat_pos=True, 
                     feat_atomicno_onehot=[1, 6, 7, 8, 9], 
                     feat_valence=True, aromatic=True, hybridization=True, 
                     
                     partial_charge=True, formal_charge=True, r_covalent=True,
                     r_vanderwals=True, default_valence=True, rings=False, 
                     total_valence_onehot=False, 
                     conf_idx = 0):

    """
    Featurize a molecule on a per-atom basis
    feat_atomicno_onehot : list of atomic numbers

    Always assume using conf_idx unless otherwise passed

    Returns an (ATOM_N x feature) float32 tensor

    NOTE: Performs NO santization or cleanup of molecule, 
    assumes all molecules have sanitization calculated ahead
    of time. 

    """

    pt = Chem.GetPeriodicTable()
    mol = Chem.Mol(mol) # copy molecule

    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)    

    #Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
    #                 catchErrors=True)


    if partial_charge:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

    atom_features = []
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        atom_feature = []

        if feat_atomicno:
            atom_feature += [atomic_num]

        if feat_pos:
            atom_feature += coords[i].tolist()

        if feat_atomicno_onehot is not None : 
            atom_feature += to_onehot(atomic_num, feat_atomicno_onehot)
            
        if feat_valence:
            atom_feature += [a.GetTotalValence()]
        if total_valence_onehot:
            atom_feature +=  to_onehot(a.GetTotalValence(), range(1, 7))

        if aromatic:
            atom_feature += [a.GetIsAromatic()]

        if hybridization:
            atom_feature += to_onehot(a.GetHybridization(), HYBRIDIZATIONS)

        if partial_charge:
            gc = float(a.GetProp('_GasteigerCharge'))
            #assert np.isfinite(gc)
            if not np.isfinite(gc):
                gc = 0.0
            atom_feature += [gc]

        if formal_charge:
            atom_feature += to_onehot(a.GetFormalCharge(), [-1, 0, 1])

        if r_covalent:
            atom_feature += [pt.GetRcovalent(atomic_num)]
        if r_vanderwals:
            atom_feature += [pt.GetRvdw(atomic_num)]
            

        if default_valence:
            atom_feature += to_onehot(pt.GetDefaultValence(atomic_num), range(1, 7))

        if rings:
           atom_feature +=  [a.IsInRingSize(r) for r in range(3, 8)]

        atom_features.append(atom_feature)

    return torch.Tensor(atom_features)

