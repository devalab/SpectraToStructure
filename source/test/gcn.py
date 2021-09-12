import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv
from rdkit import rdBase

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors as rdDesc


import pickle
import numpy as np
import pandas as pd
import random
import warnings


import sys
import argparse


class GCN(nn.Module):
    """
    """
    def __init__(self,
                 node_input_dim=36,
                 edge_input_dim=6,
                 node_hidden_dim=36,
                 edge_hidden_dim=36,
                 num_step_message_passing=3,
                 n_heads=3):
        super(GCN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(nn.Linear(edge_input_dim, edge_hidden_dim),
                                    nn.ReLU(),
	                                nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(
                           in_feats=node_hidden_dim,
	                       out_feats=node_hidden_dim,
	                       edge_func=edge_network,
	                       aggregator_type='sum',
                           residual=True
                          )
    
    def forward(self, g):
        """
        g: dgl graph
        """
        n_feat = g.ndata['x']

        

        init = n_feat.clone()
        out = torch.relu(self.lin0(n_feat))
        

        if 'e' not in g.edata or g.edata['e'].shape[0] == 0:
            return out + init
        
        e_feat = g.edata['e']

        for i in range(self.num_step_message_passing):
            m = torch.relu(self.conv(g, out, e_feat))
            out = self.message_layer(torch.cat([m, out],dim=1))
        
        return out + init


 
