import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
import torch.nn as nn
from scipy import signal
from rdkit.Chem import rdMolDescriptors as rdDesc
import numpy as np
from time import time

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

total_time = 0

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
        
    return list(map(lambda s: x == s, allowable_set))

def makeSpectrumFeature(shift, bins = 64, shiftRange = (0,220), sd=2):
    centre = (shift-shiftRange[0])/(shiftRange[1]-shiftRange[0])
    centre *= bins
    centre = int(centre)
    gaussianList = signal.gaussian(bins*3,sd)
    shiftedList = gaussianList[int(bins*1.5-centre):int(bins*2.5-centre)]
    return shiftedList

def makeSpectrumFeature(shift, splitting, bins = 64, shiftRange = (0,220), sd=2):
    centre = (shift-shiftRange[0])/(shiftRange[1]-shiftRange[0])
    centre *= bins
    centre = int(centre)
    gaussianList = signal.gaussian(bins*3,sd)
    shiftedList = gaussianList[int(bins*1.5-centre):int(bins*2.5-centre)]
    splittingFeature = one_of_k_encoding_unk(splitting, ["S", "D", "T", "Q"])
    output = np.append(shiftedList, splittingFeature)
    return output

def shiftValFromFeature(feature :list, bins = 64, shiftRange = (0,220)):
    return (np.argmax(feature)+1)*(shiftRange[1]-shiftRange[0])/bins

def netSpectrumFeature(array):
    net = []
    for i in array:
        net.append(makeSpectrumFeature(i))
    net = np.array(net)
    net = np.sum(net, axis=0)
    return net


def get_atom_features(atom, nmr_feature, explicit_H=False):

    """
    Method that computes atom level features from rdkit atom object
    :param atom: rdkit atom object
    :return: atom features, 1d numpy array
    """
    # todo: take list  of all possible atoms
    possible_atoms = ['C','N','O','F']
    atom_features  = one_of_k_encoding_unk(atom.GetSymbol(),possible_atoms)
    # atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
    #             Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    #             Chem.rdchem.HybridizationType.SP3])
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4])
    atom_features += list(nmr_feature)

    #todo: add aromacity,acceptor,donor and chirality
    # if not explicit_H:
    #     atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])

    return np.array(atom_features)

def get_bond_features(bond):
    
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """
    
    bond_type = bond.GetBondType()
    bond_feats = [
      bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
      bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
    ]
    # bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()),["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)

def countAtoms(mol, atomicNum):
    pat = Chem.MolFromSmarts("[#"+ str(atomicNum) + "]")
    return len(mol.GetSubstructMatches(pat))

def get_graph_from_molstate(state):
    
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param env: Env() object
    :return: DGL graph object, Node features and Edge features
    """
    G = DGLGraph()
    molecule = state.rdmol
    try:
        Chem.SanitizeMol(molecule)
    except:
        pass
    numInMol = molecule.GetNumAtoms()

    G.add_nodes(state.totalNumOfAtoms)
    node_features = []
    edge_features = []
    for i in range(state.totalNumOfAtoms):

        if state.presentInRdmol[i] == 1:
            atom_i = molecule.GetAtomWithIdx(int(state.rdmolIdx[i]))
            atom_i_features =  get_atom_features(atom_i,state.molGraphSpectra[i])
            node_features.append(atom_i_features)

            for j in range(state.totalNumOfAtoms):
                if state.presentInRdmol[j] != 1:
                    continue

                bond_ij = molecule.GetBondBetweenAtoms(int(state.rdmolIdx[i]), int(state.rdmolIdx[j]))
                if bond_ij is not None:
                    G.add_edge(i,j)
                    bond_features_ij = get_bond_features(bond_ij)
                    edge_features.append(bond_features_ij)
        else :
            atom_feature = one_of_k_encoding(state.AtomicNum[i],[6,7,8,9])
            if state.AtomicNum[i] == 6:
            	atom_feature += [0,0,0,0,1] 
            if state.AtomicNum[i] == 7:
            	atom_feature += [0,0,0,1,0] 
            if state.AtomicNum[i] == 8:
            	atom_feature += [0,0,1,0,0] 
            if state.AtomicNum[i] == 9:
            	atom_feature += [0,1,0,0,0] 
            # atom_feature += [0] * 3
            atom_feature += list(state.molGraphSpectra[i])
            node_features.append(atom_feature)

    G.ndata['x'] = torch.FloatTensor(node_features)
    G.edata['e'] = torch.FloatTensor(edge_features)
    return G


def get_graph_from_env(env):
    
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param env: Env() object
    :return: DGL graph object, Node features and Edge features
    """

    G = DGLGraph()
    molecule = env.state
    numInMol = molecule.GetNumAtoms()
    features = rdDesc.GetFeatureInvariants(molecule)
    
    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0]* molecule.GetNumAtoms()

    for i in stereo:
        # nodeNumber = list(env.indices.keys)[list(env.indices.values).index(i[0])]
        chiral_centers[i[0]] = i[1]
        
    G.add_nodes(env.numberOfAtoms)
    node_features = []
    edge_features = []

    for i in range(numInMol):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features =  get_atom_features(atom_i,chiral_centers[i],features[i])
        node_features.append(atom_i_features)
        
        for j in range(numInMol):

            bond_ij = molecule.GetBondBetweenAtoms(i, j)

            if bond_ij is not None:
                G.add_edge(i,j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)
    
    for i in range(4):
        for j in range( env.numAtoms[i] - countAtoms(env.state,i+6) ):
            atom_feature = one_of_k_encoding(int(i+6),[6,7,8,9])
            atom_feature += [0] * 32
            node_features.append(atom_feature)

    G.ndata['x'] = torch.FloatTensor(node_features)
    G.edata['e'] = torch.FloatTensor(edge_features)
    return G

