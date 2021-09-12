from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import numpy as np
import copy
import random
from collections import OrderedDict
from rdkit.Chem import rdMolDescriptors as rdDesc
from .molecular_graph import *
from .molecule_state import MolState
import torch
import warnings
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit import DataStructs
from .RPNMR import predictor_new as P
from scipy.stats import wasserstein_distance
import pickle
from copy import deepcopy
import ray
with open("../../data/10kMolSplit.pkl", "rb") as inFile:
    dat = pickle.load(inFile)
    dat = dat["test_df"]

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

class Env:
    def __init__(self, molForm, targetSpectra,episode_actor):
        self.molForm = molForm
        self.state = MolState(molForm,targetSpectra)
        self.targetSpectra = targetSpectra
        self.episode_actor = episode_actor

    def __del__(self) :
        pass

    def __str__(self):
        return "Current Env State : " + str(self.state) + " MolForm : " + str(self.molForm)

    def convertStrToList(self, string):
        return [int(i) for i in string.strip('][').split(', ')]

    def reset(self,index):
        while True:
            row = dat.iloc[index]
            molForm = row.MolForm
            self.molForm = self.convertStrToList(molForm)
            self.targetSpectra = row.Spectrum
            if self.molForm[0] == len(self.targetSpectra):
                break
            index = index + 1

        self.targetmol = row.Smiles
        self.targetrdmol = row.Rdmol
        self.state = MolState(self.molForm, self.targetSpectra)

    def reward(self, state = None, target = None):
        if state is None:
            state = self.state
        if target is None:
            target = self.targetSpectra
        mol = deepcopy(self.state.rdmol)
        Chem.SanitizeMol(mol)

        target_nmr = self.targetSpectra[:]
        S, D, T, Q = 0, 0, 0, 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                if atom.GetNumImplicitHs() == 0:
                    S += 1
                if atom.GetNumImplicitHs() == 1:    
                    D += 1
                if atom.GetNumImplicitHs() == 2:
                    T += 1
                if atom.GetNumImplicitHs() == 3:
                    Q += 1
        splits = [i[1] for i in target_nmr]
            
        if splits.count('Q') > Q:
            return 0
        elif splits.count('Q') == Q and splits.count('T') > T:
            return 0
        elif splits.count('Q') == Q and splits.count('T') == T and splits.count('D') > D:
            return 0

        predicted_nmr = ray.get(self.episode_actor.predict.remote(Chem.MolToSmiles(mol)))
        if predicted_nmr == -1:
            return 0
        predicted_nmr = np.array(predicted_nmr)

        if(np.any(predicted_nmr<0)):
            reward = 0
            return 0 
            
        predicted_nmr /= 220

        
        target_nmr = [i[0] for i in target_nmr]
        target_nmr = np.array(target_nmr, dtype=np.float64)
        target_nmr /= 220
        reward = 1 - wasserstein_distance(predicted_nmr,target_nmr)
        return 2*(reward-0.5)

    def terminal_reward(self, state = None, target = None):
        if state is None:
            state = self.state
        if target is None:
            target = self.targetSpectra
        mol = deepcopy(state.rdmol)
        Chem.SanitizeMol(mol)


        if state.numInRdmol < state.totalNumOfAtoms:
            return 0

        predicted_nmr = ray.get(self.episode_actor.predict.remote(Chem.MolToSmiles(mol)))
        if predicted_nmr == -1:
            return 0
        predicted_nmr = np.array(predicted_nmr)

        if(np.any(predicted_nmr<0)):
            reward = 0
            return 0 
            
        predicted_nmr /= 220

        target_nmr = self.targetSpectra[:]
        target_nmr = [i[0] for i in target_nmr]
        if len(target_nmr) != self.molForm[0]:
            print(self.targetSpectra)
            print(target_nmr)
            raise "Target NMR Error"
        target_nmr = np.array(target_nmr, dtype=np.float64)
        target_nmr /= 220
        reward = 1 - wasserstein_distance(predicted_nmr,target_nmr)
        return 2*(reward-0.5) 

    def invalidAction(self):
        raise Exception("Invalid Action has been chosen :(")

    def isTerminal(self, state: MolState = None):
        try:
            if state is None:
                state = self.state
            if sum(state.valid_actions()) == 0:
                return True
            if state.numInRdmol < state.totalNumOfAtoms:
                return False
            target_nmr = self.targetSpectra[:]
            mol = deepcopy(state.rdmol)
            Chem.SanitizeMol(mol)
            
            # mol = Chem.AddHs(mol)
            S, D, T, Q = 0, 0, 0, 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    if atom.GetNumImplicitHs() == 0:
                        S += 1
                    if atom.GetNumImplicitHs() == 1:    
                        D += 1
                    if atom.GetNumImplicitHs() == 2:
                        T += 1
                    if atom.GetNumImplicitHs() == 3:
                        Q += 1
            splits = [i[1] for i in target_nmr]
                
            if splits.count('Q') > Q:
                return True
            elif splits.count('Q') == Q and splits.count('T') > T:
                return True
            elif splits.count('Q') == Q and splits.count('T') == T and splits.count('D') > D:
                return True
            elif splits.count('Q') == Q and splits.count('T') == T and splits.count('D') == D and splits.count('S') == S:
               return True

            if S > splits.count('S'):
                return True
            elif S == splits.count('S') and D > splits.count('D'):
                return True
            elif S == splits.count('S') and D == splits.count('D') and T > splits.count('T'):
                return True

            if self.terminal_reward(state) > 0.99 :
                return True
            else:
                return False
        except Exception as e:
            print(e)
            print(Chem.MolToSmiles(self.state.rdmol))

    
    def step(self,  actionInt: int, state: MolState = None):
        if state is None:
            state = self.state
        
        valid_actions = state.action_mask
        if valid_actions[actionInt] == 0:
            action = state._actionIntToList(actionInt)
            print(state._actionIntToList(actionInt))
            print(state)

            return self.invalidAction()

        state.doStep(state._actionIntToList(actionInt))

        terminal = self.isTerminal(state)
        _ = state.valid_actions()
        if terminal:
            reward = self.reward(state)
            return state, reward,terminal
        else:
            return state, 0,terminal
        

        


# %%


