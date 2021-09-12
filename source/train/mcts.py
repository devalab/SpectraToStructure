import numpy as np
import argparse
from collections import OrderedDict
import os
import time
import copy
import matplotlib.pyplot as plt
import random
from rdkit import Chem
import torch
from rdkit.Chem.QED import qed
from tqdm import tqdm
plt.style.use('ggplot')

from utils.helpers import (argmax,stable_normalizer)
from environment.environment import  Env
from environment.molecule_state import MolState
from time import time
BOND_TYPES = 3
MAX_NODES = 9
called = 0
total_time = []
cache = {}
##### MCTS functions #####

def get_select_indexes(state):
    arr = []
    nodes = len(state.mol_graph.ndata['x'])
    if nodes not in cache:
        for i in range(nodes):
            arr.append(np.arange(i*nodes*BOND_TYPES,(i+1)*nodes*BOND_TYPES))
            arr.append(np.zeros((MAX_NODES-nodes)*BOND_TYPES))
        arr = np.hstack(arr)
        arr = np.pad(arr,(0,state.na-len(arr)),constant_values=0)

        cache[nodes] = arr.copy()

    return cache[nodes]
      
class Action():
    ''' Action object '''
    def __init__(self,index,parent_state,Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
                
    def add_child_state(self,s1,r,terminal,model,valuemodel):
        self.child_state = State(s1,r,terminal,self,self.parent_state.na,model,valuemodel)
        return self.child_state
        
    def update(self,R):
        self.n += 1
        self.W += R
        self.Q = self.W/self.n

class State():
    ''' State object '''

    def __init__(self,mol_state,r,terminal,parent_action,na,model,valuemodel):
        ''' Initialize a new state '''
        
        self.mol_state = copy.deepcopy(mol_state)
        self.mol_graph = self.mol_state.molGraph # state (constructed molecule) + individual atoms
            
        self.r = r # reward upon arriving in this state
        self.terminal = terminal # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model

        self.valuemodel = valuemodel
        
        if not self.terminal:
            self.calc_rollout_reward()
        else:
            self.rollout_reward = 0.0
        # Child actions
        self.na = na 
        self.action_mask = self.mol_state.action_mask.copy() #better to have a standalone function also.
        self.index_mask = get_select_indexes(self)*self.action_mask
        self.action_1_mask = self.action_mask.copy()

        self.N_value = torch.zeros(na)
        self.W_value = torch.zeros(na)
        self.Q_value = torch.zeros(na)
        self.Q_value[self.action_mask==0] = -100

        self.child_actions = list(np.zeros(na))
        
        for i in np.where(self.action_mask)[0]:
            self.child_actions[i] = (Action(i,parent_state=self,Q_init=self.Q_value[i]))
        

        self.model = model
        self.priors = model.predict_pi(self.mol_graph,self.action_mask,self.action_1_mask,self.index_mask)
        self.priors[(self.action_mask==0).nonzero()] = -1e8
        self.priors = (torch.softmax(self.priors,dim=0)*torch.FloatTensor(self.action_mask)).flatten()
        self.addPriorNoise()


    def update_Q(self,R,index):
        self.N_value[index] += 1
        self.W_value[index] += R
        self.Q_value[index] = self.W_value[index]/self.N_value[index]

    def select(self,c=0.01):
        ''' Select one of the child actions based on UCT rule '''
        random.seed()
        UCT = self.Q_value + (self.priors*c*np.sqrt(self.n+1)/(self.N_value+1))
        # winner = argmax(UCT)
        winner = random.choice(torch.where(UCT==torch.max(UCT))[0])
        return self.child_actions[winner]

    def addPriorNoise(self):
        """
        Adds dirichlet noise to priors.
        Called when the state is the root node
        """
        np.random.seed()
        alpha = 2/10
        e = 0.25
        noise = np.random.dirichlet([alpha] * int(sum(self.action_mask)))
        noiseReshape = np.zeros(self.action_mask.shape)
        noiseIdx = 0
        for i in range(len(self.action_mask)):
            if self.action_mask[i]:
                noiseReshape[i] = noise[noiseIdx]
                noiseIdx += 1

        self.priors = (1-e) * self.priors + e * noiseReshape

    def step(self, action):
        ret = self.env.step(self.env._actionIntToList(int(action)))
        self.env, self.r , self.terminal = ret

    def calc_rollout_reward(self):
        """
        performs R random rollout, the total reward in each rollout is computed.
        returns: mean across the R random rollouts.
        """
        self.rollout_reward = self.model.predict_V(self.mol_state.molGraph)[0]
        return self.rollout_reward

    def do_rollout(self,k):
        raise "This Function is deprecated. Use approximator instead."

    def update(self):
        ''' update count on backward pass '''
        self.n += 1
        
class MCTS():
    ''' MCTS object '''

    def __init__(self,root,root_molstate,model,valuemodel,na,gamma):
        self.root = root
        self.root_molstate = root_molstate
        self.model = model
        self.valuemodel = valuemodel
        self.na = na
        self.gamma = gamma
    
    def search(self,n_mcts,c,Env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_molstate,r=0.0,terminal=False,parent_action=None,na=self.na,model=self.model,valuemodel=self.valuemodel) # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))
        # for i in tqdm(range(n_mcts), desc="Search Progress"):    
        for i in range(n_mcts):
            state = self.root # reset to root for new trace
            mcts_env = copy.deepcopy(Env)          

            flag = 0
            while not state.terminal:
                action = state.select(c=c)
                return_step = mcts_env.step(int(action.index))
                s1,r,t = return_step
                if hasattr(action,'child_state'):
                    state = action.child_state # select
                    continue
                else:
                    
                    state = action.add_child_state(s1,r,t,self.model,self.valuemodel) # expand
                    
                    break

            # Back-up 
            R = state.rollout_reward
            while state.parent_action is not None: # loop back-up until root is reached
                R = state.r + self.gamma * R 
                action = state.parent_action
                state = action.parent_state
                state.update_Q(R,action.index)
                state.update() 
            del mcts_env

    def return_results(self,temp):
        ''' Process the output at the root node '''
        counts = self.root.N_value
        Q = self.root.Q_value
        pi_target = stable_normalizer(counts,temp)
        action_1 = np.expand_dims(np.array(pi_target),-1).reshape(MAX_NODES,-1)
        action_1 = np.sum(action_1,1)
        V_target = torch.sum((counts/torch.sum(counts))*Q)
        return self.root, pi_target.numpy(), torch.FloatTensor([V_target]), action_1
    
    def forward(self,a,s1):
        ''' Move the root forward '''
        if not hasattr(self.root.child_actions[a],'child_state'):
            print("ERROR")
            self.root = None
            self.env = s1
        else:
            self.root = self.root.child_actions[a].child_state
