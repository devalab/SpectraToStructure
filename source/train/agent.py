#The code for agent would go here
# python improts

# torch imports
import torch

# local imports
from utils.helpers import  Database, store_safely
from model import ActionPredictionModel
from valuemodel import RollOutPredictionModel
from mcts import  MCTS
import copy
import numpy as np
import os
import time
import sys
from environment.environment import  Env
from environment.molecule_state import MolState
from train import train
from train import train_value
from rdkit import Chem
from collections import OrderedDict
# import wandb
# wandb.init(project="nmr")   

class Agent:

    def __init__(self, num_episodes = 100000, n_mcts=100 ,max_ep_len=20 ,lr= 1e-4, c=4 ,gamma=1.0 ,data_size=2000,
                batch_size=32,temp=1,):
        #initialise hyper params.
        self.num_episodes = num_episodes
        self.n_mcts = n_mcts
        self.max_ep_len = max_ep_len
        self.lr = lr
        self.c = c
        self.gamma =  gamma
        self.data_size = data_size
        self.batch_size = batch_size
        self.temp = temp
        # Random Params to the APM here 
        self.model =  ActionPredictionModel(73, 6, 73,64)
        # wandb.watch(self.model)
        self.valuemodel =  RollOutPredictionModel(node_hidden_dim=73)
        #self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.valueoptimizer = torch.optim.Adam(self.valuemodel.parameters(), lr=lr)
        self.env = Env([6,0,0,0], [128.8188,128.8188,128.8188,128.8188,128.8188,128.8188])
        self.database = Database(max_size=data_size, batch_size=batch_size)
        self.t_total = 0
        self.episode_returns = []
        self.R_best = -np.Inf

        
        #initialize model here instead of in mcts.py
    
    def load_model(self):
        self.valuemodel.load_state_dict(state_dict=torch.load("./trained_models/best_model.tar"))

    def optimize_models(model,batch):
        # train models here
        pass

    def execute_episode(self,ep):
        # episode execution goes here
        # initialize mcts
        episode_rewards = []
        timepoints = []
        a_store = []
        self.env.reset()
        start = time.time()
        s = MolState(self.env.molForm, self.env.targetSpectra)
        mcts = MCTS(root_molstate=s, root=None, model=self.model, valuemodel=self.valuemodel,na = 3*20*20,gamma=self.gamma)  # the object responsible for MCTS searches
        R = 0 
        for t in range(self.max_ep_len):
            # MCTS step
            # mcts_env = copy.deepcopy(self.env)
            self.env.targetmol = "C1=CC=CC=C1"
            mcts.search(n_mcts=self.n_mcts, c=self.c, Env=self.env)  # perform a forward search
            state, pi, V,f = mcts.return_results(self.temp)  # extract the root output
            self.database.store(({'mol_graph':state.mol_graph,'action_mask':state.action_mask,'action_1_mask':state.action_1_mask,'index_mask':state.index_mask}, V, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)
            s1, r, terminal = self.env.step(int(a))
            print(self.env.state,"    Reward:",r,"   Episode: ",ep)
            R += r
            self.t_total += self.n_mcts  # total number of environment steps (counts the mcts steps)

            if terminal:
                break
            else:
                mcts.forward(a, s1)
        del mcts

        # Finished episode
        episode_rewards.append(R)  # store the total episode return
        # wandb.log({'Episode Reward':R})
        timepoints.append(self.t_total)  # store the timestep count of the episode return
        store_safely(os.getcwd(), 'result', {'R': episode_rewards, 't': timepoints})
        sys.stdout.flush()


        if R > self.R_best:
            a_best = a_store
#             seed_best = seed
            self.R_best = R
        print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
                                                                                 np.round((time.time() - start), 1)))
        # Train
        print(self.database.size,"DB Size")
        sys.stdout.flush()
        if self.database.size > 256:
            self.database.reshuffle()
            train_loss = train(1,self.model, self.optimizer, self.database)
            # wandb.log({'Train Prior Loss':train_loss})
            print(train_loss,"train prior loss")
            value_train_loss = train_value(1,self.valuemodel, self.valueoptimizer, self.database)
            # wandb.log({"Train Value Loss":value_train_loss})
            print(value_train_loss,"train value loss")

def run():
    agent_obj = Agent()
    for i in range(agent_obj.num_episodes):
        agent_obj.execute_episode(i)
        quit()

if __name__ == '__main__':
    run()
