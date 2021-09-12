
MAX_NODES = 9
MAX_ACTIONS = MAX_NODES * MAX_NODES * 3

NUM_GPU = 4

from datetime import datetime

LOG_FILE = "./log_file_{}.txt".format(datetime.now().strftime("%d_%m"))
import torch

# local imports
from utils.helpers import  Database, store_safely
from model import ActionPredictionModel
from mcts import  MCTS
import numpy as np
import os
import time
import sys
from environment.environment import  Env
from environment.molecule_state import MolState
from train import train
# from train import train_value
from rdkit import Chem
from collections import OrderedDict
# import wandb
import ipdb
import pickle
from copy import deepcopy
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from queue import Queue
# from multiprocessing import Manager, Queue
# from multiprocessing.managers import BaseManager, BaseProxy
import random
import argparse
from environment.RPNMR import predictor_new as P

parser = argparse.ArgumentParser()
parser.add_argument('--smiles_counter', required=False)
args = parser.parse_args()
if args.smiles_counter:
    smiles_counter = int(args.smiles_counter)

PRETRAINING_SIZE = 64
BATCH_SIZE = 32
import ray
ray.init(address='auto', _redis_password='5241590000000000')


nmr_list = None

@ray.remote
class Agent(object):

    def __init__(self, num_episodes = 100000, n_mcts=1 ,max_ep_len=20 ,lr= 1e-4, c=1 ,gamma=1.0 ,data_size=10000,
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
        self.t_total = 0
        self.episode_returns = []
        self.R_best = -np.Inf

@ray.remote
def execute_episode(idx,ep,model, valuemodel,nmr_list,episode_actor):
    np.random.seed()
    env = Env([7,0,1,0],[(25.2, 'T', 1), (25.2, 'T', 3), (26.1, 'T', 0), (26.1, 'T', 2), (26.1, 'T', 4), (50.1, 'D', 5), (204.7, 'D', 6)],episode_actor)
    episode_rewards = []

    timepoints = []
    a_store = []
    exps = []
    env.reset(idx)
    # print(env.molForm)
    start = time.time()
    s = MolState(env.molForm, env.targetSpectra)
    mcts = MCTS(root_molstate=s, root=None, model=model, valuemodel=valuemodel,na = 3*MAX_NODES*MAX_NODES,gamma=1.0)  # the object responsible for MCTS searches
    R = 0 
    target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(env.targetrdmol,isomericSmiles=False)))
    print("Targetmol:",target_mol)
    smiles_dict = ray.get(episode_actor.print_dict.remote())
    if(target_mol not in smiles_dict):
        ray.wait([episode_actor.new_set_dict.remote(target_mol)],num_returns=1)
    n_mcts = 1000
    # n_mcts = 50 # Fail test to see if the framework is configured properly
    t = 1
    while True:
        # MCTS step
        if(1 in ray.get(episode_actor.get_dict.remote(target_mol))):
            break
        mcts.search(n_mcts=n_mcts, c=1, Env=env)  # perform a forward search

        state, pi, V,fpib = mcts.return_results(t)  # extract the root output
        if(state.mol_state.rdmol.GetNumAtoms() > (sum(env.molForm)//2)):
            n_mcts = 1000
            # n_mcts = 50 # Fail test to see if the framework is configured properly
        exps.append(({'mol_graph':state.mol_graph,'action_mask':state.action_mask,'action_1_mask':state.action_1_mask,'index_mask':state.index_mask}, V, pi,fpib))
        # Make the true step
        a = np.random.choice(len(pi), p=pi)
        a_store.append(a)
        s1, r, terminal = env.step(int(a))
        print("{:30}".format(str(env.state)),"  Reward:",r,"model prior argmax:  ",np.argmax(state.priors),"  action taken:  ",a," prior argmax:  ",np.argmax(pi),"   Episode: ",ep)
        predicted_mol = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(deepcopy(s1.rdmol))),isomericSmiles=False)
        R = r
        if terminal:
            with open(LOG_FILE, "a") as outFile:
                outFile.write("T: {:30}, P: {:30}, R:{:.3f}, EpIdx: {}, SmilesIdx: {}\n".format(target_mol, predicted_mol,R, ep, idx))
            if target_mol == predicted_mol:
                ray.get(episode_actor.update_dict.remote(target_mol,1))
                ray.get(episode_actor.update_dict.remote(target_mol,R))
            ray.get(episode_actor.update_dict.remote(target_mol,predicted_mol))
            sys.stdout.flush()
            break
        else:
            mcts.forward(a, s1)
    del mcts


    for i in range(1):    
        for exp in exps:
            episode_actor.add_exprience.remote(tuple(exp))


    # Finished episode
    episode_rewards.append(R)  # store the total episode return
    # store_safely(os.getcwd(), 'result', {'R': episode_rewards, 't': timepoints})
    sys.stdout.flush()


    print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),np.round((time.time() - start), 1)))
    if ray.get(episode_actor.get_size.remote()) > PRETRAINING_SIZE:
        ray.get(episode_actor.store_trainreward.remote(R))
    else:
        ray.get(episode_actor.store_reward.remote(R))
    return 1




@ray.remote(num_gpus=1)
class EpisodeActor(object):
    def __init__(self):
        self.experiences = []
        self.clearFlag = False
        self.max_size = 1000
        self.size = 0
        self.insert_index = 0
        self.reward = []
        self.trainreward = []
        self.in_queue = False
        self.model_dict = []
        self.lock = False
        self.smiles_dict = {}
        self.NMR_Pred = P.NMRPredictor("../../trainedModels/RPNMRModels/best_model.meta","../../trainedModels/RPNMRModels/best_model.00000000.state",True)

    def clear_full(self):
        self.__init__()
        return 1

    def predict(self,smile):
    	return self.NMR_Pred.predict(smile)
    
    def add_exprience(self, experience):
        self.store(experience)

    def store(self,experience):
        if self.size < self.max_size:
            self.experiences.append(experience)
            self.size +=1
        else:
            self.experiences[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0
    
    def clear_exprience(self):
        self.messages = []

    def get_experience(self):
        experiences = self.experiences
        return experiences

    def new_set_dict(self,key):
        self.smiles_dict[key] = set()
        return 1

    def update_dict(self,key,value):
        self.smiles_dict[key].add(value)
        return 1

    def print_dict(self):
        return self.smiles_dict

    def get_dict(self,key):
        return self.smiles_dict[key]

    def get_size(self):
        return self.size

    def increment_size(self):
        self.size += 1

    def set_lock(self,value):
        self.lock = value
        return value

    def get_lock(self):
        return self.lock

    def set_queue(self,value):
        self.in_queue = value
        return value

    def get_queue(self):
        return self.in_queue

    def add_model(self,value):
        self.model_dict.append(value)

    def get_model(self):
        return self.model_dict.pop()

    def get_model_length(self):
        return len(self.model_dict)

    def store_reward(self,value):
        self.reward.append(value)
        return 1

    def empty_reward(self):
        self.reward = []
        return 1

    def get_reward(self):
        return self.reward

    def store_trainreward(self,value):
        self.trainreward.append(value)
        return 1

    def empty_trainreward(self):
        self.trainreward = []
        return 1

    def get_trainreward(self):
        return self.trainreward

def run():
    num_episodes = 1
    agent_obj = Agent.remote()
    num_processes = 20
    manager = mp.Manager()
    database = manager.dict()

    episode_actors = [EpisodeActor.remote() for i in range(NUM_GPU)]



    ### --------------models------------------ ###
    model =  ActionPredictionModel(77, 6, 77,64)
    model.load_state_dict(torch.load("../../trainedModels/default.state",map_location='cpu'))
    model_episode =  ActionPredictionModel(77, 6, 77,64)
    model_episode.load_state_dict(deepcopy(model.state_dict()))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    valuemodel =  "valuemodel"
    valuemodel_episode =  "valuemodel"

    ### --------------models------------------ ###
    episode_counter = 0

    #indexes from the total test dataset that are in test for out usecase i.e <= 9 heavy atoms:
    #These are the indices that are tested. They are loaded in the class environment when env.reset is called. idx is passes as the parameter to the environment to initiate the environment and load the target NMR Spectra.
    smiles_idx = [2, 7, 11, 20, 21, 34, 40, 45, 48, 49, 50, 54, 55, 56, 57, 63, 64, 67, 72, 77, 78, 81, 88, 90, 92, 95, 104, 113, 114, 117, 120, 123, 125, 134, 139, 144, 148, 154, 158, 159, 161, 165, 168, 172, 181, 188, 189, 192, 194, 205, 208, 216, 229, 233, 235, 236, 240, 246, 249, 250, 255, 256, 260, 263, 271, 281, 282, 286, 287, 288, 289, 293, 298, 303, 304, 305, 315, 316, 320, 323, 336, 337, 343, 348, 351, 353, 356, 357, 359, 360, 362, 367, 371, 374, 379, 382, 383, 393, 407, 415, 419, 422, 423, 433, 434, 441, 442, 446, 449, 455, 462, 466, 468, 470, 475, 476, 481, 482, 483, 487, 489, 493, 502, 506, 509, 523, 526, 527, 528, 537, 539, 540, 544, 550, 556, 557, 580, 584, 609, 617, 622, 628, 629, 631, 647, 652, 658, 661, 663, 666, 690, 692, 693, 698, 711, 713, 722, 732, 734, 746, 753, 764, 769, 780, 783, 784, 787, 789, 790, 796, 797, 799, 801, 811, 813, 814, 815, 822, 824, 825, 828, 833, 834, 837, 839, 842, 843, 852, 856, 858, 862, 868, 869, 872, 875, 878, 881, 883, 892, 896, 899, 903, 904, 905, 907, 917, 920, 930, 935, 947, 961, 963, 966, 969, 970, 984, 991, 998, 1000, 1004, 1005, 1006, 1007, 1009]
    for smiles_counter in smiles_idx:
        for i in range(num_episodes):
            idxs = np.arange(smiles_counter,smiles_counter+num_processes)
            for temp in range(NUM_GPU):
                res_dict = ray.get(episode_actors[temp].set_lock.remote(True))
            results = [execute_episode.remote(smiles_counter,j*NUM_GPU + episode_actor_idx,model_episode, valuemodel_episode,nmr_list,episode_actors[episode_actor_idx]) for idx,j in enumerate(range(num_processes*i,num_processes*(i+1))) for episode_actor_idx in range(NUM_GPU)]
            print("[EXECUTE CALL] Called Execute Episode for :", smiles_counter)
            ready_ids, remaining_ids = ray.wait(results,num_returns=num_processes*NUM_GPU)
            print("-----Episodes done---- Smiles: ", smiles_counter, " i:" , i)

            for temp in range(NUM_GPU):
                res_dict = ray.get(episode_actors[temp].print_dict.remote())
                print(res_dict)
            for temp in range(NUM_GPU):
                ray.get(episode_actors[temp].set_lock.remote(False))

        res_dict = []
        for temp in range(NUM_GPU):
            res_dict.append(ray.get(episode_actors[temp].print_dict.remote()))
        filename = "test/smile_"+str(smiles_counter) + ".pkl"

        with open(filename, 'wb') as handle:
            pickle.dump(res_dict, handle)

        for temp in range(NUM_GPU):
            ray.get(episode_actors[temp].clear_full.remote())


    # train_process.join()
if __name__ == '__main__':
    run()
