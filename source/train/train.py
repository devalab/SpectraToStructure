#THESE ARE THE LOG FILES USED:
TRAIN_P_FILE = "./prior_log.txt"
TRAIN_V_FILE = "./value_log.txt"

import torch
import dgl
import numpy as np
import torch.nn as nn

import time
from copy import deepcopy
import sys
import torch.multiprocessing
import random
torch.multiprocessing.set_sharing_strategy('file_system')
import ray
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import os



PRETRAINING_SIZE = 3000
# PRETRAINING_SIZE = 100
BATCH_SIZE = 32

device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum+l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)

def loss_pi(predicted,target):
    loss = torch.sum(-target*((1e-8+predicted).log()),1)
    return torch.mean(loss)

def loss_v(predicted,target):
    target = target.unsqueeze(-1)
    criterion = nn.MSELoss().to(device1)
    loss = criterion(predicted,target)
    return loss

def collate(samples):
    graph, mask, indexmask,pib,Vb = map(list, zip(*samples))
    graph = dgl.batch(graph)
    len_matrix = get_len_matrix(graph.batch_num_nodes())
    mask = torch.cat(mask,0)
    return_mask = mask.clone()
    mask[return_mask==0] = -1e8
    mask[return_mask!=0] = 0
    indexmask = torch.cat(indexmask,0)
    pib = torch.cat(pib,0)
    Vb = torch.cat(Vb,0)
    return [graph, len_matrix,mask, indexmask,], pib,Vb




@ray.remote(num_gpus=1)
def train(model, value_model,episode_actor,wandb):
    epoch = 0
    # prior_file = open("./train_prior.log","w") #NOT USED ANYMORE
    # value_file = open("./train_value.log","w") #NOT USED ANYMORE
    prior_file, value_file = None, None
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # wandb.log(model)
    # wandb.log(value_model)
    class Dataclass(Dataset):
        def __init__(self,dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            exp = self.dataset[idx]
            graph = exp[0]['mol_graph']
            mask = exp[0]['action_mask']
            indexmask = exp[0]['index_mask']
            pib = exp[2]
            Vb = exp[1]
            return [graph,torch.FloatTensor(mask).unsqueeze(0),torch.FloatTensor(indexmask).unsqueeze(0),torch.FloatTensor(pib).unsqueeze(0),Vb]
    
    model.to(device1)

    while(ray.get(episode_actor.get_size.remote()) < PRETRAINING_SIZE):
        True
    print("--------Training Started--------")
    while(1):
        if (ray.get(episode_actor.get_size.remote()) >= PRETRAINING_SIZE ): 
            experiences = ray.get(episode_actor.get_experience.remote())

            print("DB Size: ",len(experiences))
            train_dataset = Dataclass(experiences)
            prior_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
            train_prior(1, model, optimizer, prior_loader,wandb,prior_file)

            epoch += 1

            filename = "./saved_models/prior" +str(epoch//50) +".state"
            torch.save(model.state_dict(), filename)

        if (not ray.get(episode_actor.get_lock.remote())) and (not ray.get(episode_actor.get_queue.remote())):
            time.sleep(0.2)
            model.to(torch.device("cpu"))
            print("------------model sending--------------")
            print(model.state_dict()['final.bias'])
            print("------------model sending--------------")
            ray.get(episode_actor.add_model.remote([deepcopy(model.state_dict()), "value_model"]))
            ray.get(episode_actor.set_queue.remote(True))

            model.to(device1)

            


def train_prior(epochs,model,optimizer,database,wandb,file):
    loss_prior_arr = []
    loss_value_arr = []
    loss_arr = []
    start_time = time.time()
    print("--------Training Started--------")
    for i in range(epochs):
        for samples in database:
            optimizer.zero_grad()
            data,pib,vb = samples
            output = model(data)
            loss1 = loss_pi(output[0],pib.to(device1))    #final
            loss2 = loss_v(output[1],vb.to(device1))    #final
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
            loss_prior_arr.append(loss1.item())
            loss_value_arr.append(loss2.item())
    print("-----Training Ended----- in {} seconds-----{}".format((time.time() - start_time),np.mean(loss_arr)))

    curTime = datetime.now().strftime("%d %b %H:%M:%S :: ")
    with open(TRAIN_P_FILE, "a") as outFile:
        outFile.write("{} Train Loss Prior: {}\n".format(curTime, np.mean(loss_prior_arr)))

    with open(TRAIN_V_FILE, "a") as outFile:
        outFile.write("{} Train Loss Value: {}\n".format(curTime, np.mean(loss_value_arr)))
        
        
    
    sys.stdout.flush()
    return np.mean(loss_arr)


def train_value(epochs,model,optimizer,database,wandb,file):
    loss_arr = []
    print("--------ValueTraining Started--------")
    start_time = time.time()
    for i in range(epochs):
        for samples in database:
            optimizer.zero_grad()
            data,pib,vb = samples
            output = model(data[0].to(device2))
            loss = loss_v(output,vb.to(device2))
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())

    print("-----Value Training Ended----- in {} seconds-----{}".format((time.time() - start_time),np.mean(loss_arr)))
    sys.stdout.flush()
    # wandb.log({"Train Value Loss":np.mean(loss_arr)})
    file.write("Train Value Loss: "+ str(np.mean(loss_arr)) +"\n")
    # sys.stdout.flush()
    return np.mean(loss_arr)
