import time
from .environment import *
import numpy as np
import random
from tqdm import tqdm

stepTimes = np.array([])
validTimes = np.array([])
validNewTimes = np.array([])
for i in range(10):
    x = Env([19,1,0,0],[100,200,150,140,75,100,200,150,140,75,100,200,150,140,75,100,200,150,140])
    for j in tqdm(range(10), desc="Action iter"):
        timeStart = time.process_time()
        valid = x.state.valid_actions()
        validTimes = np.append(validTimes, time.process_time() - timeStart)
        timeStart = time.process_time()
        if np.sum(valid) == 0:
            break
        action = int(random.choice(np.argwhere(valid)))
        timeStart = time.process_time()
        x.step(action)
        stepTimes = np.append(stepTimes, time.process_time() - timeStart)
    print(x)
print("Average Step time: ", np.average(stepTimes))
print("Average Valid Action Time: ", np.average(validTimes))

# For the previous: environment:
# Average Step time:  0.0027921693296088486                                                                                                                                        
# Average Valid Action Time:  0.07053368033165826

# #After refactoring:
# Average Step time:  0.03225421156666663   #step time increased because terminal condition changed                                                                                                                                       
# Average Valid Action Time:  0.029156465826086966