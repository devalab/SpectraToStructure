#CODE TO MEASURE TIME EFFICIENCY OF ENVIRONMENT [DEPRECATED]
import time
from .environment import *
import numpy as np
import random
from tqdm import tqdm

stepTimes = np.array([])
validTimes = np.array([])
validNewTimes = np.array([])
for i in range(10):    
    x = Env([7,0,1,0],[(25.2, 'T', 1), (25.2, 'T', 3), (26.1, 'T', 0), (26.1, 'T', 2), (26.1, 'T', 4), (50.1, 'D', 5), (204.7, 'D', 6)])
    for j in tqdm(range(10), desc="Action iter"):
        valid = x.state.valid_actions()
        # validTimes = np.append(validTimes, time.process_time() - timeStart)
        if np.sum(valid) == 0:
            break
        action = int(random.choice(np.argwhere(valid)))
        x.step(action)
    print(x)
    print(x.terminal_reward())
print("Average Step time: ", np.average(stepTimes))
print("Average Valid Action Time: ", np.average(validTimes))

# For the previous: environment:
# Average Step time:  0.0027921693296088486                                                                                                                                        
# Average Valid Action Time:  0.07053368033165826

# #After refactoring:
# Average Step time:  0.03225421156666663   #step time increased because terminal condition changed                                                                                                                                       
# Average Valid Action Time:  0.029156465826086966
