import sys
import os
sys.path.append(os.getcwd())
import numpy as np

# from packages.states import get_tensor_state
# print(get_tensor_state(2,[0,0,0]))

data = np.load(f'./data/fidelity/noise_None_SIC_2/opt_2_shots_50000_0.npy')