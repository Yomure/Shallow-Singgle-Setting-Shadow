import sys
import os
sys.path.append(os.getcwd())
from tkinter import font
from turtle import color
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from sympy import true
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

y_axis_limit = (0.8, 1.05)
x_axis_limit = (0, 100)
noises = ['T1T2norst']
sics = ['SIC_1']
opts = ['opt_0','opt_1','opt_2']
sets = [(noise,sic,opt) for noise in noises for sic in sics for opt in opts]
sets.append(('None','SIC_1','opt_2'))
total_data_points_per_set = 2000000  # 每个set的数据点数
num_datasets = 20 # 数据集的数量
true_fidelity = 1
qubit_counts = [6]
set_name_mapping = {
    sets[0]: 'SIC 3-CNOT',
    sets[1]: 'SIC 2-CNOT',
    sets[2]: 'SIC 1-CNOT',
    sets[3]: 'Noiseless'
}
markers = ['o', 's', '^', '>', '<', 'v', 'p', '*', '+']
color = ['#FE3234', '#049DBF', '#B1D728' , '#9B237D', '#FD9900','y', 'k', 'w']
# color = []
dataset_index = 1
A = 10000
sample_step = 2000
estimated_fidelities = {}
for set_name in tqdm(sets, desc="Processing sets"):
    for num_qubits in tqdm(qubit_counts, desc=f"Processing qubit counts for set {set_name}"):
        if set_name == sets[3]:
            file_path = f'./data/fidelity/noise_{set_name[0]}_{set_name[1]}/{set_name[2]}_shots_{total_data_points_per_set}_19.npy'
            data = np.load(file_path, mmap_mode='r')[50000:50000+105000]
            acc = np.cumsum(data) / np.arange(1, len(data) + 1)
            estimated_fidelities[set_name] = acc
        else:
            file_path = f'./data/fidelity/noise_{set_name[0]}_{set_name[1]}/{set_name[2]}_shots_{total_data_points_per_set}_{dataset_index}.npy'
            data = np.load(file_path, mmap_mode='r')[A:A+105000]
            acc = np.cumsum(data) / np.arange(1, len(data) + 1)
            estimated_fidelities[set_name] = acc

plt.figure(figsize=(4,3.5))
for idx,set_name in enumerate(sets):
    sample_index = range(500, len(estimated_fidelities[set_name]), sample_step)
    label_index = [index // 1000 for index in sample_index]
    sample_data = [estimated_fidelities[set_name][index] for index in sample_index]
    new_set_name = set_name_mapping.get(set_name, set_name)
    current_marker = markers[idx % len(markers)]
    current_color = color[idx % len(color)]
    if set_name == sets[3]:
        plt.plot(label_index, sample_data, label=f'{new_set_name}', 
            linewidth=0.6, linestyle='-.', color='grey')
    else:
        plt.plot(label_index, sample_data, 
                    marker=current_marker, linewidth=0.8, markersize=1.5,color=current_color, label=f'{new_set_name}')
    # plt.plot(range(1,13), variances_by_set[set_name], label=f'{new_set_name}', 
    #          marker=current_marker, linewidth=1, markersize=2)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Times New Roman',
    # 'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{times}'
    })
# plt.xlim(512, 4097)
# plt.ylim(-2,-1)
# plt.xlim(1,32770)
plt.axhline(y=true_fidelity, linewidth=0.6, linestyle='-.', color='grey')
plt.ylim(*y_axis_limit)
plt.xlim(*x_axis_limit)
# plt.xticks(ticks=[512, 1024, 2048, 4096], labels=['512', '1024', '2048', '4096'])
# plt.xscale('log', base=2)
plt.xlabel('Shots($\\times10^3$)')
# plt.ylabel('$\\mathrm{log_2\\sqrt{Var}}$')
plt.ylabel('Fidelity')
# plt.legend(frameon = False)
plt.tight_layout()
plt.savefig('./graph/fidelity_shots_T1T2norst.png', dpi=600)
plt.show()