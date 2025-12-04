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

# all_sets = ['opt_0', 'opt_1', 'opt_2']
sics = ['SIC_1','SIC_2']
opts = ['opt_2']
sets = [(sic,opt) for sic in sics for opt in opts]
total_data_points_per_set = 2000000  # 每个set的数据点数
num_datasets = 20 # 数据集的数量
group_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  
true_fidelity = 1
# true_fidelity = 0.803125
qubit_counts = [6]
variances_by_set = {s: [] for s in sets}
set_name_mapping = {
    'opt_0': 'Optimization 1',
    'opt_1': 'Optimization 2',
    'opt_2': 'Optimization 3',
    
}
markers = ['o', 's', '^', '>', '<', 'v', 'p', '*', '+']
color = ['#FE3234', '#049DBF', '#B1D728' , '#9B237D', '#FD9900''y', 'k', 'w']
# color = []

for set_name in tqdm(sets, desc="Processing sets"):
    for num_qubits in tqdm(qubit_counts, desc=f"Processing qubit counts for set {set_name}"):
        for group_size in tqdm(group_sizes, desc=f"Processing group sizes for set {set_name}, qubit count {num_qubits}"):
            a_values = []
            for dataset_index in range(num_datasets):
                file_path = f'./data/fidelity/noise_None_{set_name[0]}/{set_name[1]}_shots_{total_data_points_per_set}_{dataset_index}.npy'
                data = np.load(file_path)
                if group_size <= 64:
                    num_a_values = 8000
                else:
                    num_a_values = total_data_points_per_set // group_size
                for k in range(0, group_size*num_a_values , group_size):
                    subset = data[k:k + group_size]
                    a = np.mean(subset)
                    a_values.append(a)
            print('\n',len(a_values))
            
            b = np.mean([(a - true_fidelity) ** 2 for a in a_values])
            variances_by_set[set_name].append(np.log2(np.sqrt(b)))

# 绘制每个set，每个qubit数量的方差曲线
# ax = plt.figure(figsize=(2.775,2)).gca()  # 获取当前轴的实例
plt.figure(figsize=(3.375,3))
# ax = plt.figure(figsize=(3.375,3)).gca()  # 获取当前轴的实例
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for idx,set_name in enumerate(sets):
    new_set_name = set_name_mapping.get(set_name, set_name)
    current_marker = markers[idx % len(markers)]
    current_color = color[idx % len(color)]
    # if set_name in ['PRA_86_062107', 'PRA_86_062107_refined_method2', 'fiducial_generate_vector', 'fiducial_generate_vector_refined_method2', 'basic_bell'] :
    if set_name == 'nonoise':
        plt.plot(group_sizes, variances_by_set[set_name], label=f'{new_set_name}', 
            linewidth=0.6, linestyle='-.', color='grey')
    else:
        plt.plot(group_sizes, variances_by_set[set_name], 
                    marker=current_marker, linewidth=0.8, markersize=1.5,color=current_color, label=f'{new_set_name}')
    # plt.plot(range(1,13), variances_by_set[set_name], label=f'{new_set_name}', 
    #          marker=current_marker, linewidth=1, markersize=2)
    print(len(variances_by_set[set_name]))
print(len(group_sizes))
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Times New Roman',
    # 'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{times}'
    })
# plt.xlim(512, 4097)
# plt.ylim(-2,-1)
plt.xlim(1, 4097)
# plt.xticks(ticks=[512, 1024, 2048, 4096], labels=['512', '1024', '2048', '4096'])
plt.xscale('log', base=2)
plt.xlabel('Shots')
# plt.ylabel('$\\mathrm{log_2\\sqrt{Var}}$')
plt.ylabel('$\\mathrm{log_2\\sqrt{Var}}$')
plt.legend(frameon = False)
plt.tight_layout()
# plt.savefig('./graph/fidelity_variance_shots_6qubit_r.2.png', dpi=600)
plt.show()