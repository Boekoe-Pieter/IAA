import matplotlib.pyplot as plt
import numpy as np
from NGAs import NGA_data  

total_comets = len(NGA_data)
print(f"Total comets in NGA_data: {total_comets}")

for comet, data_dict in NGA_data.items():
    total_entries = sum(len(data_dict[key]) for key in data_dict)
    print(f"{comet}: {total_entries} NGA entries")

A1_list, A2_list, A3_list = [], [], []
asymmetric = []
CO = []

for comet, data_dict in NGA_data.items():
    for category, entries in data_dict.items():
        for key, params in entries.items():
            A1_list.append(params['A1'])
            A2_list.append(params['A2'])
            A3_list.append(params['A3'])
            if params['tau'] != 0:
                asymmetric.append(params['tau'])
            if params['r0'] == 10:
                CO.append(params['r0'])

A1_array = np.array(A1_list)
A2_array = np.array(A2_list)
A3_array = np.array(A3_list)

print("Statistics of NGA parameters across all comets:")
print(f"A1: mean={np.mean(A1_array):.3f}, std={np.std(A1_array):.3f}, min={np.min(A1_array):.3f}, max={np.max(A1_array):.3f}")
print(f"A2: mean={np.mean(A2_array):.3f}, std={np.std(A2_array):.3f}, min={np.min(A2_array):.3f}, max={np.max(A2_array):.3f}")
print(f"A3: mean={np.mean(A3_array):.3f}, std={np.std(A3_array):.3f}, min={np.min(A3_array):.3f}, max={np.max(A3_array):.3f}")

nonzero_A1 = np.sum(A1_array != 0)
nonzero_A2 = np.sum(A2_array != 0)
nonzero_A3 = np.sum(A3_array != 0)

print(f"Entries with nonzero A1: {nonzero_A1}/{len(A1_array)}")
print(f"Entries with nonzero A2: {nonzero_A2}/{len(A2_array)}")
print(f"Entries with nonzero A3: {nonzero_A3}/{len(A3_array)}")

category_counts = {}

for comet, data_dict in NGA_data.items():
    for category in data_dict:
        category_counts[category] = category_counts.get(category, 0) + len(data_dict[category])

print("Number of NGA entries per category across all comets:")
for cat, count in category_counts.items():
    print(f"{cat}: {count}")

print("Number of asymmetric NGA entries per category across all comets:")
print(len(asymmetric))
print("Number of CO-like NGA entries per category across all comets:")
print(len(CO))

A_pre, B_pre, C_pre = [], [], []
A_full, B_full, C_full = [], [], []
A_pre_co, B_pre_co, C_pre_co = [], [], []
A_full_co, B_full_co, C_full_co = [], [], []

for comet, data_dict in NGA_data.items():
    for category, entries in data_dict.items():
        if category == "Pre":
            for key, params in entries.items():
                if params['r0'] ==10:
                    A_pre_co.append(params['A1'])
                    B_pre_co.append(params['A2'])
                    C_pre_co.append(params['A3'])
                A_pre.append(params['A1'])
                B_pre.append(params['A2'])
                C_pre.append(params['A3'])
        elif category == "Full":
            for key, params in entries.items():
                if params['r0'] ==10:
                    A_full_co.append(params['A1'])
                    B_full_co.append(params['A2'])
                    C_full_co.append(params['A3'])
                A_full.append(params['A1'])
                B_full.append(params['A2'])
                C_full.append(params['A3'])


def plot_histograms_with_mean_subplot(category_name, A_list, B_list, C_list):
    data_lists = [A_list, B_list, C_list]
    xlabels = [r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$"]
    labels = ["A1","A2","A3"]
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(15,4))
    
    for i, (xlabel,labels, data, color) in enumerate(zip(xlabels,labels, data_lists, colors)):
        data_array = np.array(data)
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        
        ax = plt.subplot(1, 3, i+1)
        ax.hist(data_array, bins=20, color=color, alpha=0.7, density=True)
        ax.axvline(mean_val, color='k', linestyle='--', label=f'Mean = {mean_val:.3f}\nStd = {std_val:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(f'{category_name} - {labels}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"Histogram_{category_name}")
    plt.show()


plot_histograms_with_mean_subplot("Pre", A_pre, B_pre, C_pre)

plot_histograms_with_mean_subplot("Full", A_full, B_full, C_full)

plot_histograms_with_mean_subplot("Pre-CO", A_pre_co, B_pre_co, C_pre_co)

plot_histograms_with_mean_subplot("Full-CO", A_full_co, B_full_co, C_full_co)
print(f"Entries with nonzero A1: {np.sum(np.array(A_pre) != 0)}/{len(A_pre)}")
print(f"Entries with nonzero A2: {np.sum(np.array(B_pre) != 0)}/{len(A_pre)}")
print(f"Entries with nonzero A3: {np.sum(np.array(C_pre) != 0)}/{len(A_pre)}")

print(A_full, B_full, C_full )
print(f"Entries with nonzero A1: {np.sum(np.array(A_full) != 0)}/{len(A_full)}")
print(f"Entries with nonzero A2: {np.sum(np.array(B_full) != 0)}/{len(A_full)}")
print(f"Entries with nonzero A3: {np.sum(np.array(C_full) != 0)}/{len(A_full)}")

print(f"Entries with nonzero A1: {np.sum(np.array(A_pre_co) != 0)}/{len(A_pre_co)}")
print(f"Entries with nonzero A2: {np.sum(np.array(B_pre_co) != 0)}/{len(A_pre_co)}")
print(f"Entries with nonzero A3: {np.sum(np.array(C_pre_co) != 0)}/{len(A_pre_co)}")