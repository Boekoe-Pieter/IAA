import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/pieter/IAA/Coding')

from NGAs_prefered import NGA_data  

import requests
import json
from tudatpy.data.sbdb import SBDBquery

url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

classes = ["COM", "PAR", "HYP"]
all_results = []

for sbclass in classes:
    request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF", "A2|DF"]}'
    request_dict = {
        'fields': 'spkid',
        'sb-class': sbclass, 
        'sb-cdata': request_filter,
    }
    response = requests.get(url, params=request_dict)
    if response.ok:
        all_results.extend(response.json().get("data", []))
        
A1_list, A2_list, A3_list = [], [], []

for body in all_results:
    target_sbdb = SBDBquery(body,full_precision=True,covariance="mat")
    com_name = target_sbdb["object"]["des"]
    full_name = target_sbdb["object"]["fullname"]
    save_body = com_name

    orbit_data = target_sbdb["orbit"]

    A1 = orbit_data["model_pars"].get("A1")
    A2 = orbit_data["model_pars"].get("A2")
    A3 = orbit_data["model_pars"].get("A3")
    DT = orbit_data["model_pars"].get("DT")

    A1 = A1.value if A1 is not None else 0
    A2 = A2.value if A2 is not None else 0
    A3 = A3.value if A3 is not None else 0
    DT = DT.value if DT is not None else 0

    A1_list.append(A1*10**8)
    A2_list.append(A2*10**8)
    A3_list.append(abs(A3*10**8))

A1_array = np.array(A1_list)
A2_array = np.array(A2_list)
A3_array = np.array(A3_list)

def plot_histograms_with_mean_subplot(category_name, A_list, B_list, C_list):
    data_lists = [A_list, B_list, C_list]
    xlabels = [r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$"]
    labels = ["A1","A2","A3"]
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(15,4))
    
    for i, (xlabel,labels, data, color) in enumerate(zip(xlabels,labels, data_lists, colors)):
        data_array = np.array(data)
        mean_val = np.median(data_array)
        std_val = np.std(data_array)
        print(mean_val)
        ax = plt.subplot(1, 3, i+1)
        ax.hist(data_array, bins=50, color=color, alpha=0.7, density=True)
        ax.axvline(mean_val, color='k', linestyle='--', label=f'Median = {mean_val:.3f}\nStd = {std_val:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(f'{category_name} - {labels}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"Histogram_{category_name}")
    plt.show()

A1_CoCO, A2_CoCO, A3_CoCO = [], [], []

for comet, data_dict in NGA_data.items():
    for category, entries in data_dict.items():
        for key, params in entries.items():
            A1_CoCO.append(params['A1'])
            A2_CoCO.append(params['A2'])
            A3_CoCO.append(params['A3'])

def plot_double(category_name, A_list_JPL, A_list_DB, B_list_JPL, B_list_DB, C_list_JPL, C_list_DB):
    data_pairs = [
        (A_list_JPL, A_list_DB, "A1"),
        (B_list_JPL, B_list_DB, "A2"),
        (C_list_JPL, C_list_DB, "A3"),
    ]
    
    xlabels = [r"$au/day^2$ $[10^{-8}]$"] * 3
    colors = ['r', 'b']
    
    plt.figure(figsize=(15,4))
    
    for i, ((data_JPL, data_DB, label), xlabel) in enumerate(zip(data_pairs, xlabels)):
        ax = plt.subplot(1, 3, i+1)

        arr_JPL = np.array(data_JPL)
        arr_DB = np.array(data_DB)

        ax.hist(arr_JPL, bins=20, color=colors[0], alpha=0.6, density=True, label="JPL")
        ax.hist(arr_DB, bins=50, color=colors[1], alpha=0.6, density=True, label="CoCO")
        
        for arr, col, name in zip([arr_JPL, arr_DB], colors, ["JPL", "Other"]):
            if len(arr) > 0:
                median_val = np.median(arr)
                std_val = np.std(arr)
                ax.axvline(median_val, color=col, linestyle='--', linewidth=1,
                           label=f"{name} Median={median_val:.3f}\nStd={std_val:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(f'{category_name} - {label}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"DoubleHistogram.png", dpi=300)
    plt.show()
plot_histograms_with_mean_subplot("JPL", A1_list, A2_list, A3_list)
plot_histograms_with_mean_subplot("Catalogue of Cometary Orbits", A1_CoCO, A2_CoCO, A3_CoCO)
plot_double("JPL and CoCO",A1_list, A1_CoCO, A2_list, A2_CoCO, A3_list, A3_CoCO)



# for comet, data_dict in NGA_data.items():
#     for category, entries in data_dict.items():
#         for key, params in entries.items():
#             A1_list.append(params['A1'])
#             A2_list.append(params['A2'])
#             A3_list.append(params['A3'])
#             if params['tau'] != 0:
#                 asymmetric.append(params['tau'])
#             if params['r0'] == 10:
#                 CO.append(params['r0'])

# A1_array = np.array(A1_list)
# A2_array = np.array(A2_list)
# A3_array = np.array(A3_list)

# print("Statistics of NGA parameters across all comets:")
# print(f"A1: mean={np.mean(A1_array):.3f}, std={np.std(A1_array):.3f}, min={np.min(A1_array):.3f}, max={np.max(A1_array):.3f}")
# print(f"A2: mean={np.mean(A2_array):.3f}, std={np.std(A2_array):.3f}, min={np.min(A2_array):.3f}, max={np.max(A2_array):.3f}")
# print(f"A3: mean={np.mean(A3_array):.3f}, std={np.std(A3_array):.3f}, min={np.min(A3_array):.3f}, max={np.max(A3_array):.3f}")

# nonzero_A1 = np.sum(A1_array != 0)
# nonzero_A2 = np.sum(A2_array != 0)
# nonzero_A3 = np.sum(A3_array != 0)

# print(f"Entries with nonzero A1: {nonzero_A1}/{len(A1_array)}")
# print(f"Entries with nonzero A2: {nonzero_A2}/{len(A2_array)}")
# print(f"Entries with nonzero A3: {nonzero_A3}/{len(A3_array)}")

# category_counts = {}

# for comet, data_dict in NGA_data.items():
#     for category in data_dict:
#         category_counts[category] = category_counts.get(category, 0) + len(data_dict[category])

# print("Number of NGA entries per category across all comets:")
# for cat, count in category_counts.items():
#     print(f"{cat}: {count}")

# print("Number of asymmetric NGA entries per category across all comets:")
# print(len(asymmetric))
# print("Number of CO-like NGA entries per category across all comets:")
# print(len(CO))

# A_pre, B_pre, C_pre = [], [], []
# A_full, B_full, C_full = [], [], []
# A_pre_co, B_pre_co, C_pre_co = [], [], []
# A_full_co, B_full_co, C_full_co = [], [], []

# for comet, data_dict in NGA_data.items():
#     for category, entries in data_dict.items():
#         if category == "Pre":
#             for key, params in entries.items():
#                 if params['r0'] ==10:
#                     A_pre_co.append(params['A1'])
#                     B_pre_co.append(params['A2'])
#                     C_pre_co.append(params['A3'])
#                 A_pre.append(params['A1'])
#                 B_pre.append(params['A2'])
#                 C_pre.append(params['A3'])
#         elif category == "Full":
#             for key, params in entries.items():
#                 if params['r0'] ==10:
#                     A_full_co.append(params['A1'])
#                     B_full_co.append(params['A2'])
#                     C_full_co.append(params['A3'])
#                 A_full.append(params['A1'])
#                 B_full.append(params['A2'])
#                 C_full.append(params['A3'])


# def plot_histograms_with_mean_subplot(category_name, A_list, B_list, C_list):
#     data_lists = [A_list, B_list, C_list]
#     xlabels = [r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$", r"$au/day^2$ $[10^{-8}]$"]
#     labels = ["A1","A2","A3"]
#     colors = ['r', 'g', 'b']
    
#     plt.figure(figsize=(15,4))
    
#     for i, (xlabel,labels, data, color) in enumerate(zip(xlabels,labels, data_lists, colors)):
#         data_array = np.array(data)
#         mean_val = np.mean(data_array)
#         std_val = np.std(data_array)
        
#         ax = plt.subplot(1, 3, i+1)
#         ax.hist(data_array, bins=20, color=color, alpha=0.7, density=True)
#         ax.axvline(mean_val, color='k', linestyle='--', label=f'Mean = {mean_val:.3f}\nStd = {std_val:.3f}')
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel('Density')
#         ax.set_title(f'{category_name} - {labels}')
#         ax.legend()
    
#     plt.tight_layout()
#     plt.savefig(f"Histogram_{category_name}")
#     plt.show()


# plot_histograms_with_mean_subplot("Pre", A_pre, B_pre, C_pre)

# plot_histograms_with_mean_subplot("Full", A_full, B_full, C_full)

# plot_histograms_with_mean_subplot("Pre-CO", A_pre_co, B_pre_co, C_pre_co)

# plot_histograms_with_mean_subplot("Full-CO", A_full_co, B_full_co, C_full_co)
# print(f"Entries with nonzero A1: {np.sum(np.array(A_pre) != 0)}/{len(A_pre)}")
# print(f"Entries with nonzero A2: {np.sum(np.array(B_pre) != 0)}/{len(A_pre)}")
# print(f"Entries with nonzero A3: {np.sum(np.array(C_pre) != 0)}/{len(A_pre)}")

# print(A_full, B_full, C_full )
# print(f"Entries with nonzero A1: {np.sum(np.array(A_full) != 0)}/{len(A_full)}")
# print(f"Entries with nonzero A2: {np.sum(np.array(B_full) != 0)}/{len(A_full)}")
# print(f"Entries with nonzero A3: {np.sum(np.array(C_full) != 0)}/{len(A_full)}")

# print(f"Entries with nonzero A1: {np.sum(np.array(A_pre_co) != 0)}/{len(A_pre_co)}")
# print(f"Entries with nonzero A2: {np.sum(np.array(B_pre_co) != 0)}/{len(A_pre_co)}")
# print(f"Entries with nonzero A3: {np.sum(np.array(C_pre_co) != 0)}/{len(A_pre_co)}")