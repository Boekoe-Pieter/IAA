import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
import matplotlib.cm as cm
import scipy.constants as const
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.patches as mpatches
import random
from datetime import datetime
from astropy.time import Time

from tudatpy.astro import time_representation

import os
import sys

sys.path.append('/Users/pieter/IAA/Coding')
# ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/"
comet = 'C2001Q4'

comet_path = os.path.join(base_path, "data_"+comet)

for sim_folder in sorted(os.listdir(comet_path)):
    sim_path = os.path.join(comet_path, sim_folder)
    data_file_TUDAT = os.path.join(sim_path, f"TUDAT_Simulation_data.pkl")
    info_file_TUDAT = os.path.join(sim_path, f"TUDAT_Simulation_info.pkl")
    data_file_Rebound = os.path.join(sim_path, f"Rebound_Simulation_data.pkl")
    info_file_Rebound = os.path.join(sim_path, f"Rebound_Simulation_info.pkl")

    with open(data_file_TUDAT, "rb") as f:
        data_tudat = pickle.load(f)
    with open(info_file_TUDAT, "rb") as f:
        info_tudat = pickle.load(f)
    with open(data_file_Rebound, "rb") as f:
        data_Rebound = pickle.load(f)
    with open(info_file_Rebound, "rb") as f:
        info_Rebound = pickle.load(f)

    tudat_nominal = data_tudat['Nominal_trajectory']
    rebound_nominal = data_Rebound['Nominal_trajectory']

    tudat_xyz = np.array(data_tudat['Nominal_trajectory'])
    rebound_xyz = np.array(data_Rebound['Nominal_trajectory'])

    AU = 1.495978707e11
    tudat_xyz_AU = tudat_xyz / AU
    rebound_xyz_AU = rebound_xyz / AU
    difference = tudat_xyz_AU-rebound_xyz_AU

    t_tudat = np.arange(len(tudat_xyz_AU))
    t_rebound = np.arange(len(rebound_xyz_AU))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['x (AU)', 'y (AU)', 'z (AU)']
    for i in range(3):
        axs[i].plot(t_tudat, difference[:, i], label='difference', color='tab:blue')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Time step index')

    fig.suptitle('3D Cartesian Position Comparison (Tudat vs Rebound)', fontsize=14)
    plt.tight_layout()
    plt.show()

    norm_tudat = np.linalg.norm(tudat_xyz,axis=1)
    norm_rebound= np.linalg.norm(rebound_xyz,axis=1)
    plt.figure(figsize=(15,8))
    plt.plot(norm_tudat/AU,norm_tudat/AU,label='tudat',linestyle='-')
    plt.plot(norm_rebound/AU,norm_rebound/AU,label='rebound',linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    


    def D_traject(tudat,rebound, alpha=1.0):
        cartesian_tudat = np.array(tudat)
        cartesian_rebound = np.array(rebound)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal', adjustable='box')

        ax.plot(cartesian_tudat[:, 0]/const.au, cartesian_tudat[:, 1]/const.au, cartesian_tudat[:, 2]/const.au, 
                label="TUDAT", color="#BBCCEE", alpha=alpha)
        ax.scatter(cartesian_tudat[0, 0]/const.au, cartesian_tudat[0, 1]/const.au, cartesian_tudat[0, 2]/const.au, marker="o", s=10, color="Green")
        ax.scatter(cartesian_tudat[-1, 0]/const.au, cartesian_tudat[-1, 1]/const.au, cartesian_tudat[-1, 2]/const.au, marker="x", s=10, color="Red")
        
        ax.plot(cartesian_rebound[:, 0]/const.au, cartesian_rebound[:, 1]/const.au, cartesian_rebound[:, 2]/const.au, 
                label="TUDAT", color="#CCEEFF", alpha=alpha)
        ax.scatter(cartesian_rebound[0, 0]/const.au, cartesian_rebound[0, 1]/const.au, cartesian_rebound[0, 2]/const.au, marker="o", s=10, color="Green")
        ax.scatter(cartesian_rebound[-1, 0]/const.au, cartesian_rebound[-1, 1]/const.au, cartesian_rebound[-1, 2]/const.au, marker="x", s=10, color="Red")

        plt.show()
    
    D_traject(tudat_nominal,rebound_nominal, alpha=0.9)
