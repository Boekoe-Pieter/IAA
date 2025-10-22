import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as patches

import pickle
import scipy.constants as const
import random
from datetime import datetime
from astropy.time import Time
import re

from tudatpy.astro import time_representation

import os
import sys

sys.path.append('/Users/pieter/IAA/Coding')

plt.rcParams.update({
        "font.size": 14,              # Base font size
        "axes.titlesize": 14,         # Title font size
        "axes.labelsize": 14,         # X/Y label font size
        "xtick.labelsize": 12,        # X tick label size
        "ytick.labelsize": 12,        # Y tick label size
        "legend.fontsize": 12,        # Legend font size
        "figure.titlesize": 18        # Figure title size (if using suptitle)
    })

general_statistics = {
    "Clone_Divergence_Norm_peri": {},  
    "Clone_Divergence_Vel_Norm_peri": {},  
    "Clone_Divergence_Norm_AU": {},  
    "Clone_Divergence_Vel_Norm_AU": {},  
}

obs_scatter = {
    "N_obs": {},
    "Date": {},
    "Helio": {},
}

valid_clone_dict = {
    "N_obs": {},
    "N_clones": {},
    "N_valid_clones": {},
}

# -------------------------------
# data
# comets = ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/"
comet = 'C2008A1'

comet_path = os.path.join(base_path, "data_"+comet)

def natural_key(name):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

pref_Nobs = 0
for sim_folder in sorted(os.listdir(comet_path), key=natural_key):
    sim_path = os.path.join(comet_path, sim_folder)
    data_file = os.path.join(sim_path, f"Rebound_Simulation_data.pkl")
    info_file = os.path.join(sim_path, f"Rebound_Simulation_info.pkl")

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    with open(info_file, "rb") as f:
        info = pickle.load(f)

    valid_clones = info["N_valid_clones"]
    # -------------------------------
    # Gathering data
    Family = {
        "Nominal_norm":{},
        "Fitted_norm":{},
        "Clone_norm": {},
        "Clone_Divergence": {},
        "Clone_Divergence_Norm": {},
        "Clone_Divergence_Norm_peri": {},                  
        "Clone_Divergence_Sample": {},

        "Clone_vel_norm":{},
        "Clone_divergence_vel":{},
        "Clone_divergence_vel_norm":{},
        "Clone_Divergence_Nor_vel_peri": {},                  

        "Pos_div_1AU": {},
        "Vel_div_1AU": {},
    }

    if info['used_obs'] < pref_Nobs:
        continue
    pref_Nobs = info['used_obs']

    def compute_family(dict, data):
        monte_sample = data['Monte_trajectory']
        Nominal_trajectory = data['Nominal_trajectory']
        Nominal_pos_norm = np.linalg.norm(Nominal_trajectory[:, 0:3], axis=1)
        dict["Nominal_norm"] = Nominal_pos_norm

        for key in monte_sample.keys():
            dict["Clone_norm"][key] = np.linalg.norm(monte_sample[key][:, 0:3], axis=1) 
            dict["Clone_vel_norm"][key] = np.linalg.norm(monte_sample[key][:, 3:], axis=1) 

            dict["Clone_Divergence"][key] = (monte_sample[key][:, 0:3] - Nominal_trajectory[:, 0:3])
            dict["Clone_Divergence_Norm"][key] = np.linalg.norm(dict["Clone_Divergence"][key][:, 0:3], axis=1)

            dict["Clone_divergence_vel"][key] = (monte_sample[key][:, 3:] - Nominal_trajectory[:, 3:])
            dict["Clone_divergence_vel_norm"][key] = np.linalg.norm(dict["Clone_divergence_vel"][key][:,:3], axis=1)

            arr = Nominal_pos_norm
            idx_peri = np.argmin(arr)

            dict["Clone_Divergence_Norm_peri"][key] = dict["Clone_Divergence_Norm"][key][idx_peri]
            dict["Clone_Divergence_Nor_vel_peri"][key] = dict["Clone_divergence_vel_norm"][key][idx_peri]

            arr_AU = np.array(arr)/const.au
            mask = arr_AU <= 1.2
            if np.any(mask):
                idx_1AU = np.argmax(mask)
                if idx_1AU > 0:
                    x0, x1 = arr_AU[idx_1AU - 1], arr_AU[idx_1AU]
                    y0_pos = dict["Clone_Divergence_Norm"][key][idx_1AU - 1]
                    y1_pos = dict["Clone_Divergence_Norm"][key][idx_1AU]
                    y0_vel = dict["Clone_divergence_vel_norm"][key][idx_1AU - 1]
                    y1_vel = dict["Clone_divergence_vel_norm"][key][idx_1AU]

                    frac = (1.2 - x0) / (x1 - x0)
                    pos_div_1AU = y0_pos + frac * (y1_pos - y0_pos)
                    vel_div_1AU = y0_vel + frac * (y1_vel - y0_vel)
                else:
                    pos_div_1AU = dict["Clone_Divergence_Norm"][key][idx_1AU]
                    vel_div_1AU = dict["Clone_divergence_vel_norm"][key][idx_1AU]

                dict["Pos_div_1AU"][key] = pos_div_1AU
                dict["Vel_div_1AU"][key] = vel_div_1AU

    # ----------------------------------------------------
    # Gathering data and storing in general_statistics
    compute_family(Family, data)

    def D_traject(cartesian, ax=None, label=None, color=None, linestyle="-", alpha=1.0, lw=1.2):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal', adjustable='box')

        ax.plot(cartesian[:, 0]/const.au, cartesian[:, 1]/const.au, cartesian[:, 2]/const.au,
                label=label, color=color, linestyle=linestyle, alpha=alpha, lw=lw)
        ax.scatter(cartesian[0, 0]/const.au, cartesian[0, 1]/const.au, cartesian[0, 2]/const.au, marker="o", s=10, color=color)
        ax.scatter(cartesian[-1, 0]/const.au, cartesian[-1, 1]/const.au, cartesian[-1, 2]/const.au, marker="x", s=10, color=color)
        return ax

    def plot_ensemble(data, info, n_clones=20):
        trajectories = data["Monte_trajectory"]

        perturbing_bodies = [
            "Mercury", "Venus", "Earth", "Luna",
            "Mars", "Phobos", "Deimos",
            "Jupiter", "Io", "Europa", "Ganymede", "Callisto",
            "Saturn", "Titan", "Rhea", "Iapetus", "Dione", "Tethys", "Enceladus", "Mimas",
            "Uranus", "Miranda", "Ariel", "Umbriel", "Titania", "Oberon",
            "Neptune", "Triton"
        ]

        planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        moons = [b for b in perturbing_bodies if b not in planets]

        def draw_sun(ax):
            radius_sun = 696340e3 /const.au
            _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
            _x = radius_sun * np.cos(_u) * np.sin(_v)
            _y = radius_sun * np.sin(_u) * np.sin(_v)
            _z = radius_sun * np.cos(_v)
            ax.plot_wireframe(_x, _y, _z, color="orange", alpha=0.5, lw=0.5, zorder=0)

        # --- Figure setup ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal", adjustable="box")

        # --- Monte Carlo clones (no legend) ---
        clone_keys = list(trajectories.keys())
        selected_keys = random.sample(clone_keys, min(n_clones, len(clone_keys)))
        for key in selected_keys:
            cartesian = trajectories[key]
            ax.plot(cartesian[:, 0]/const.au, cartesian[:, 1]/const.au, cartesian[:, 2]/const.au,
                    alpha=0.35, color="gray", lw=0.8)

        # --- Comet orbits ---
        D_traject(np.array(data["Nominal_trajectory"]), ax=ax,
                label='Nominal Orbit (JPL)', color="black", linestyle="-", alpha=0.9, lw=1.8)
        D_traject(np.array(data["Nominal_trajectory_fit"]), ax=ax,
                label='Fitted Orbit', color="red", linestyle=":", alpha=0.9, lw=1.8)

        # --- Perturbing bodies ---
        colors = plt.cm.tab20(np.linspace(0, 1, len(perturbing_bodies)))
        planet_handles, moon_handles = [], []

        for i, body in enumerate(perturbing_bodies):
            color = colors[i]
            linestyle = "-" if body in planets else ":"
            D_traject(np.array(data["perturbing_bodies"][body]), ax=ax,
                    color=color, linestyle=linestyle, alpha=0.9, lw=1.2)
            handle = Line2D([0], [0], color=color, lw=1.2, linestyle=linestyle)
            if body in planets:
                planet_handles.append((handle, body))
            else:
                moon_handles.append((handle, body))

        draw_sun(ax)

        # --- Axes limits ---
        max_value = max(np.max(np.abs(np.array(traj))) for traj in data["Nominal_trajectory"])
        ax.set_xlim([-max_value/const.au, max_value/const.au])
        ax.set_ylim([-max_value/const.au, max_value/const.au])
        ax.set_zlim([-max_value/const.au, max_value/const.au])
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_zlabel("z [AU]")

        # --- Grouped Legends ---
        comet_legend = [
            Line2D([0], [0], color='black', lw=1.2, label='Nasa JPL SBDB Fit'),
            Line2D([0], [0], color='red', lw=1.2, linestyle=":", label='Synthetic Observations Fit'),
            Line2D([0], [0], color='gray', lw=1.2, linestyle=":", label='Monte carlo samples')

        ]
        planet_legend = [h for h, lbl in planet_handles]
        moon_legend = [h for h, lbl in moon_handles]

        leg1 = ax.legend(handles=comet_legend, loc='upper left', title="Comet Orbits", fontsize=8)
        leg2 = ax.legend(planet_legend, [lbl for _, lbl in planet_handles],
                        loc='upper right', title="Planets", fontsize=8)
        leg3 = ax.legend(moon_legend, [lbl for _, lbl in moon_handles],
                        loc='lower left', title="Moons", fontsize=7, ncol=2)

        ax.add_artist(leg1)
        ax.add_artist(leg2)
        ax.add_artist(leg3)

        plt.title(f"{n_clones} Monte Carlo samples of comet {info['Body']} — {info['used_obs']} observations")
        plt.savefig(f"{base_path}/{comet}_3D_trajectory.pdf", dpi=300)

        plt.show()

    # plot_ensemble(data,info, n_clones=100)

    def diff_orbit_fits(data):
        
        JPL_elements = data["Nominal_trajectory"]
        Fit_elements = data["Nominal_trajectory_fit"]
        plt.figure(figsize=(8,5))
        plt.title(f'{info["Body"]},{info["N_valid_clones"]},{info["used_obs"]}')

        plt.plot(np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,linestyle="-",color="black")
        plt.plot(np.linalg.norm(Fit_elements[:,:3],axis=1)/const.au,np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,linestyle=":",color="red")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    # diff_orbit_fits(data)

    mask = Family['Nominal_norm']/const.au <= 1.2
    if np.any(mask):
        idx_1AU = np.argmax(mask)
        time = data['Nominal_trajectory_times'][idx_1AU]
    
    time_JD = time_representation.modified_julian_day_to_julian_day(time)
    time_1AU = Time(time_JD, format='jd', scale='utc') 

    last_obs_str = info["Sim_time"].get("last obs")
    last_obs = Time(last_obs_str, format='isot', scale='utc')

    t1_tdb = time_1AU.tdb
    t2_tdb = last_obs.tdb

    dt_days = (t1_tdb - t2_tdb).to('day').value
    dt_days = round(dt_days, 3)

    times_MJD = np.array(data['Nominal_trajectory_times']).flatten()

    r_norm = np.array(Family['Nominal_norm']) / const.au

    t_obs_MJD= t2_tdb.mjd

    r_obs_AU = np.interp(t_obs_MJD, times_MJD, r_norm)
    
    perihelion_str = info["Sim_time"].get("End_iso")
    last_obs_str = info["Sim_time"].get("last obs")

    perihelion = datetime.fromisoformat(perihelion_str.replace("Z", "+00:00"))
    last_obs = datetime.fromisoformat(last_obs_str.replace("Z", "+00:00"))

    dt_days_peri = round((perihelion - last_obs).total_seconds() / 86400, 3)   

    if comet not in general_statistics["Clone_Divergence_Norm_peri"]:
        general_statistics["Clone_Divergence_Norm_peri"][comet] = {}
        
    if comet not in general_statistics["Clone_Divergence_Vel_Norm_peri"]:
        general_statistics["Clone_Divergence_Vel_Norm_peri"][comet] = {}

    if comet not in general_statistics["Clone_Divergence_Norm_AU"]:
        general_statistics["Clone_Divergence_Norm_AU"][comet] = {}

    if comet not in general_statistics["Clone_Divergence_Vel_Norm_AU"]:
        general_statistics["Clone_Divergence_Vel_Norm_AU"][comet] = {}

    general_statistics["Clone_Divergence_Norm_peri"][comet][dt_days_peri] = list(Family["Clone_Divergence_Norm_peri"].values())
    general_statistics["Clone_Divergence_Vel_Norm_peri"][comet][dt_days_peri] = list(Family["Clone_Divergence_Nor_vel_peri"].values())
    general_statistics["Clone_Divergence_Norm_AU"][comet][dt_days] = list(Family["Pos_div_1AU"].values())
    general_statistics["Clone_Divergence_Vel_Norm_AU"][comet][dt_days] = list(Family["Vel_div_1AU"].values())

    obs_scatter.setdefault("N_obs", {}).setdefault(comet, []).append(info['used_obs'])
    obs_scatter.setdefault("Date", {}).setdefault(comet, []).append(last_obs_str)
    obs_scatter.setdefault("Helio", {}).setdefault(comet, []).append(r_obs_AU)

    valid_clone_dict.setdefault("N_obs", {}).setdefault(comet, []).append(info['used_obs'])
    valid_clone_dict.setdefault("N_clones", {}).setdefault(comet, []).append(info['N_clones'])
    valid_clone_dict.setdefault("N_valid_clones", {}).setdefault(comet, []).append(info['N_valid_clones'])

# ---------------------------------------------------------------------------------------------
def diff_orbit_fits():
    if comet == "C2001Q4":
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    elif comet == "C2008A1":
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    else:
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    
    JPL_elements = data["Nominal_trajectory"]
    Fit_elements = data["Nominal_trajectory_fit"]
    diff_elements = JPL_elements - Fit_elements
    
    diff = np.linalg.norm(diff_elements[:,:3],axis=1)
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    labels = ['x (km)', 'y (km)', 'z (km)']
    for i in range(3):
        axs[i].plot(data["Nominal_trajectory_times"], diff_elements[:, i] / 1e3, color='tab:blue')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time [MJD]')

    distance_sorted = sorted(np.linalg.norm(JPL_elements,axis=1) / const.au, reverse=True)
    axs[3].plot(distance_sorted, diff / 1e3, color='tab:orange')
    axs[3].set_ylabel(r'$||r_{diff}||$ (km)')
    axs[3].grid(True)
    axs[3].set_xlabel('Distance [AU]')
    axs[3].invert_xaxis()

    fig.suptitle(f'Difference between fitted state vector and JPL fitted, {info["used_obs"]} observations')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{base_path}/{comet}_Fit_SBDB_difference.pdf", dpi=300)
    # plt.show()

diff_orbit_fits()

def diff_orbit_fits():
    if comet == "C2001Q4":
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    elif comet == "C2008A1":
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    else:
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    
    JPL_elements = data["Nominal_trajectory"]
    Fit_elements = data["Nominal_trajectory_fit"]
    plt.figure(figsize=(8,5))
    plt.plot(np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,linestyle="-",color="black")
    plt.plot(np.linalg.norm(Fit_elements[:,:3],axis=1)/const.au,np.linalg.norm(JPL_elements[:,:3],axis=1)/const.au,linestyle=":",color="red")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.savefig(f"{base_path}/{comet}_Fit_SBDB_difference.pdf", dpi=300)
    # plt.show()

diff_orbit_fits()

def clone_orbits():
    if comet == "C2001Q4":
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2001Q4/Simulation_38/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    elif comet == "C2008A1":
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2008A1/Simulation_25/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    else:
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("Pedro Lacerda/data_C2013US10/Simulation_54/Rebound_Simulation_Info.pkl", "rb") as f:
            info = pickle.load(f)
    
    JPL_elements = data["Nominal_trajectory"]
    Fit_elements = data["Nominal_trajectory_fit"]
    Clones = data["Monte_trajectory"]
    diff_elements = JPL_elements - Fit_elements
    JPL_norm = np.linalg.norm(JPL_elements[:, 0:3], axis=1) 
    Fit_norm = np.linalg.norm(Fit_elements[:,:3],axis=1)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Difference to JPL SBDB, {info["used_obs"]} observations')
    for key in Clones.keys():
        diff_norm = np.linalg.norm(JPL_elements[:,0:3]-Clones[key][:, 0:3],axis=1)
        fit_jpl_dif = np.linalg.norm(diff_elements, axis=1) 
        axs.plot(JPL_norm/const.au,diff_norm/1000)
    axs.plot(JPL_norm/const.au,fit_jpl_dif/1000,label='Fitted orbit',color="black",linewidth=3.0)
    axs.invert_xaxis()
    plt.legend()
    plt.xlabel(f"Heliocentric distance [AU]")
    plt.ylabel(f"Divergence [km]")
    plt.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{base_path}/{comet}_Clone_difference.pdf", dpi=300)
    # plt.show()

clone_orbits()

def valid_clones_plot(valid_clone_dict):
    N_obs = np.array(valid_clone_dict["N_obs"][comet], dtype=float)
    N_clones = np.array(valid_clone_dict["N_clones"][comet], dtype=float)
    N_valid_clones = np.array(valid_clone_dict["N_valid_clones"][comet], dtype=float)

    valid_percent = (N_valid_clones / N_clones) * 100

    sort_idx = np.argsort(N_obs)
    N_obs = N_obs[sort_idx]
    valid_percent = valid_percent[sort_idx]

    plt.figure(figsize=(8,5))
    plt.plot(N_obs, valid_percent, 'o-', linewidth=1.8, markersize=6)
    plt.xlabel('Number of Observations', fontsize=12)
    plt.ylabel('Valid Clones (%)', fontsize=12)
    plt.title('Clone Validity vs. Number of Observations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{base_path}/{comet}_Valid_observations.pdf", dpi=300)
    # plt.show()

valid_clones_plot(valid_clone_dict)

# ---------------------------------------------------------------------------------------------
extra_time = 15
height_pos = 1000
height_vel = 1

divergence = general_statistics["Clone_Divergence_Norm_peri"][comet]
dt = sorted(divergence.keys(), reverse=True)

dt = np.array(dt, dtype=float)

data = [np.array(divergence[n]) / 1e3 for n in dt]

fig, ax = plt.subplots(figsize=(15, 8))

box = plt.boxplot(
    data,
    positions=dt,          
    widths=5, 
    patch_artist=True,
    manage_ticks=False
)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)

time = np.arange(max(dt),min(dt)-extra_time,-extra_time)

plt.xticks(time,rotation=70)
width = -60 

x_start = 60      
y_start = 0
square = patches.Rectangle(
    (x_start, y_start),   
    width,                 
    height_pos,                
    linewidth=1,
    edgecolor='black',
    facecolor='green',
    alpha=0.3             
)
ax.add_patch(square)

plt.gca().invert_xaxis()
plt.yscale("log")
plt.ylabel("Clone Position Divergence Norm at perihelion [km]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(
    f"Clone Position Divergence vs. Days to Perihelion {comet}\n"
    f"{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec"
)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"{base_path}/{comet}_boxplot_POS_Peri.pdf", dpi=300)
# plt.show()

# ---------------------------------------------------------------------------------------------
divergence = general_statistics["Clone_Divergence_Vel_Norm_peri"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n]) for n in dt]
positions = np.arange(len(dt)) * 2

fig, ax = plt.subplots(figsize=(15, 8))

box = plt.boxplot(
    data,
    positions=dt,          
    widths=5, 
    patch_artist=True,
    manage_ticks=False
)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)

time = np.arange(max(dt),min(dt)-extra_time,-extra_time)

plt.xticks(time,rotation=70)
width = -60

x_start = 60      
y_start = 0
square = patches.Rectangle(
    (x_start, y_start),   
    width,                 
    height_vel,                
    linewidth=1,
    edgecolor='black',
    facecolor='green',
    alpha=0.3             
)
ax.add_patch(square)

plt.gca().invert_xaxis()

plt.yscale("log")
plt.ylabel("Clone Velocity Divergence Norm at perihelion [m/s]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to Perihelion {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{comet}_boxplot_VEL_Peri.pdf',dpi=300)
# plt.show()
# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Norm_AU"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n])/1000 for n in dt]
positions = np.arange(len(dt)) * 2

fig, ax = plt.subplots(figsize=(15, 8))

box = plt.boxplot(
    data,
    positions=dt,          
    widths=5, 
    patch_artist=True,
    manage_ticks=False
)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)

time = np.arange(max(dt),min(dt)-extra_time,-extra_time)

plt.xticks(time,rotation=70)

plt.gca().invert_xaxis()

plt.yscale("log")
plt.ylabel("Clone Position Divergence Norm at 1.2 AU [km]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to 1.2 AU {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{comet}_boxplot_POS_AU.pdf',dpi=300)
# plt.show()

# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Vel_Norm_AU"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n]) for n in dt]
positions = np.arange(len(dt)) * 2

fig, ax = plt.subplots(figsize=(15, 8))

box = plt.boxplot(
    data,
    positions=dt,          
    widths=5, 
    patch_artist=True,
    manage_ticks=False
)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)

time = np.arange(max(dt),min(dt)-extra_time,-extra_time)

plt.xticks(time,rotation=70)
width = time[-1]-60 if time[-1]<0 else -60

plt.gca().invert_xaxis()

plt.yscale("log")
plt.ylabel("Clone Velocity Divergence Norm at 1.2 AU [m/s]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to 1.2 AU {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{comet}_boxplot_VEL_AU.pdf',dpi=300)
# plt.show()

dates_str = obs_scatter["Date"][comet]
helios = obs_scatter["Helio"][comet]
n_obs = obs_scatter["N_obs"][comet]

dates = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in dates_str]

fig, ax = plt.subplots(figsize=(15, 8))

sc = ax.scatter(dates, helios, c='dodgerblue', s=60, edgecolor='k', alpha=0.8)

for x, y, label in zip(dates, helios, n_obs):
    ax.text(x, y + 0.03, str(label), ha='center', va='bottom', fontsize=8)

peri_date = datetime.fromisoformat(info['Sim_time'].get('End_iso').replace("Z", "+00:00"))
ax.axvline(x = peri_date, color = 'b', label = 'Perihelion')
ax.set_xlabel("Last Observation Date (UTC)")
ax.set_ylabel("Heliocentric Distance [AU]")
ax.set_title(f"Observation Scatter for {comet}")
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.savefig(f'{base_path}/{comet}_Observation_scatter.pdf',dpi=300)