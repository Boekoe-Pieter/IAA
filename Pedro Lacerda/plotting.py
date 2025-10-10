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

# -------------------------------
# data
# comets = ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/"
comet = 'C2001Q4'

comet_path = os.path.join(base_path, "data_"+comet)

simulator = 'Rebound_' # Rebound_ 'TUDAT_'

for sim_folder in sorted(os.listdir(comet_path)):
    sim_path = os.path.join(comet_path, sim_folder)
    data_file = os.path.join(sim_path, f"{simulator}Simulation_data.pkl")
    info_file = os.path.join(sim_path, f"{simulator}Simulation_info.pkl")

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    with open(info_file, "rb") as f:
        info = pickle.load(f)

    # -------------------------------
    # Gathering data
    Family = {
        "Nominal_norm":{},

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
    # print(data["Nominal_trajectory_times"])

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
            # if np.any(mask):
            #     idx_1AU = np.argmax(mask)
            #     print(arr_AU[idx_1AU])
            #     dict["Pos_div_1AU"][key] = dict["Clone_Divergence_Norm"][key][idx_1AU]
            #     dict["Vel_div_1AU"][key] = dict["Clone_divergence_vel_norm"][key][idx_1AU]
    
    # ----------------------------------------------------
    # Gathering data and storing in general_statistics
    compute_family(Family, data)

    def D_traject(cartesian, ax=None, label=None, color=None, alpha=1.0):
        if ax is None:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal', adjustable='box')

        ax.plot(cartesian[:, 0]/const.au, cartesian[:, 1]/const.au, cartesian[:, 2]/const.au, 
                label=label, color=color, alpha=alpha)
        ax.scatter(cartesian[0, 0]/const.au, cartesian[0, 1]/const.au, cartesian[0, 2]/const.au, marker="o", s=10, color=color)
        ax.scatter(cartesian[-1, 0]/const.au, cartesian[-1, 1]/const.au, cartesian[-1, 2]/const.au, marker="x", s=10, color=color)

        return ax

    def plot_ensemble(data, n_clones=20):
        trajectories = data["Monte_trajectory"]

        def draw_sun(ax):
            radius_sun = 696340 * 1000/const.au
            _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
            _x = radius_sun * np.cos(_u) * np.sin(_v)
            _y = radius_sun * np.sin(_u) * np.sin(_v)
            _z = radius_sun * np.cos(_v)
            ax.plot_wireframe(_x, _y, _z, color="orange", alpha=0.5, lw=0.5, zorder=0)


        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal", adjustable="box")

        
        clone_keys = list(trajectories.keys())
        selected_keys = random.sample(clone_keys, min(n_clones, len(clone_keys)))

        for key in selected_keys:
            cartesian = trajectories[key]
            ax.plot(cartesian[:, 0]/const.au, cartesian[:, 1]/const.au, cartesian[:, 2]/const.au, alpha=0.7)
            ax.scatter(cartesian[0, 0]/const.au, cartesian[0, 1]/const.au, cartesian[0, 2]/const.au,
                    marker="o", s=10, color="green")
            ax.scatter(cartesian[-1, 0]/const.au, cartesian[-1, 1]/const.au, cartesian[-1, 2]/const.au,
                    marker="x", s=10, color="red") 
        
        D_traject(np.array(data["Nominal_trajectory"]), ax=ax, label='JPL API / Horizons start', color="black", alpha=0.9)

        draw_sun(ax)

        # max_value = max(np.max(np.abs(np.array(traj))) for traj in trajectories.values())
        # ax.set_xlim([-max_value/const.au, max_value/const.au])
        # ax.set_ylim([-max_value/const.au, max_value/const.au])
        # ax.set_zlim([-max_value/const.au, max_value/const.au])
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_zlabel("z [AU]")

        ax.legend()
        plt.title({info['used_obs']})
        plt.show()

    # plot_ensemble(data, n_clones=20)
    if simulator == 'TUDAT_':
        mask = Family['Nominal_norm']/const.au <= 1.2
        if np.any(mask):
            idx_1AU = np.argmax(mask)
            time = data['Nominal_trajectory_times'][idx_1AU]
        
        time_JD = time_representation.seconds_since_epoch_to_julian_day(time)
        time_1AU = Time(time_JD, format='jd', scale='utc') 

        last_obs_str = info["Sim_time"].get("last obs")
        last_obs = Time(last_obs_str, format='isot', scale='utc')

        t1_tdb = time_1AU.tdb
        t2_tdb = last_obs.tdb

        dt_days = (t1_tdb - t2_tdb).to('day').value
        dt_days = round(dt_days, 3)

        times_sec = np.array(data['Nominal_trajectory_times']).flatten() 
        r_norm = np.array(Family['Nominal_norm']) / const.au

        t_obs_sec = time_representation.iso_string_to_epoch(str(t2_tdb))

        r_obs_AU = np.interp(t_obs_sec, times_sec, r_norm)
        
        perihelion_str = info["Sim_time"].get("End_iso")
        last_obs_str = info["Sim_time"].get("last obs")

        perihelion = datetime.fromisoformat(perihelion_str.replace("Z", "+00:00"))
        last_obs = datetime.fromisoformat(last_obs_str.replace("Z", "+00:00"))

        dt_days_peri = round((perihelion - last_obs).total_seconds() / 86400, 3)

    else:
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


# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Norm_peri"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n])/ 1e3 for n in dt]
positions = np.arange(len(dt)) * 2

plt.figure(figsize=(15, 8))
box = plt.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)  

plt.xticks(positions, [f"{n}" for n in dt],rotation=70)
plt.yscale('log')
plt.ylabel("Clone Position Divergence Norm at perihelion [km]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Position Divergence vs. Days to Perihelion {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{simulator}boxplot_{comet}_POS_Peri.pdf',dpi=300)
# plt.show()

# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Vel_Norm_peri"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n]) for n in dt]
positions = np.arange(len(dt)) * 2

plt.figure(figsize=(15, 8))
box = plt.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)  

plt.xticks(positions, [f"{n}" for n in dt],rotation=70)
plt.yscale('log')
plt.ylabel("Clone Velocity Divergence Norm at perihelion [m/s]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to Perihelion {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{simulator}boxplot_{comet}_VEL_Peri.pdf',dpi=300)
# plt.show()
# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Norm_AU"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n])/1000 for n in dt]
positions = np.arange(len(dt)) * 2

plt.figure(figsize=(15, 8))
box = plt.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)  

plt.xticks(positions, [f"{n}" for n in dt],rotation=70)
plt.yscale('log')
plt.ylabel("Clone Position Divergence Norm at 1.2 AU [km]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to 1.2 AU {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{simulator}boxplot_{comet}_POS_AU.pdf',dpi=300)
# plt.show()

# ---------------------------------------------------------------------------------------------

divergence = general_statistics["Clone_Divergence_Vel_Norm_AU"][comet]
dt = sorted(divergence.keys(), reverse=True)

data = [np.array(divergence[n]) for n in dt]
positions = np.arange(len(dt)) * 2

plt.figure(figsize=(15, 8))
box = plt.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

for patch in box["boxes"]:
    patch.set_facecolor("tab:blue")
    patch.set_alpha(0.6)  

plt.xticks(positions, [f"{n}" for n in dt],rotation=70)
plt.yscale('log')
plt.ylabel("Clone Velocity Divergence Norm at 1.2 AU [m/s]")
plt.xlabel(r"$\Delta{Days}$")
plt.title(f"Clone Velocity Divergence vs. Days to 1.2 AU {comet}\n{info['N_clones']} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f'{base_path}/{simulator}boxplot_{comet}_VEL_AU.pdf',dpi=300)
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
plt.savefig(f'{base_path}/{simulator}scatter_{comet}_Obsdate_AU.pdf',dpi=300)























# # -------------------------------
# # Simulation plots
# def plot_sampled(data, info_dict, covar_names=None):
#     Sampled_data, mean_data = data["Sampled_data"], data["Initial_condition"]
#     body = info_dict['Body']

#     if covar_names is None:
#         covar_names = ['x','y','z','vx','vy','vz']

#     orbital_labels = {
#         "x": "x [m]",
#         "y": "y [m]",
#         "z": "z [m]",
#         "vx": "vx [m/s]",
#         "vy": "vy [m/s]",
#         "vz": "vz [m/s]",
#     }

#     n_orb = min(6, len(covar_names))
#     nrows = (n_orb + 2) // 3
#     fig, axs = plt.subplots(nrows, 3, figsize=(12, 4*nrows))
#     axs = axs.flatten()

#     for i in range(n_orb):
#         param = covar_names[i]
#         label = orbital_labels.get(param, param)
#         axs[i].hist(Sampled_data[:, i], bins=30, alpha=0.6, label="Samples")
#         # axs[i].axvline(mean_data[i], color="red", linestyle="--", label="Mean" if i == 0 else "")
#         axs[i].set_xlabel(label)
#         axs[i].set_ylabel("Count")

#     axs[0].legend()
#     plt.tight_layout()
#     # plt.savefig(f"{img_save_dir}/Sampled_initial.pdf", dpi=300)
#     plt.show()
#     # plt.close()












# # plot_sampled(data,info)
# # D_traject(data['Nominal_trajectory']/const.au,info)

# # -------------------------------
# # Statistical plots
# def plot_divergence_timeline(Family, sample_dis,info_dict):
#     body,Int,dt = info_dict["Body"], info_dict["Integrator"],info_dict["timestep"]

#     Nf = (Family["Nominal_norm"][0] - Family["Nominal_norm"][-1]) / sample_dis
#     Nf_round = int(np.round(Nf))
#     sample_array_NGAf = np.array(Family["Nominal_norm"][::Nf_round])
#     NGAf_mu, NGAf_sig = np.array(Family["fit"]["mu"]), np.array(Family["fit"]["sigma"])

#     fig, ax2 = plt.subplots(1, 1, figsize=(15, 9), sharey=True)
#     fig.suptitle(f"{body}, Integrator: {Int}, dt: {dt} [sec]")

#     ax2.set_xlabel("Heliocentric distance")
#     ax2.plot(sample_array_NGAf, NGAf_mu/1000, color="red", label=r"$\mu$ [km]")
#     ax2.plot(sample_array_NGAf, NGAf_sig/1000, color="blue", label=r"$\sigma$ [km]")
#     ax2.axvline(x=1, linestyle="dotted", color="black", label="1 AU")
#     ax2.grid(True, which="both")

#     plt.tight_layout()
#     # plt.savefig(f"{img_save_dir}/stats_timeline.pdf",dpi=300)
#     plt.show()
#     # plt.close()

# def plot_histogram_peri(hist_data, simulation_info):
#     body = simulation_info['Body']
#     values = np.array(list(hist_data.values()))/ 1000

#     fig, axes = plt.subplots(1, 1, figsize=(14, 6), sharey=True)

#     mu, sigma = norm.fit(values)
#     count, bins, _ = axes.hist(values, bins=50, density=True,
#                                 color="steelblue", alpha=0.6, edgecolor="k", label="Histogram")

#     x = np.linspace(min(values), max(values), 2000)
#     pdf = norm.pdf(x, mu, sigma)
#     axes.plot(x, pdf, 'r-', lw=2, label=f"Normal fit\nμ={mu:.4e}, σ={sigma:.3e}")

#     axes.set_xlabel(rf"Deviation at perihelion [km]")
#     axes.set_ylabel("Probability density")
#     axes.grid(alpha=0.3)

#     fig.suptitle(
#         f"Position distribution of deviations at 1 AU for comet {body}\n"
#         # f"Integrator: {simulation_info['Integrator']}, dt: {simulation_info['dt']}, "
#         f"samples: {simulation_info['N_clones']}, Observations: {simulation_info['used_obs']}"
#     )

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # plt.savefig(f"{img_save_dir}/Histogram_1AU.pdf", dpi=300)
#     plt.show()
#     # plt.close()

# def plot_div_traj(data,info):
#     trajectories = data["Clone_Divergence_Norm"]
#     clone_norm = data["Clone_norm"]
#     for idx in range(len(trajectories)):
#         plt.plot(clone_norm[idx]/const.au,trajectories[idx]/1000)
#     plt.xlabel("Distance [AU]")
#     plt.ylabel(r"$||\Delta{r}|| [km]$")
#     plt.title(f"divergence of {info['N_clones']} clones, Number of observations: {info['used_obs']}\n Integrator: {info['Integrator']}{info['Integrator_type']}, timestep: {info['timestep']}")
#     plt.show()

# # plot_divergence_timeline(Family,0.2,info)
# # plot_histogram_peri(Family["Clone_Divergence_Norm_peri"],info)
# # plot_div_traj(Family,info)