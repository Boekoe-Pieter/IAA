import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import numpy as np
import random
import matplotlib.patches as patches

import pickle

from tudatpy import constants

AU = constants.ASTRONOMICAL_UNIT
day = constants.JULIAN_DAY

NGA_False = "Coding/W79 Estimation/Sim_data/Data_NGA_False.pkl"
NGA_True = "Coding/W79 Estimation/Sim_data/Data_NGA_True.pkl"
NGA_Est = "Coding/W79 Estimation/Sim_data/Data_NGA_Est.pkl"

with open(NGA_False, "rb") as f:
    NGA_False_Data = pickle.load(f)
with open(NGA_True, "rb") as f:
    NGA_True_Data = pickle.load(f)
with open(NGA_Est, "rb") as f:
    NGA_Est_Data = pickle.load(f)

saving_dir = f"Coding/W79 Estimation/"

Family = {
    "Truth_norm":{},

    "Fitted_norm":{},

    "Clone_norm": {},
    "Clone_Divergence": {},
    "Clone_Divergence_Norm": {},
    "Clone_Divergence_Norm_peri": {},                  

    "Clone_vel_norm":{},
    "Clone_divergence_vel":{},
    "Clone_divergence_vel_norm":{},
    "Clone_Divergence_Nor_vel_peri": {},                  

    "Pos_div_1AU": {},
    "Vel_div_1AU": {},
    }

general_statistics = {
    "Clone_Divergence_Norm_peri_DPERI": {},  
    "Clone_Divergence_Vel_Norm_peri_DPERI": {},
    "Clone_Divergence_Norm_peri_NOBS": {},  
    "Clone_Divergence_Vel_Norm_peri_NOBS": {},
}


# finding 3 and 1 AU Time to Perihelion
Truth_trajectory_NGA = NGA_True_Data['Truth_Reference_trajectory']
Truth_AU = np.linalg.norm(Truth_trajectory_NGA[:,:3],axis=1)/AU
Truth_times_NGA = NGA_True_Data['Truth_Reference_trajectory_times']
Mask1 = Truth_AU<=1.0
idx_1AU = np.argmax(Mask1)
Time_1AU = Truth_times_NGA[idx_1AU]
Time_to_Peri = Truth_times_NGA[-1] - Time_1AU
DT_1AU = Time_to_Peri/day

Mask3 = Truth_AU<=3
idx_3AU = np.argmax(Mask3)
Time_3AU = Truth_times_NGA[idx_3AU]
Time_to_Peri = Truth_times_NGA[-1] - Time_3AU
DT_3AU = Time_to_Peri/day


def compute_family(family_dict, data,label):
    comet = f"C2001Q4_{label}"

    if comet not in general_statistics["Clone_Divergence_Norm_peri_DPERI"]:
        general_statistics["Clone_Divergence_Norm_peri_DPERI"][comet] = {}
    if comet not in general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"]:
        general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"][comet] = {}

    if comet not in general_statistics["Clone_Divergence_Norm_peri_NOBS"]:
        general_statistics["Clone_Divergence_Norm_peri_NOBS"][comet] = {}
    if comet not in general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"]:
        general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"][comet] = {}

    monte_sample = data['Montecarlo_trajectory']
    Truth_trajectory = data['Truth_Reference_trajectory']
    Truth_pos_norm = np.linalg.norm(Truth_trajectory[:, 0:3], axis=1)
    family_dict["Truth_norm"] = Truth_pos_norm

    for sim, sim_data in monte_sample.items():
        for key, traj in sim_data.items():
            pos = traj[:, 0:3]
            vel = traj[:, 3:]

            # ----------------------------------------------------------------------
            # Compute divergence
            family_dict["Clone_norm"].setdefault(sim, {})[key] = np.linalg.norm(pos, axis=1)
            family_dict["Clone_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(vel, axis=1)

            family_dict["Clone_Divergence"].setdefault(sim, {})[key] = pos - Truth_trajectory[:, 0:3]
            family_dict["Clone_Divergence_Norm"].setdefault(sim, {})[key] = np.linalg.norm(
                family_dict["Clone_Divergence"][sim][key], axis=1
            )

            family_dict["Clone_divergence_vel"].setdefault(sim, {})[key] = vel - Truth_trajectory[:, 3:]
            family_dict["Clone_divergence_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(
                family_dict["Clone_divergence_vel"][sim][key], axis=1
            )

            # ----------------------------------------------------------------------
            # Find perihelion divergence
            idx_peri = np.argmin(Truth_pos_norm)

            family_dict["Clone_Divergence_Norm_peri"].setdefault(sim, {})[key] = \
                family_dict["Clone_Divergence_Norm"][sim][key][idx_peri]
            family_dict["Clone_Divergence_Nor_vel_peri"].setdefault(sim, {})[key] = \
                family_dict["Clone_divergence_vel_norm"][sim][key][idx_peri]
            
            # ----------------------------------------------------------------------
            # timeline
            perihelion = data["Truth_Reference_trajectory_times"][-1]
            last_obs = data["observation_times"][sim][-1]

            N_obs = data["Sim_info"][sim].get("N_obs")
            dt_days_peri = (perihelion - last_obs) / constants.JULIAN_DAY

            # ----------------------------------------------------------------------
            # Saving to dictionary             
            general_statistics["Clone_Divergence_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                family_dict["Clone_Divergence_Norm_peri"][sim][key]
            )

            general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])



            general_statistics["Clone_Divergence_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                family_dict["Clone_Divergence_Norm_peri"][sim][key]
            )

            general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])

def boxplot_all(extra_time=15):
    labels = ["False", "True", "Est"]
    plot_lables = ["PG Reference","NG Reference","NG Estimated"]
    comets = [f"C2001Q4_{label}" for label in labels]
    stats = general_statistics

    def make_boxplot_multiple(stat_dict_name, ylabel, title, scale=1.0, save_name=None, xlabel=r"$\Delta{Days}$ to Perihelion"):
        all_keys = sorted(set().union(*[stat_dict.keys() for stat_dict in 
                                        [stats[stat_dict_name][c] for c in comets if c in stats[stat_dict_name]]]),
                          reverse=True)

        fig, ax = plt.subplots(figsize=(15, 8))
        width = 3  
        spacing = 0  

        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, comet in enumerate(comets):
            positions = [x - spacing/2 + i*(spacing/len(labels)) for x in all_keys]
            data = [np.array(stats[stat_dict_name][comet].get(x, [])) / scale for x in all_keys]
            box = plt.boxplot(data, positions=positions, widths=width, patch_artist=True, manage_ticks=False)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.6)
            plt.plot([], [], color=colors[i], label=plot_lables[i])

        if isinstance(all_keys[0], (int, float)) and xlabel == r"$\Delta{Days}$ to Perihelion":
            plt.gca().invert_xaxis()
            plt.xticks(all_keys, rotation=70)
            plt.axvline(x = DT_3AU, color = 'black', linestyle = '--', label = '3AU line')
            plt.text(DT_3AU - 2, plt.ylim()[1]*0.6, '3AU line', rotation=90,
                    va='center', ha='left', fontsize=9, color='black')
            
            plt.axvline(x = 60, color = 'black', linestyle = '--',  label = '60 days line')
            plt.text(60 - 2, plt.ylim()[1]*0.6, '60 days line', rotation=90,
                    va='center', ha='left', fontsize=9, color='black')
                 
            plt.axvline(x = DT_1AU, color = 'black', linestyle = '--',  label = '1AU line')
            plt.text(DT_1AU - 2, plt.ylim()[1]*0.6, '1AU line', rotation=90,
                    va='center', ha='left', fontsize=9, color='black')
            
        plt.yscale("log")
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        if save_name:
            plt.savefig(f"{save_name}.pdf", dpi=300)
        plt.close()

    # ------------------------------------------------------
    # Position vs Days
    make_boxplot_multiple(
        "Clone_Divergence_Norm_peri_DPERI",
        ylabel="Clone Position Divergence Norm at perihelion [km]",
        title="Clone Position Divergence vs. Days to Perihelion",
        scale=1e3,
        save_name=f"{saving_dir}/Stat_Position_boxplot_dtPeri_All"
    )
    
    # ------------------------------------------------------
    # Velocity vs Days
    make_boxplot_multiple(
        "Clone_Divergence_Vel_Norm_peri_DPERI",
        ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
        title="Clone Velocity Divergence vs. Days to Perihelion",
        scale=1.0,
        save_name=f"{saving_dir}/Stat_Velocity_boxplot_dtPeri_All"
    )
    
    # ------------------------------------------------------
    # Position vs NOBS
    make_boxplot_multiple(
        "Clone_Divergence_Norm_peri_NOBS",
        ylabel="Clone Position Divergence Norm at perihelion [km]",
        title="Clone Position Divergence Norm vs. Number of Observations",
        scale=1e3,
        xlabel="Number of Observations",
        save_name=f"{saving_dir}/Stat_Position_boxplot_NOBS_All"
    )

    # ------------------------------------------------------
    # Velocity vs NOBS
    make_boxplot_multiple(
        "Clone_Divergence_Vel_Norm_peri_NOBS",
        ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
        title="Clone Velocity Divergence Norm vs. Number of Observations",
        scale=1.0,
        xlabel="Number of Observations",
        save_name=f"{saving_dir}/Stat_Velocity_boxplot_NOBS_All"
    )




compute_family(Family, NGA_False_Data, "False")
compute_family(Family, NGA_True_Data, "True")
compute_family(Family, NGA_Est_Data, "Est")

boxplot_all()
