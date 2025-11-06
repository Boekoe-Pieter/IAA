import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import numpy as np
import random
import matplotlib.patches as patches
from tudatpy.data.sbdb import SBDBquery

import pickle

from tudatpy import constants

AU = constants.ASTRONOMICAL_UNIT
day = constants.JULIAN_DAY



comet_name = "C2013US10"  #"C2001Q4","C2008A1","C2013US10"]

NGA_True = f"Coding/W79 Estimation/Sim_data/Data_NGA_True_{comet_name}.pkl"
NGA_Est = f"Coding/W79 Estimation/Sim_data/Data_NGA_Est_{comet_name}.pkl"
Arcwise = f"Coding/W79 Estimation/Sim_data/Data_arcwise_{comet_name}.pkl"
with open(NGA_True, "rb") as f:
    NGA_True_Data = pickle.load(f)
with open(NGA_Est, "rb") as f:
    NGA_Est_Data = pickle.load(f)
with open(Arcwise, "rb") as f:
    Arcwise_data = pickle.load(f)
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

general_statistics_obs = {
    "Clone_Divergence_Norm_peri_DPERI": {},  
    "Clone_Divergence_Vel_Norm_peri_DPERI": {},
    "Clone_Divergence_Norm_peri_NOBS": {},  
    "Clone_Divergence_Vel_Norm_peri_NOBS": {},
}

general_statistics_AU = {
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

# finding 60 days in AU
in_seconds = 60 * day
s_to_peri = Truth_times_NGA[-2] - in_seconds
Mask = Truth_times_NGA==s_to_peri
idx_AU = np.argmax(Mask)
AU_60d = Truth_AU[idx_AU]

def compute_family(family_dict, data,label):
    comet = f"{comet_name}_{label}"

    if comet not in general_statistics_obs["Clone_Divergence_Norm_peri_DPERI"]:
        general_statistics_obs["Clone_Divergence_Norm_peri_DPERI"][comet] = {}
    if comet not in general_statistics_obs["Clone_Divergence_Vel_Norm_peri_DPERI"]:
        general_statistics_obs["Clone_Divergence_Vel_Norm_peri_DPERI"][comet] = {}

    if comet not in general_statistics_obs["Clone_Divergence_Norm_peri_NOBS"]:
        general_statistics_obs["Clone_Divergence_Norm_peri_NOBS"][comet] = {}
    if comet not in general_statistics_obs["Clone_Divergence_Vel_Norm_peri_NOBS"]:
        general_statistics_obs["Clone_Divergence_Vel_Norm_peri_NOBS"][comet] = {}

    if comet not in general_statistics_AU["Clone_Divergence_Norm_peri_DPERI"]:
        general_statistics_AU["Clone_Divergence_Norm_peri_DPERI"][comet] = {}
    if comet not in general_statistics_AU["Clone_Divergence_Vel_Norm_peri_DPERI"]:
        general_statistics_AU["Clone_Divergence_Vel_Norm_peri_DPERI"][comet] = {}

    if comet not in general_statistics_AU["Clone_Divergence_Norm_peri_NOBS"]:
        general_statistics_AU["Clone_Divergence_Norm_peri_NOBS"][comet] = {}
    if comet not in general_statistics_AU["Clone_Divergence_Vel_Norm_peri_NOBS"]:
        general_statistics_AU["Clone_Divergence_Vel_Norm_peri_NOBS"][comet] = {}


    monte_sample = data['Montecarlo_trajectory']

    for sim, sim_data in monte_sample.items():
        for key, traj in sim_data.items():
            Monte_Carlo_times = data['Montecarlo_trajectory_times'][sim][key]
            Truth_trajectory_times = data['Truth_Reference_trajectory_times']
            Truth_trajectory = np.array(data['Truth_Reference_trajectory'])
            Truth_pos_norm = np.linalg.norm(Truth_trajectory[:, 0:3], axis=1)

            mask = np.isin(Truth_trajectory_times, Monte_Carlo_times)
            mask = np.squeeze(mask)

            Truth_trajectory_filtered = Truth_trajectory[mask]
            Truth_pos_norm_filtered = Truth_pos_norm[mask]

            family_dict["Truth_norm"] = Truth_pos_norm_filtered

            Monte_carlo_pos = traj[:, 0:3]
            Monte_carlo_vel = traj[:, 3:]

            # ----------------------------------------------------------------------
            # Compute divergence
            family_dict["Clone_norm"].setdefault(sim, {})[key] = np.linalg.norm(Monte_carlo_pos, axis=1)
            family_dict["Clone_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(Monte_carlo_vel, axis=1)

            family_dict["Clone_Divergence"].setdefault(sim, {})[key] = Monte_carlo_pos -  Truth_trajectory_filtered[:, 0:3]
            family_dict["Clone_Divergence_Norm"].setdefault(sim, {})[key] = np.linalg.norm(
                family_dict["Clone_Divergence"][sim][key], axis=1
            )

            family_dict["Clone_divergence_vel"].setdefault(sim, {})[key] = Monte_carlo_vel -  Truth_trajectory_filtered[:, 3:]
            family_dict["Clone_divergence_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(
                family_dict["Clone_divergence_vel"][sim][key], axis=1
            )

            # ----------------------------------------------------------------------
            # Find perihelion divergence
            idx_peri = np.argmin(Truth_pos_norm_filtered)

            family_dict["Clone_Divergence_Norm_peri"].setdefault(sim, {})[key] = \
                family_dict["Clone_Divergence_Norm"][sim][key][idx_peri]
            family_dict["Clone_Divergence_Nor_vel_peri"].setdefault(sim, {})[key] = \
                family_dict["Clone_divergence_vel_norm"][sim][key][idx_peri]
            
            # ----------------------------------------------------------------------
            # timeline
            perihelion = Truth_trajectory_times[-1]
            last_obs = data["observation_times"][sim][-1]

            N_obs = data["Sim_info"][sim].get("N_obs")
            dt_days_peri = (perihelion - last_obs) / constants.JULIAN_DAY

            # ----------------------------------------------------------------------
            # AU
            Mask = Truth_times_NGA == last_obs
            idx_last_obs = np.argmax(Mask)
            AU_last_obs = Truth_pos_norm[idx_last_obs]/AU

            # ----------------------------------------------------------------------
            # Saving to dictionary  

            # Time           
            general_statistics_obs["Clone_Divergence_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                family_dict["Clone_Divergence_Norm_peri"][sim][key]
            )

            general_statistics_obs["Clone_Divergence_Vel_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])



            general_statistics_obs["Clone_Divergence_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                family_dict["Clone_Divergence_Norm_peri"][sim][key]
            )

            general_statistics_obs["Clone_Divergence_Vel_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])

            # AU
            general_statistics_AU["Clone_Divergence_Norm_peri_DPERI"][comet].setdefault(AU_last_obs, []).append(
                family_dict["Clone_Divergence_Norm_peri"][sim][key]
            )

            general_statistics_AU["Clone_Divergence_Vel_Norm_peri_DPERI"][comet].setdefault(AU_last_obs, []).append(
                family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])



def boxplot_all(extra_time=15):
    labels = ["True","Est", "Arc"]
    plot_lables = ["NG Reference", "NG Estimated", "Arcwise"]
    comets = [f"{comet_name}_{label}" for label in labels]
    AU_width = 0.05
    def make_boxplot_multiple(stats,stat_dict_name, ylabel, title, scale=1.0,width=5, save_name=None, xlabel=r"$\Delta{Days}$ to Perihelion"):
        all_keys = sorted(set().union(*[stat_dict.keys() for stat_dict in 
                                        [stats[stat_dict_name][c] for c in comets if c in stats[stat_dict_name]]]),
                          reverse=True)

        fig, ax = plt.subplots(figsize=(15, 8))
        spacing = 0  

        colors = ["tab:blue", "tab:green", "tab:red"]

        for i, comet in enumerate(comets):
            positions = [x - spacing/2 + i*(spacing/len(labels)) for x in all_keys]
            time = np.arange(max(all_keys),min(all_keys)-extra_time,-extra_time)
            data = [np.array(stats[stat_dict_name][comet].get(x, [])) / scale for x in all_keys]
            box = plt.boxplot(data, positions=positions, widths=width, patch_artist=True, manage_ticks=False)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.6)
            plt.plot([], [], color=colors[i], label=plot_lables[i])

        if isinstance(all_keys[0], (int, float)) and xlabel == r"$\Delta{Days}$ to Perihelion":
            plt.gca().invert_xaxis()
            plt.xticks(time, rotation=70)
            # plt.axvline(x = DT_3AU, color = 'black', linestyle = '--', label = '3AU line')
            # plt.text(DT_3AU - 2, plt.ylim()[1]*0.6, '3AU line', rotation=90,
            #         va='center', ha='left', fontsize=9, color='black')
            
            # plt.axvline(x = 60, color = 'black', linestyle = '--',  label = '60 days line')
            # plt.text(60 - 2, plt.ylim()[1]*0.6, '60 days line', rotation=90,
            #         va='center', ha='left', fontsize=9, color='black')

            # plt.axvline(x = DT_1AU, color = 'black', linestyle = '--',  label = '1AU line')
            # plt.text(DT_1AU - 2, plt.ylim()[1]*0.6, '1AU line', rotation=90,
            #         va='center', ha='left', fontsize=9, color='black')

        if  xlabel == "Distance [AU]":
            plt.gca().invert_xaxis()
            # plt.axvline(x = 3.0, color = 'black', linestyle = '-',  label = '3.0 AU')
            # plt.axvline(x = 2.808, color = 'black', linestyle = '--',  label = '2.808 AU')
            # plt.axvline(x = AU_60d, color = 'black', linestyle = ':',  label = '60 Days')
 

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
        general_statistics_obs,
        "Clone_Divergence_Norm_peri_DPERI",
        ylabel="Clone Position Divergence Norm at perihelion [km]",
        title="Clone Position Divergence vs. Days to Perihelion",
        scale=1e3,
        save_name=f"{saving_dir}/{comet_name}_Stat_Position_boxplot_dtPeri_All"
    )
    
    # ------------------------------------------------------
    # Velocity vs Days
    make_boxplot_multiple(
        general_statistics_obs,
        "Clone_Divergence_Vel_Norm_peri_DPERI",
        ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
        title="Clone Velocity Divergence vs. Days to Perihelion",
        scale=1.0,
        save_name=f"{saving_dir}/{comet_name}_Stat_Velocity_boxplot_dtPeri_All"
    )
    
    # ------------------------------------------------------
    # Position vs NOBS
    make_boxplot_multiple(
        general_statistics_obs,
        "Clone_Divergence_Norm_peri_NOBS",
        ylabel="Clone Position Divergence Norm at perihelion [km]",
        title="Clone Position Divergence Norm vs. Number of Observations",
        scale=1e3,
        xlabel="Number of Observations",
        save_name=f"{saving_dir}/{comet_name}_Stat_Position_boxplot_NOBS_All"
    )

    # ------------------------------------------------------
    # Velocity vs NOBS
    make_boxplot_multiple(
        general_statistics_obs,
        "Clone_Divergence_Vel_Norm_peri_NOBS",
        ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
        title="Clone Velocity Divergence Norm vs. Number of Observations",
        scale=1.0,
        xlabel="Number of Observations",
        save_name=f"{saving_dir}/{comet_name}_Stat_Velocity_boxplot_NOBS_All"
    )

    # ------------------------------------------------------
    # Position vs AU
    make_boxplot_multiple(
        general_statistics_AU,
        "Clone_Divergence_Norm_peri_DPERI",
        ylabel="Clone Position Divergence Norm at perihelion [km]",
        title="Clone Position Divergence vs. AU from Perihelion",
        scale=1e3,
        width = AU_width,
        xlabel=r"Distance [AU]",
        save_name=f"{saving_dir}/{comet_name}_Stat_Position_boxplot_dAU_All"
    )
    
    # ------------------------------------------------------
    # Velocity vs AU
    make_boxplot_multiple(
        general_statistics_AU,
        "Clone_Divergence_Vel_Norm_peri_DPERI",
        ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
        title="Clone Velocity Divergence vs. AU from Perihelion",
        scale=1.0,
        width = AU_width,
        xlabel=r"Distance [AU]",
        save_name=f"{saving_dir}/{comet_name}_Stat_Velocity_boxplot_dAU_All"
    )


# compute_family(Family, NGA_False_Data, "False")
compute_family(Family, NGA_True_Data, "True")
compute_family(Family, NGA_Est_Data, "Est")
compute_family(Family, Arcwise_data, "Arc")

boxplot_all()

# Make a table for the NGA estimations
import numpy as np
from tudatpy.kernel import constants

AU = constants.ASTRONOMICAL_UNIT

def Make_table(data,comet_name):
    target_sbdb = SBDBquery(comet_name,full_precision=True)
    A1 = target_sbdb["orbit"]["model_pars"].get("A1")
    A2 = target_sbdb["orbit"]["model_pars"].get("A2")
    A3 = target_sbdb["orbit"]["model_pars"].get("A3")
    N_obs = target_sbdb["orbit"].get("n_obs_used")

    JPL_NGA= np.array(
        [A1.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A1 is not None else 0,
        A2.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A2 is not None else 0,
        A3.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A3 is not None else 0,
        ]
        )
    
    NGA_data = data["NGA_Est"]
    Truth_trajectory = data['Truth_Reference_trajectory']
    Truth_times_NGA = data["Truth_Reference_trajectory_times"]
    Truth_pos_norm = np.linalg.norm(Truth_trajectory[:, 0:3], axis=1)
    
    #NGA initial guess
    A1 = 2.195*10**-8*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2
    A2 = 0.006*10**-8*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2
    A3 = 0*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2

    print("\n============================")
    print(f"TUDAT NGA Estimations {comet_name}")
    print("============================\n")
    print(f"{'Sim':<8}{'A1':>6}{'A2':>16}{'A3':>16}{'N_obs':>18}{'ΔT peri [days]':>18}{'AU last obs':>15}")
    print("-" * 100)
    print(f"{'JPL':<8}{JPL_NGA[0]:>12.6e}{JPL_NGA[1]:>16.6e}{JPL_NGA[2]:>16.6e}"
                f"{N_obs:>10}{'N/A':>15}{'N/A':>17}")
    print("-" * 100)
    print(f"{'Guess':<8}{A1:>12.6e}{A2:>16.6e}{A3:>16.6e}"
          f"{'N/A':>10}{'N/A':>15}{'N/A':>17}")
    print("-" * 100)
    for sim, Est_NGA in NGA_data.items():
        NGA_values = Est_NGA

        # timeline
        perihelion = Truth_times_NGA[-1]
        last_obs = data["observation_times"][sim][-1]

        N_obs = data["Sim_info"][sim].get("N_obs", "-")
        dt_days_peri = (perihelion - last_obs) / constants.JULIAN_DAY

        # AU
        idx_last_obs = np.argmin(np.abs(Truth_times_NGA - last_obs))
        AU_last_obs = Truth_pos_norm[idx_last_obs] / AU
        print(f"{sim:<8}{NGA_values[0]:>12.6e}{NGA_values[1]:>16.6e}{NGA_values[2]:>16.6e}"
                    f"{N_obs:>10}{dt_days_peri[0]:>15.2f}{AU_last_obs:>18.6f}")

    print("-" * 100)
Make_table(NGA_Est_Data,comet_name)
