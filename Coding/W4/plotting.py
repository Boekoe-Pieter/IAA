import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from scipy.stats import norm
import matplotlib.cm as cm
import scipy.constants as const
from mpl_toolkits.mplot3d import Axes3D 
import os

# -------------------------------
# Loading data
target_body = 'C/2001 Q4'
save_body = re.sub(r'[^\w\-_\. ]', '_', target_body)

rel_dir = os.path.dirname(os.path.abspath(__file__))
directory_name = f"{rel_dir}/data_{save_body}"


with open(f"{directory_name}/info_trace_2000_1.txt", "rb") as f:
    info = pickle.load(f)
with open(f"{directory_name}/data_trace_2000_1.pkl", "rb") as f:
    data = pickle.load(f)

# -------------------------------
# General info
info_dict = info["Info"]
N_clones = info["Info"]["N_clones"]
Time_array = data["Sim_time"]["Time"]

img_save_dir = f"{directory_name}/{info["Info"]["N_clones"]}_{info["Info"]["dt"]}_{info["Info"]["Model"]}"
try:
    os.mkdir(img_save_dir)
    print(f"Directory '{img_save_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{img_save_dir}' already exists.")

# -------------------------------
# Retrieving dictionaries
Traj_NGA = data["Traj_NGA"]         # True NGA
Traj_NGAf = data["Traj_NGAf"]       # False NGA

# -------------------------------
# Gathering data
sample_dis = 0.2 # AU

NGA_family = {"Nominal_norm": {},
              "Clone_norm": {},
              "Clone_Divergence": {},
              "Clone_Divergence_Norm": {},
              "Clone_Divergence_Norm_1AU": {},          
              "Clone_Divergence_Sample": {},
               "fit": {
                   "mu": [],
                   "sigma":[]
               },
}
NGAf_family = {"Nominal_norm": {},
               "Clone_norm": {},
               "Clone_Divergence": {},
               "Clone_Divergence_Norm": {},
               "Clone_Divergence_Norm_1AU": {},          
               "Clone_Divergence_Sample": {},
               "fit": {
                   "mu": [],
                   "sigma":[]
               },
}

def compute_family(Trajectory_data,dict):
    for key in Trajectory_data.keys():
        if not key.isdigit():
            continue
        if key == "0":
            dict["Nominal_norm"][key] = np.linalg.norm(Trajectory_data[key][:, 0:3],axis=1)
            continue    
        dict["Clone_Divergence"][key] = (Trajectory_data[key][:, 0:3] - Trajectory_data["0"][:, 0:3]) 
        dict["Clone_Divergence_Norm"][key] = np.linalg.norm(dict["Clone_Divergence"][key][:, 0:3],axis=1)
        dict["Clone_norm"][key] = np.linalg.norm(Trajectory_data[key][:, 0:3],axis=1) 

        idx_1AU = np.argmin(np.abs(dict["Clone_norm"][key] - 1.0))
        dict["Clone_Divergence_Norm_1AU"][key] = dict["Clone_Divergence_Norm"][key][idx_1AU]

def divergence_sample(sample_dis,dict):
    N = (dict["Nominal_norm"]["0"][0] - dict["Nominal_norm"]["0"][-1]) / sample_dis
    N_round = int(np.round(N))
    sample_array = np.array(dict["Nominal_norm"]["0"][::N_round])
    sample_indx = np.array([np.where(dict["Nominal_norm"]["0"]==i) for i in sample_array])
    for idx in sample_indx:
        dict["Clone_Divergence_Sample"][f"{str(idx[0][0])}"] = {f"{str(idx)}": [dict["Clone_Divergence_Norm"][key][idx] for key in dict["Clone_Divergence_Norm"].keys()]}

def compute_statistics(dict):
    mu_ls = []
    sig_ls = []
    for key in dict["Clone_Divergence_Sample"].keys():
        values = np.array(list(dict["Clone_Divergence_Sample"][key].values()))
        mu, sigma = norm.fit(values)
        mu_ls.append(mu)
        sig_ls.append(sigma)
    dict["fit"]["mu"] = mu_ls
    dict["fit"]["sigma"]= sig_ls

compute_family(Traj_NGA,NGA_family)
compute_family(Traj_NGAf,NGAf_family)

divergence_sample(sample_dis,NGA_family)
divergence_sample(sample_dis,NGAf_family)

compute_statistics(NGA_family)
compute_statistics(NGAf_family)

# -------------------------------
# Simulation plots
def Correlation(cov_matrix,simulation_info):
    body = simulation_info['Body']
    correlations = np.corrcoef(cov_matrix)
    estimated_param_names = [ "e",  "q",  "tp",  "RAAN",    "AoP",  "i",    "A1",      "A2",     "A3"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
    ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

    for i in range(len(estimated_param_names)):
        for j in range(len(estimated_param_names)):
            text = ax.text(
                j, i, round(correlations[i, j], 2), ha="center", va="center", color="w"
            )

    cb = plt.colorbar(im)

    ax.set_xlabel("Estimated Parameter")
    ax.set_ylabel("Estimated Parameter")

    fig.suptitle(f"Correlations for estimated parameters for {body}")

    fig.set_tight_layout(True)

    plt.savefig(f'{img_save_dir}/Corr_matrix.pdf',dpi=300)
    plt.show()

def D_traject(cartesian, ax=None, label=None, color=None, alpha=1.0):
    """Plot a single trajectory in 3D"""
    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal', adjustable='box')

    ax.plot(cartesian[:, 0], cartesian[:, 1], cartesian[:, 2], 
            label=label, color=color, alpha=alpha)
    ax.scatter(cartesian[0, 0], cartesian[0, 1], cartesian[0, 2], marker="o", s=10, color=color)
    ax.scatter(cartesian[-1, 0], cartesian[-1, 1], cartesian[-1, 2], marker="x", s=10, color=color)

    return ax

def plot_ensemble(traj,planet_traj,info_dict, n_clones=20, random=True):
    mean=traj["0"]
    body = info_dict["Body"]

    def draw_sun(ax):
        radius_sun = 696340 * 1000 / const.au 
        _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
        _x = radius_sun * np.cos(_u) * np.sin(_v)
        _y = radius_sun * np.sin(_u) * np.sin(_v)
        _z = radius_sun * np.cos(_v)
        ax.plot_wireframe(_x, _y, _z, color="orange", alpha=0.5, lw=0.5, zorder=0)


    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect("equal", adjustable="box")

    D_traject(np.array(mean), ax=ax, label=body, color="red", alpha=0.5)

    for planet, traj in planet_traj.items():
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=planet, lw=0.5)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker="o", s=1)

    draw_sun(ax)

    max_value = max(np.max(np.abs(np.array(traj))) for traj in traj.values())
    ax.set_xlim([-max_value, max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")

    ax.legend()
    plt.savefig(f'{img_save_dir}/3D_images.pdf', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect("equal", adjustable="box")

    D_traject(np.array(mean), ax=ax, label=body, color="red", alpha=0.5)

    for planet, traj in planet_traj.items():
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=planet, lw=0.5)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker="o", s=1)

    draw_sun(ax)

    max_value = 1.5
    ax.set_xlim([-max_value, max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")

    ax.legend()
    plt.savefig(f'{img_save_dir}/3D_image_zoomed.pdf', dpi=300)
    plt.show()

def plot_sampled(data, info_dict):
    Sampled_data, mean_data = data["sampled_data"], data["mean_data"]
    body = info_dict['Body']

    mean_vals = mean_data

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()

    labels = ["T [MJD]", "ecc", "q",  
              "RAAN [rad]", "AoP [rad]", "inc [rad]"]

    for i in range(6):
        axs[i].hist(Sampled_data[:, i], bins=30, alpha=0.6, label="Samples")
        axs[i].axvline(mean_vals[i], color="red", linestyle="--", label="Mean" if i == 0 else "")
        axs[i].set_xlabel(labels[i])
        axs[i].set_ylabel("Count")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f'{img_save_dir}/Sampled_initial.pdf', dpi=300)
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    A_labels = ["A1", "A2", "A3"]

    for i in range(3):
        axs[i].hist(Sampled_data[:, 6+i], bins=30, alpha=0.6, label="Samples")
        axs[i].axvline(mean_vals[6+i], color="red", linestyle="--", label="Mean" if i == 0 else "")
        axs[i].set_xlabel(A_labels[i])
        axs[i].set_ylabel("Count")

    plt.suptitle(f"Distribution of NGA coefficients for {body}")
    plt.tight_layout()
    plt.savefig(f'{img_save_dir}/Sampled_NGA.pdf', dpi=300)
    plt.show()

Correlation(data["covariance"],info_dict,)
plot_ensemble(data["Traj_NGA"],data["Planet_orb"], info_dict, n_clones=0, random=True)
plot_sampled(data,info_dict)

# -------------------------------
# Statistical plots
def plot_divergence_timeline(NGA, NGAf, sample_dis,info_dict):
    body,arc,model,Int,dt = info_dict["Body"],info_dict["Arc"],info_dict["Model"], info_dict["Int"],info_dict["dt"]
    
    N = (NGA["Nominal_norm"]["0"][0] - NGA["Nominal_norm"]["0"][-1]) / sample_dis
    N_round = int(np.round(N))
    sample_array_NGA = np.array(NGA["Nominal_norm"]["0"][::N_round])

    Nf = (NGAf["Nominal_norm"]["0"][0] - NGAf["Nominal_norm"]["0"][-1]) / sample_dis
    Nf_round = int(np.round(Nf))
    sample_array_NGAf = np.array(NGAf["Nominal_norm"]["0"][::Nf_round])

    NGA_mu, NGA_sig = np.array(NGA["fit"]["mu"]), np.array(NGA["fit"]["sigma"])
    NGAf_mu, NGAf_sig = np.array(NGAf["fit"]["mu"]), np.array(NGAf["fit"]["sigma"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    fig.suptitle(f"{body}, {arc}-data arc, model: {model}, Integrator: {Int}, dt: {dt}")
    ax1.set_title("NGA")
    ax1.set_xlabel("Heliocentric distance")
    ax1.set_ylabel("Distance [km]")
    ax1.plot(sample_array_NGA, NGA_mu*const.au/1000, color="red", label=r"$\mu$ [km]")
    ax1.plot(sample_array_NGA, NGA_sig*const.au/1000, color="blue", label=r"$\sigma$ [km]")
    ax1.axvline(x=1, linestyle="dotted", color="black", label="1 AU")
    ax1.grid(True, which="both")
    ax1.legend()

    ax2.set_title("NGA false")
    ax2.set_xlabel("Heliocentric distance")
    ax2.plot(sample_array_NGAf, NGAf_mu*const.au/1000, color="red", label=r"$\mu$ [km]")
    ax2.plot(sample_array_NGAf, NGAf_sig*const.au/1000, color="blue", label=r"$\sigma$ [km]")
    ax2.axvline(x=1, linestyle="dotted", color="black", label="1 AU")
    ax2.grid(True, which="both")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{img_save_dir}/stats_timeline.pdf",dpi=300)
    plt.show()

def plot_divergence_timeline_difference(NGA, NGAf, sample_dis,info_dict):
    body,arc,model,Int,dt = info_dict["Body"],info_dict["Arc"],info_dict["Model"], info_dict["Int"],info_dict["dt"]
    
    N = (NGA["Nominal_norm"]["0"][0] - NGA["Nominal_norm"]["0"][-1]) / sample_dis
    N_round = int(np.round(N))
    sample_array_NGA = np.array(NGA["Nominal_norm"]["0"][::N_round])

    Nf = (NGAf["Nominal_norm"]["0"][0] - NGAf["Nominal_norm"]["0"][-1]) / sample_dis
    Nf_round = int(np.round(Nf))
    sample_array_NGAf = np.array(NGAf["Nominal_norm"]["0"][::Nf_round])

    NGA_mu, NGA_sig = np.array(NGA["fit"]["mu"]), np.array(NGA["fit"]["sigma"])
    NGAf_mu, NGAf_sig = np.array(NGAf["fit"]["mu"]), np.array(NGAf["fit"]["sigma"])

    difference = NGA_mu-NGAf_mu
    plt.subplot()
    plt.plot(sample_array_NGA,difference*const.au)
    plt.xlabel("Heliocentric Distance")
    plt.ylabel(r"$\Delta{\mu}$ [m]")
    plt.savefig(f"{img_save_dir}/stats_timeline_difference.pdf",dpi=300)
    plt.show()

def plot_histogram(hist_data, hist_data_f, simulation_info):
    body = simulation_info['Body']
    values = np.array(list(hist_data.values())) * const.au / 1000
    values_f = np.array(list(hist_data_f.values())) * const.au / 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    mu, sigma = norm.fit(values)
    count, bins, _ = axes[0].hist(values, bins=50, density=True,
                                  color="steelblue", alpha=0.6, edgecolor="k", label="Histogram")

    x = np.linspace(min(values), max(values), 2000)
    pdf = norm.pdf(x, mu, sigma)
    axes[0].plot(x, pdf, 'r-', lw=2, label=f"Normal fit\nμ={mu:.4e}, σ={sigma:.3e}")

    axes[0].set_xlabel(rf"Deviation at 1 AU [km]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title(f"Distribution with NGA")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    mu_f, sigma_f = norm.fit(values_f)
    count_f, bins_f, _ = axes[1].hist(values_f, bins=50, density=True,
                                      color="darkorange", alpha=0.6, edgecolor="k", label="Histogram")

    x_f = np.linspace(min(values_f), max(values_f), 2000)
    pdf_f = norm.pdf(x_f, mu_f, sigma_f)
    axes[1].plot(x_f, pdf_f, 'r-', lw=2, label=f"Normal fit\nμ={mu_f:.4e}, σ={sigma_f:.3e}")

    axes[1].set_xlabel(rf"Deviation at 1 AU [km]")
    axes[1].set_title(f"Distribution w/o NGA")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # --- Common supertitle ---
    fig.suptitle(
        f"Position distribution of deviations at 1 AU for comet {body}\n"
        f"Integrator: {simulation_info['Int']}, dt: {simulation_info['dt']}, "
        f"samples: {len(hist_data.keys())}, CPU time: {simulation_info['Int_time']}"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{img_save_dir}/Histogram_1AU.pdf", dpi=300)
    plt.show()


plot_divergence_timeline(NGA_family, NGAf_family,sample_dis,info_dict)
plot_divergence_timeline_difference(NGA_family, NGAf_family,sample_dis,info_dict)
plot_histogram(NGA_family["Clone_Divergence_Norm_1AU"],NGAf_family["Clone_Divergence_Norm_1AU"],info_dict)

    
# def plot_difference_trajectory(trajectories,diffs_all,time):
#     body = simulation_info['body']
#     safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
#     plt.figure(figsize=(10, 6))
    
#     for idx in range(1, len(trajectories)):
#         plt.plot(time, diffs_all[str(idx)], alpha=0.1, color="blue")

#     plt.xlabel("Time [days]")
#     plt.ylabel(r"$||\Delta{r}|| [km]$")
#     plt.title(f"Divergence of clones from mean orbit {body}")
#     plt.savefig(f'Coding/W3/covariance_analysis/Difference_{len(Sampled_data)}_samples_{safe_name}_{simulation_info['int']}')
#     plt.show()

# def plot_trajectory(trajectories,time):
#     plt.figure(figsize=(10, 6))
    
#     for idx in range(1, len(trajectories)):
#         plt.plot(time, trajectories[str(idx)], alpha=0.1, color="blue")

#     plt.xlabel("Time [days]")
#     plt.ylabel("||r|| [AU]")
#     plt.title("Divergence of clones from mean orbit")
#     plt.show()

# # -------------------------------
# # plotting data
# # plot_sampled(Sampled_data,mean_data)
# plot_difference_trajectory(Trajectory_norms,Trajectory_difference,simulation_time["time"])
# plot_trajectory(Trajectory_norms,simulation_time["time"])
# plot_ensemble(data, n_clones=50, random=True)
# Osculating(osc_diffs_at_1AU,simulation_info)
# plot_gr(gr["0"],simulation_time,simulation_info)
