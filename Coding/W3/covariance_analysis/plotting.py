import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from scipy.stats import norm
import matplotlib.cm as cm
import scipy.constants as const
from mpl_toolkits.mplot3d import Axes3D 

# -------------------------------
# Loading data
with open("Coding/W3/data_C_2001 Q4_whfast_1000_1.pkl", "rb") as f:
    data = pickle.load(f)

planet_names = ["Mercury", "Venus", "Earth", "Mars", 
                "Jupiter", "Saturn", "Uranus", "Neptune"]
# -------------------------------
# Retrieving dictionaries
simulation_time = data["Sim_time"]
simulation_info = data["info"]
Trajectory_data = data["trajectories"]
Osculating_data = data["Osculating"]
Covariance_matrix = data["covariance"]
Perturbations_used = data["Perturbations"]
mean_data = data["mean_data"]
Sampled_data = data["sampled_data"]
gr = data["gr"]
# -------------------------------
# Gathering data
Trajectory_norms = {}
Trajectory_difference = {}
diffs_at_1AU = {}

mean_traj = Trajectory_data["0"][:, 0:3]
mean_osc = np.array(data["Osculating"]["0"])
Trajectory_norms["0"] = np.linalg.norm(mean_traj, axis=1)

for key in Trajectory_data.keys():
    if not key.isdigit():
        continue
    if key == "0":
        continue

    orbit = Trajectory_data[key][:, 0:3]
    Trajectory_norms[key] = np.linalg.norm(orbit, axis=1)

    Trajectory_difference[key] = np.linalg.norm(orbit - mean_traj, axis=1) * const.au / 1000

    idx_1AU = np.argmin(np.abs(Trajectory_norms[key] - 1.0))
    diffs_at_1AU[key] = Trajectory_difference[key][idx_1AU]

Planet_trajectories = {
    name: np.array(data["trajectories"][name]) 
    for name in planet_names if name in data["trajectories"]
}

osc_diffs_at_1AU = { "a": [], "e": [], "i": [], "RAAN": [], "omega": [], "theta": [] }

for key, orbit in data["Osculating"].items():
    if key == "0" or not key.isdigit():
        continue
    
    orbit = np.array(orbit) 
    
    r_norms_clone = np.linalg.norm(data["trajectories"][key], axis=1)
    r_norms_ref   = np.linalg.norm(data["trajectories"]["0"], axis=1)

    idx_1AU = np.argmin(np.abs(r_norms_ref - 1.0))

    diffs = orbit[idx_1AU] - mean_osc[idx_1AU]
    osc_diffs_at_1AU["a"].append(diffs[0])
    osc_diffs_at_1AU["e"].append(diffs[1])
    osc_diffs_at_1AU["i"].append(diffs[2])
    osc_diffs_at_1AU["RAAN"].append(diffs[3])
    osc_diffs_at_1AU["omega"].append(diffs[4])
    osc_diffs_at_1AU["theta"].append(diffs[5])


# -------------------------------
# Plot defenitions
def plot_histogram(hist_data,simulation_info,Perturbations_used):
    body =simulation_info['body']
    values = np.array(list(hist_data.values()))

    plt.figure(figsize=(8,6))
    mu, sigma = norm.fit(values)

    count, bins, _ = plt.hist(values, bins=50, density=True,
                            color="steelblue", alpha=0.6, edgecolor="k", label="Histogram")

    x = np.linspace(min(values), max(values), 2000)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', lw=2, label=f"Normal fit\nμ={mu:.4e}, σ={sigma:.3e}")

    plt.xlabel(rf"Deviation at 1 AU [km]")
    plt.ylabel("Probability density")
    plt.title(f"Position distribution of deviations at 1 AU for comet {body}")
    plt.suptitle(f"Integrator: {simulation_info['int']}, dt: {simulation_info['dt']}, samples: {len(hist_data.keys())}, CPU time: {simulation_info['int_time']} ")
    plt.legend()
    plt.grid(alpha=0.3)
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
    plt.savefig(f'Coding/W3/covariance_analysis/Histogram_{len(hist_data.keys())}_samples_{safe_name}_{simulation_info['int']}')
    plt.show()

def Osculating(osc_diffs_at_1AU,simulation_info):
    body =simulation_info['body']
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
    labels = ["a", "e", "i", "RAAN", "omega", "theta"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, label in zip(axes.flatten(), labels):
        values = np.array(osc_diffs_at_1AU[label])
        ax.hist(values, bins=20, density=True, alpha=0.7)
        ax.axvline(np.mean(values), color="r", linestyle="--", label=f"Mean: {np.mean(values):.3e}")
        ax.set_title(f"Δ{label} at 1 AU")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'Coding/W3/covariance_analysis/Histogram_OSC_{len(osc_diffs_at_1AU.keys())}_samples_{safe_name}_{simulation_info['int']}')
    plt.show()

def Correlation(cov_matrix,simulation_info,Perturbations_used):
    body = simulation_info['body']
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
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
    plt.savefig(f'Coding/W3/covariance_analysis/covar_matrix_{safe_name}')
    plt.show()

def plot_sampled(Sampled_data,mean_vals):
    body = simulation_info['body']
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)

    mean_vals = mean_data

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()

    labels = ["ecc", " q", "T [MJD]", 
            "RAAN [rad]", "AoP [rad]", "inc [rad]"]

    for i in range(6):
        axs[i].scatter(Sampled_data[:, i], range(len(Sampled_data)), s=10, alpha=0.5, label="Samples")
        axs[i].axvline(mean_vals[i], color="red", linestyle="--", label="Mean" if i == 0 else "")
        axs[i].set_xlabel(labels[i])
        axs[i].set_ylabel("Sample index")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f'Coding/W3/covariance_analysis/Sampled_data{len(Sampled_data)}_samples_{safe_name}_{simulation_info['int']}')
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    A_labels = ["A1", "A2", "A3"]

    for i in range(3):
        axs[i].scatter(Sampled_data[:, 6+i], range(len(Sampled_data)), s=10, alpha=0.5, label="Samples")
        axs[i].axvline(mean_vals[6+i], color="red", linestyle="--", label="Mean" if i == 0 else "")
        axs[i].set_xlabel(A_labels[i])
        axs[i].set_ylabel("Sample index")


    plt.suptitle(f"Distribution of NGA coefficients for {body}")
    plt.tight_layout()
    plt.savefig(f'Coding/W3/covariance_analysis/NGA_data{len(Sampled_data)}_samples_{safe_name}_{simulation_info['int']}')
    plt.show()
    
def plot_difference_trajectory(trajectories,diffs_all,time):
    body = simulation_info['body']
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
    plt.figure(figsize=(10, 6))
    
    for idx in range(1, len(trajectories)):
        plt.plot(time, diffs_all[str(idx)], alpha=0.1, color="blue")

    plt.xlabel("Time [days]")
    plt.ylabel(r"$||\Delta{r}|| [km]$")
    plt.title(f"Divergence of clones from mean orbit {body}")
    plt.savefig(f'Coding/W3/covariance_analysis/Difference_{len(Sampled_data)}_samples_{safe_name}_{simulation_info['int']}')
    plt.show()

def plot_trajectory(trajectories,time):
    plt.figure(figsize=(10, 6))
    
    for idx in range(1, len(trajectories)):
        plt.plot(time, trajectories[str(idx)], alpha=0.1, color="blue")

    plt.xlabel("Time [days]")
    plt.ylabel("||r|| [AU]")
    plt.title("Divergence of clones from mean orbit")
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

def plot_ensemble(data, n_clones=20, random=True):
    trajectories = data["trajectories"]
    body = data["info"]["body"]
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)

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

    D_traject(np.array(trajectories["0"]), ax=ax, label=body, color="red", alpha=0.5)

    for planet, traj in Planet_trajectories.items():
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=planet, lw=0.5)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker="o", s=1)

    draw_sun(ax)

    max_value = max(np.max(np.abs(np.array(traj))) for traj in trajectories.values())
    ax.set_xlim([-max_value, max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")

    ax.legend()
    plt.savefig(f'Coding/W3/covariance_analysis/3D_{len(trajectories.keys())-9}_samples_{safe_name}_{simulation_info["int"]}.png', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect("equal", adjustable="box")

    D_traject(np.array(trajectories["0"]), ax=ax, label=body, color="red", alpha=0.5)

    for planet, traj in Planet_trajectories.items():
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=planet, lw=0.5)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker="o", s=1)

    draw_sun(ax)

    max_value = np.min(np.abs(np.linalg.norm(np.array(trajectories["0"])[:, :3],axis =1))) 
    ax.set_xlim([-max_value, max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")

    ax.legend()
    plt.savefig(f'Coding/W3/covariance_analysis/3D_{len(trajectories.keys())-9}_samples_{safe_name}_{simulation_info["int"]}_zoomed.png', dpi=300)
    plt.show()

def plot_gr(g_r,simulation_time,simulation_info):
    body =simulation_info['body']
    safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
    end = simulation_time['end']
    times = simulation_time['time']
    fig, axes = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle(f"g(r) function of {body}")

    time_scaled = times-end
    axes.plot(time_scaled, g_r,
                )

    axes.set_ylabel('g(r)')
    axes.set_xlabel("MJD from perihelion")
    axes.grid(True, which='both')
    axes.legend()
    axes.set_yscale('log')

    # margin = 20

    # margin = 5
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# -------------------------------
# plotting data
plot_histogram(diffs_at_1AU,simulation_info,Perturbations_used)
# Correlation(Covariance_matrix,simulation_info,Perturbations_used)
# plot_sampled(Sampled_data,mean_data)
plot_difference_trajectory(Trajectory_norms,Trajectory_difference,simulation_time["time"])
plot_trajectory(Trajectory_norms,simulation_time["time"])
plot_ensemble(data, n_clones=50, random=True)
Osculating(osc_diffs_at_1AU,simulation_info)
plot_gr(gr["0"],simulation_time,simulation_info)
