# loading packages
import numpy as np

# Import systems
import sys
import time as timer
import pickle
import re
import os

# load sim packages
import rebound
import reboundx 
from scipy import constants as const

# Import python files
sys.path.append('/Users/pieter/IAA/Coding')
import Utilities as Util       
from NGAs import NGA_data   
from covars import covar_matrices

np.set_printoptions(linewidth=160)

# ---------------------------------
# info

# data
data = 'Coding/Full_data_set.txt'

# target body
body = 'C/2001 Q4'
arc = "Full"
model = "n5"
save_body = re.sub(r'[^\w\-_\. ]', '_', body)

# saving directory
rel_dir = os.path.dirname(os.path.abspath(__file__))
directory_name = f"{rel_dir}/data_{save_body}"

# environment
primary = "sun"
planets = ["Mercury","Venus","Earth","Moon","mars","Jupiter","Saturn","Uranus","Neptune"]

# sim details
N_time = 10000
dt = 1 
integrator = "trace"
N_samples = 2000

# perturbations
NGA = True
NBP = False
Rel = True

# saving dictionary
sim_info = { "Info": {
             "Body": body,
             "Arc": arc,
             "Model": model,
             "Int": integrator,
             "dt": 0,
             "Int_time": 0,
             "N_clones": 0,
             "perturbing_bodies": 0,
             "Perturbations": {
                    'NBP': NBP,
                    'NGA': NGA,
                    'Rel': Rel
                    }
    }
}

data_to_write = {
    "Sim_time": {
        'Start': 0,
        'End': 0,
        'Time': 0,
        'N_lin': N_time
        }, 
    "Planet_orb": {},
    "Traj_NGA": {},
    "Traj_NGAf": {},
    "mean_data": 0,
    "sampled_data": 0,
    "covariance": 0,
}

# ---------------------------------
# load comet data
comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)

current_comet_data = data_dict[body][arc][model]  

arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)

cov_JPL_Q4 = np.array([[9.94e-13,4.3e-14,-4.15e-12,-3.89e-12,-6.51e-12,1.15e-12,-1.54e-16,-1.61e-17,9.5e-18],
                    [4.3e-14,9.08e-15,-1.44e-13,-1.85e-13,-1.99e-13,6.92e-14,-7.84e-18,-3.2e-19,4.51e-19],
                    [-4.15e-12,-1.44e-13,8.86e-11,1.82e-11,9.45e-11,-1.3e-11,6.91e-16,2.73e-16,-5.06e-17],
                    [-3.89e-12,-1.85e-13,1.82e-11,1.18e-10,4.46e-11,-5.77e-12,5.9e-16,1.6e-17,-3.05e-16],
                    [-6.51e-12,-1.99e-13,9.45e-11,4.46e-11,1.34e-10,-1.14e-11,1.07e-15,3.31e-16,-1.26e-16],
                    [1.15e-12,6.92e-14,-1.3e-11,-5.77e-12,-1.14e-11,7.69e-11,-1.72e-16,-1.91e-17,-1.35e-16],
                    [-1.54e-16,-7.84e-18,6.91e-16,5.9e-16,1.07e-15,-1.72e-16,2.58e-20,3.5e-21,-1.5e-21],
                    [-1.61e-17,-3.2e-19,2.73e-16,1.6e-17,3.31e-16,-1.91e-17,3.5e-21,5.93e-21,-2.4e-22],
                    [9.5e-18,4.51e-19,-5.06e-17,-3.05e-16,-1.26e-16,-1.35e-16,-1.5e-21,-2.4e-22,4.12e-21]])

covariance = cov_JPL_Q4

# ---------------------------------
# Initial conditions
mean_conditions = np.array([T_perihelium_mjd,q, ecc, np.deg2rad(RAAN), np.deg2rad(aop), np.deg2rad(i), 1/a_recip])
mean_to_sample = np.array([ecc,q, T_perihelium_mjd, np.deg2rad(RAAN), np.deg2rad(aop), np.deg2rad(i)
                           ,current_comet_data['A1'], current_comet_data['A2'], current_comet_data['A3']])

samples = np.random.multivariate_normal(mean_to_sample, covariance, size=N_samples)

# ---------------------------------
# Simulator
start_time = arc1_mjd
end_time = T_perihelium_mjd 

sim = Util.create_sim(primary,start_time,integrator,dt)
times = np.linspace(start_time, end_time, N_time)

# ---------------------------------
# adding perturbations
if NBP:
    Util.add_NBP(sim,start_time,planets)
    data_to_write["Planet_orb"] = {planet: np.zeros((len(times), 3)) for planet in planets}
if Rel:
    Util.add_Rel(sim)

# ---------------------------------
# add mean orbit
def add_mean(ref_sim):
    ref_sim.add(
        primary=ref_sim.particles[0],
        m=0.0,
        a=(q) / (1 - ecc),
        e=ecc,
        inc=np.deg2rad(i),
        Omega=np.deg2rad(RAAN),
        omega=np.deg2rad(aop),
        T=T_perihelium_mjd,
        hash='0'
    )

add_mean(sim)
data_to_write["Traj_NGA"]["0"] = np.zeros((len(times), 3))
data_to_write["Traj_NGAf"]["0"] = np.zeros((len(times), 3))

# ---------------------------------
# add pertubed conditions
def add_clones(ref_sim):
    for idx, s in enumerate(samples):
        if len(s) >= 9:
            ecc_s, q_s, q_date, RAAN_s, aop_s, i_s, A1_s, A2_s, A3_s = s
        else:
            ecc_s, q_s, q_date, RAAN_s, aop_s, i_s = s

        hash_id = f"{idx+1}"  
        ref_sim.add(
            primary=ref_sim.particles[0],
            m=0.0,
            a=q_s / (1 - ecc_s),
            e=ecc_s,
            inc=i_s,
            Omega=RAAN_s,
            omega=aop_s,
            T=q_date,
            hash=hash_id
        )

for idx in range(1, N_samples + 1):
    data_to_write["Traj_NGA"][f"{idx}"] = np.zeros((len(times), 3))
    data_to_write["Traj_NGAf"][f"{idx}"] = np.zeros((len(times), 3))

add_clones(sim)

print(
    f"Integrating \n"
    f"integrator: {integrator}\n"
    f"dt: {dt} day\n"
    f"Samples: {N_samples}\n"
    )



start = timer.perf_counter()
for N in range(2):
    print("integrating with NGA" if N==1 else "integrating w/o NGA")
    for j, time in enumerate(times):
        sim.integrate(time)
        p_mean = sim.particles["0"]
        if N==1:
            def add_NGAs(reb_sim):
                p_mean = sim.particles["0"]
                r_vec = np.array([p_mean.x, p_mean.y, p_mean.z])
                v_vec = np.array([p_mean.vx, p_mean.vy, p_mean.vz])

                r_norm = np.linalg.norm(r_vec)

                A_vec = np.array([
                    current_comet_data['A1'],
                    current_comet_data['A2'],
                    current_comet_data['A3']
                ]) * 1e-8

                m = current_comet_data['m']
                n = current_comet_data['n']
                k = current_comet_data['k']
                r0 = current_comet_data['r0']
                alpha = current_comet_data['alph']

                g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)
                F_vec_rtn = g * A_vec
                C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)
                F_vec_inertial = C_rtn2eci @ F_vec_rtn

                p_mean.ax += F_vec_inertial[0]
                p_mean.ay += F_vec_inertial[1]
                p_mean.az += F_vec_inertial[2]

                for idx, s in enumerate(samples):
                    if len(s) >= 9:
                        _, _, _, _, _, _, A1_s, A2_s, A3_s = s
                    else:
                        A1_s, A2_s, A3_s = (
                            current_comet_data['A1'],
                            current_comet_data['A2'],
                            current_comet_data['A3']
                        )

                    p_clone = sim.particles[f"{idx+1}"]
                    r_vec = np.array([p_clone.x, p_clone.y, p_clone.z])
                    v_vec = np.array([p_clone.vx, p_clone.vy, p_clone.vz])

                    r_norm = np.linalg.norm(r_vec)
                    A_vec = np.array([A1_s, A2_s, A3_s]) * 1e-8
                    g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)
                    F_vec_rtn = g * A_vec
                    C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)
                    F_vec_inertial = C_rtn2eci @ F_vec_rtn

                    p_clone.ax += F_vec_inertial[0]
                    p_clone.ay += F_vec_inertial[1]
                    p_clone.az += F_vec_inertial[2]
            sim.additional_forces = add_NGAs
            data_to_write["Traj_NGA"]["0"][j] = [p_mean.x, p_mean.y, p_mean.z]
            if NBP:
                for idx, planet in enumerate(planets):
                    p_planet = sim.particles[idx+1]
                    data_to_write["Planet_orb"][planet][j] = [p_planet.x, p_planet.y, p_planet.z]
            for idx in range(1, N_samples + 1):
                p_clone = sim.particles[f"{idx}"]
                data_to_write["Traj_NGA"][f"{idx}"][j] = [p_clone.x, p_clone.y, p_clone.z]
        else:
            data_to_write["Traj_NGAf"]["0"][j] = [p_mean.x, p_mean.y, p_mean.z]
            for idx in range(1, N_samples + 1):
                p_clone = sim.particles[f"{idx}"]
                data_to_write["Traj_NGAf"][f"{idx}"][j] = [p_clone.x, p_clone.y, p_clone.z]
        
end = timer.perf_counter()
runtime = end-start

sim_info["Info"] = {
             "Body": body,
             "Arc": arc,
             "Model": model,
             "Int": integrator,
             "dt": sim.dt,
             "Int_time": runtime,
             "N_clones": N_samples,
             "perturbing_bodies": planets
    }

data_to_write["Sim_time"] = {
        'Start': start_time,
        'End': end_time,
        'Time': times,
        'N_lin': N_time
        }

data_to_write["mean_data"]      = mean_to_sample
data_to_write["sampled_data"]   = samples
data_to_write["covariance"]     = covariance
print("done")

try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")

with open(f"{directory_name}/data_{integrator}_{N_samples}_{dt}.pkl", "wb") as f:
    pickle.dump(data_to_write, f)
with open(f"{directory_name}/info_{integrator}_{N_samples}_{dt}.txt", "wb") as f:
    pickle.dump(sim_info, f)
