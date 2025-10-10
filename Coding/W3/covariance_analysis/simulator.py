# loading packages
import sys
import numpy as np
import rebound
import reboundx 
import pickle
import re
from scipy import constants as const
import time as timer

import sys
sys.path.append('/Users/pieter/IAA/Coding')

import Utilities as Util       # works directly
import NGAs                    # imports NGAs.py
from NGAs import NGA_data      # imports the dictionary


np.set_printoptions(linewidth=160)

# ---------------------------------
# info
data = 'Coding/W2P1/unfiltered.txt'

body = 'C/2001 Q4'
primary = "sun"
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

N_time = 10000
dt = 1 # days
integrator = "whfast"

N_samples = 1000

NBP = True
NGA = True
Relativity = True

# ---------------------------------
# load data
comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)

current_comet_data = data_dict[body]["Full"]["ng"]  

arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)
print(arc1_mjd, arc2_mjd, )
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
def create_sim():
    sim = rebound.Simulation()
    sim.units = ('Days', 'AU', 'Msun')
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    rebx.add_force(gr)
    c_au_per_day = const.c * const.day / const.au
    gr.params["c"] = c_au_per_day
    return sim

start_time = arc1_mjd
end_time = T_perihelium_mjd 

sim = create_sim()
sim.t = start_time
sim.integrator = integrator

sim.dt = dt
times = np.linspace(start_time, end_time, N_time)

data = {
    "info": {
        "body": body,
        "dt": sim.dt,
        "int": integrator,
        "int_time": 0,
        "N_clones": 0
    },
    "Sim_time": {
        'start': start_time,
        'end': end_time,
        'time':times,
        'N': N_time
        }, 

    "trajectories": {},
    "Osculating": {},
    "gr": {},
    "mean_data": mean_to_sample,
    "sampled_data": samples,
    "covariance": covariance,
    "Perturbations": {'NBP': NBP,
                      'NGA': NGA,
                      'Rel': Relativity},
}

if current_comet_data['tau'] != 0:
    sim_asym = create_sim()
    sim_asym.add(primary)
    tau = current_comet_data['tau']
    start_asym = start_time - tau 
    end_asym = end_time - tau 
    sim_asym.t = start_asym
    JD = start_asym + 2400000.5 
    JD_str = 'JD'+str(JD)

    for planet in planets:
        sim.add(planet, date = JD_str)
    times_asym = np.linspace(start_asym, end_asym, N_time)

sim.add(primary)
JD = start_time + 2400000.5
JD_str = 'JD'+str(JD)
for planet in planets:
    data["trajectories"][planet] = np.zeros((len(times), 3))
    sim.add(planet, date = JD_str)

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
data["trajectories"]["0"] = np.zeros((len(times), 3))
data["Osculating"]["0"] = np.zeros((len(times), 6))
data['gr']["0"] = np.zeros((len(times), 1))
# ---------------------------------
# add pertubed conditions
def add_perturbed(ref_sim):
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
    data["trajectories"][f"{idx}"] = np.zeros((len(times), 3))
    data["Osculating"][f"{idx}"] = np.zeros((len(times), 6))
    data['gr'][f"{idx}"] = np.zeros((len(times), 1))

add_perturbed(sim)

# ---------------------------------
# Create asymmetric reference sim
if current_comet_data['tau'] != 0:
    print("asymmetric comet, creating reference sim")
    add_mean(sim_asym)
    add_perturbed(sim_asym)
    asym_reference = {
        f"{i}": np.zeros((len(times_asym), 6))
        for i in range(N_samples+1)}
    
    for j, time in enumerate(times_asym):
        sim_asym.integrate(time)
        asym_reference[f"0"][j] = [sim_asym.particles['0'].x, sim_asym.particles['0'].y, sim_asym.particles['0'].z, 
                                   sim_asym.particles['0'].vx, sim_asym.particles['0'].vy, sim_asym.particles['0'].vz]
        for idx in range(N_samples):
            p = sim_asym.particles[f"{idx+1}"]
            asym_reference[f"{idx+1}"][j] = [p.x, p.y, p.z, p.vx, p.vy, p.vz]
print(
    f"Integrating \n"
    f"integrator: {integrator}\n"
    f"dt: {dt} day\n"
    f"Samples: {N_samples}\n"
    )

start = timer.perf_counter()
for j, time in enumerate(times):
    def add_NGAs(reb_sim):
        p_mean = sim.particles["0"]
        if current_comet_data['tau'] != 0:
            r_vec = np.array(asym_reference["0"][j, :3])
            v_vec = np.array(asym_reference["0"][j, 3:])
        else:
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
        data["gr"]["0"][j] = g
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
            if current_comet_data['tau'] != 0:
                r_vec = np.array(asym_reference[f"{idx+1}"][j, :3])
                v_vec = np.array(asym_reference[f"{idx+1}"][j, 3:])
            else:
                r_vec = np.array([p_clone.x, p_clone.y, p_clone.z])
                v_vec = np.array([p_clone.vx, p_clone.vy, p_clone.vz])

            r_norm = np.linalg.norm(r_vec)
            A_vec = np.array([A1_s, A2_s, A3_s]) * 1e-8
            g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)
            data["gr"][f"{idx+1}"][j] = g
            F_vec_rtn = g * A_vec
            C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)
            F_vec_inertial = C_rtn2eci @ F_vec_rtn

            p_clone.ax += F_vec_inertial[0]
            p_clone.ay += F_vec_inertial[1]
            p_clone.az += F_vec_inertial[2]
    
    sim.additional_forces = add_NGAs
    sim.integrate(time)
    p_mean = sim.particles["0"]
    data["trajectories"]["0"][j] = [p_mean.x, p_mean.y, p_mean.z]
    data["Osculating"]["0"][j] = [p_mean.a, p_mean.e, p_mean.inc,p_mean.Omega,p_mean.omega,p_mean.theta]
    for idx, planet in enumerate(planets):
        p_planet = sim.particles[idx+1]
        data["trajectories"][planet][j] = [p_planet.x, p_planet.y, p_planet.z]
    for idx in range(1, N_samples + 1):
        p_clone = sim.particles[f"{idx}"]
        data["trajectories"][f"{idx}"][j] = [p_clone.x, p_clone.y, p_clone.z]
        data["Osculating"][f"{idx}"][j] = [p_clone.a, p_clone.e, p_clone.inc,p_clone.Omega,p_clone.omega,p_clone.theta]
end = timer.perf_counter()

runtime = end-start

data["info"]["int_time"] = runtime,

print("done")

safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
with open(f"Coding/W3/data_{safe_name}_{integrator}_{N_samples}_{dt}.pkl", "wb") as f:
    pickle.dump(data, f)
