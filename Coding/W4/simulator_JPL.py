# loading packages
import numpy as np

# Import systems
import sys
import time as timer
import pickle
import re
import os
import pprint
import requests
import json

# load sim packages
from scipy import constants as const
from tudatpy.data.sbdb import SBDBquery
from tudatpy.astro import time_representation

# Import python files
sys.path.append('/Users/pieter/IAA/Coding')
import Utilities as Util       
from NGAs_prefered import NGA_data   
from covars import covar_matrices

np.set_printoptions(linewidth=160)

# ---------------------------------
# info

# sim details
N_time = 10000
dt = 1 
integrator = "trace"
N_samples = 1000

# perturbations
NGA = True
NBP = True
Rel = True

# data
url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

classes = ["COM", "PAR", "HYP"]
all_results = []

for sbclass in classes:
    request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF", "A2|DF", "DT|ND"]}'
    request_dict = {
        'fields': 'spkid',
        'sb-class': sbclass, 
        'sb-cdata': request_filter,
    }
    response = requests.get(url, params=request_dict)
    if response.ok:
        all_results.extend(response.json().get("data", []))

for body in all_results:
    target_sbdb = SBDBquery(body,full_precision=True,covariance="mat")
    com_name = target_sbdb["object"]["des"]
    full_name = target_sbdb["object"]["fullname"]
    save_body = com_name

    # saving directory
    rel_dir = os.path.dirname(os.path.abspath(__file__))
    directory_name = f"{rel_dir}/data_{save_body}"

    primary = "sun"
    planets = ["Mercury","Venus","Earth","Moon","mars","Jupiter","Saturn","Uranus","Neptune"]

    sim_info = { 
        "Info": {
        "Body": full_name,
        "Data": url,
        "Filter": request_filter,
        "Requester": request_dict,
        "classes": classes,
        "Assumption": "Standard H2O g(r), no mention from JPL on model",
        "Int": integrator,
        "dt": 0,
        "CPU_time": 0,
        "N_clones": 0,
        "perturbing_bodies": 0,
        "Perturbations": {
                'NBP': NBP,
                'NGA': NGA,
                'Rel': Rel
                }
        },
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
    orbit_data = target_sbdb["orbit"]
    orbit_elements = orbit_data["elements"]

    arc1_date, arc2_date, T_perihelium_TBP, q, ecc, aop, RAAN, i, a = (
        orbit_data["first_obs"],
        orbit_data["last_obs"],
        orbit_elements["tp"],
        orbit_elements["q"],
        orbit_elements["e"],
        orbit_elements["w"],
        orbit_elements["om"],
        orbit_elements["i"],
        orbit_elements["a"],
    )    

    A1 = orbit_data["model_pars"].get("A1")
    A2 = orbit_data["model_pars"].get("A2")
    A3 = orbit_data["model_pars"].get("A3")
    DT = orbit_data["model_pars"].get("DT")

    covariance = orbit_data["covariance"].get("data")

    arc1_date, arc2_date, T_perihelium_TBP, q, ecc, aop, RAAN, i, a, A1, A2, A3, DT = (
            arc1_date,
            arc2_date,
            T_perihelium_TBP.value,
            q.value,
            ecc,
            aop.value,
            RAAN.value,
            i.value,
            a.value,
            A1.value if A1 is not None else 0,
            A2.value if A2 is not None else 0,
            A3.value if A3 is not None else 0,
            DT.value if DT is not None else 0,
        )

    y1, m1, d1 = map(int, arc1_date.split('-'))
    y2, m2, d2 = map(int, arc2_date.split('-'))

    #time_representation.seconds_since_epoch_to_julian_day
    arc1_MJD            =   time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.date_time_components_to_epoch(y1, m1, d1,0,0,0)))
    T_periheliom_mjd    =   time_representation.julian_day_to_modified_julian_day(T_perihelium_TBP)
    arc2_MJD            =   time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.date_time_components_to_epoch(y2, m2, d2,0,0,0)))
    
    # ---------------------------------
    # Initial conditions
    param_dict = {
        "e": ecc,
        "q": q,
        "tp": T_periheliom_mjd,
        "node": np.deg2rad(RAAN),
        "peri": np.deg2rad(aop),
        "i": np.deg2rad(i),
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "DT": DT,
    }

    covariance_labels = orbit_data["covariance"].get("labels")
    all_names = ["e", "q", "tp", "node", "peri", "i", "A1", "A2", "A3", "DT"]
    covar_names = [name for name in all_names if name in covariance_labels]

    mean_to_sample = np.array([param_dict[name] for name in covar_names])

    samples = np.random.multivariate_normal(mean_to_sample, covariance, size=N_samples)

    # ---------------------------------
    # Simulator
    start_time = arc1_MJD
    end_time = T_periheliom_mjd 

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
            T=T_periheliom_mjd,
            hash='0'
        )

    add_mean(sim)
    data_to_write["Traj_NGA"]["0"] = np.zeros((len(times), 3))
    data_to_write["Traj_NGAf"]["0"] = np.zeros((len(times), 3))

    # ---------------------------------
    # add pertubed conditions
    def add_clones(ref_sim, samples, covar_names):
        for idx, s in enumerate(samples):
            params = dict(zip(covar_names, s))

            ecc_s   = params["e"]
            q_s     = params["q"]
            q_date  = params["tp"]
            RAAN_s  = params["node"]
            aop_s   = params["peri"]
            i_s     = params["i"]

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

    add_clones(sim, samples, covar_names)

    # ---------------------------------
    # Run simulation
    print(
        f"Integrating \n"
        f"integrator: {integrator}\n"
        f"dt: {dt} day\n"
        f"Samples: {N_samples}\n"
        )
    
    start = timer.perf_counter()
    # ---------------------------------
    # create asymmetric sim
    if param_dict["DT"] != 0:
        print("creating asymmetric sim and reference orbit")
        print(DT)
        asym_data = {}
        start_time_asym = start_time - DT
        end_time_asym = end_time - DT
        asym_sim = Util.create_sim(primary,start_time_asym,integrator,dt)

        if NBP:
            Util.add_NBP(asym_sim,start_time_asym,planets)
        if Rel:
            Util.add_Rel(asym_sim)

        add_mean(asym_sim)
        add_clones(asym_sim, samples, covar_names)
        asym_times = np.linspace(start_time_asym,end_time_asym,N_time)         

        asym_data["0"] = np.zeros((len(asym_times), 6))
        for idx in range(1, N_samples + 1):
            asym_data[f"{idx}"] = np.zeros((len(asym_times), 6))
        for k, time in enumerate(asym_times):
            asym_sim.integrate(time)
            p_mean = asym_sim.particles["0"]
            asym_data["0"][k] = [p_mean.x, p_mean.y, p_mean.z,p_mean.vx, p_mean.vy, p_mean.vz]
            for idx in range(1, N_samples + 1):
                p_clone = asym_sim.particles[f"{idx}"]
                asym_data[f"{idx}"][k] = [p_clone.x, p_clone.y, p_clone.z,p_clone.vx, p_clone.vy, p_clone.vz]

    for N in range(2):
        print("integrating with NGA" if N == 1 else "integrating w/o NGA")
        for j, time in enumerate(times):
            sim.integrate(time)
            p_mean = sim.particles["0"]

            if N == 1:
                def add_NGAs(reb_sim):
                    p_mean = sim.particles["0"]
                    if param_dict["DT"] != 0:
                        r_vec = np.array(asym_data["0"][j,:3])
                        v_vec = np.array(asym_data["0"][j,3:])
                    else:
                        r_vec = np.array([p_mean.x, p_mean.y, p_mean.z])
                        v_vec = np.array([p_mean.vx, p_mean.vy, p_mean.vz])

                    r_norm = np.linalg.norm(r_vec)

                    A_vec = np.array([
                        A1,
                        A2,
                        A3
                    ])

                    m = 2.15
                    n = 5.093
                    k = 4.6142
                    r0 = 2.808
                    alpha = 0.1113

                    g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)
                    F_vec_rtn = g * A_vec
                    C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)
                    F_vec_inertial = C_rtn2eci @ F_vec_rtn

                    p_mean.ax += F_vec_inertial[0]
                    p_mean.ay += F_vec_inertial[1]
                    p_mean.az += F_vec_inertial[2]

                    for idx, s in enumerate(samples):
                        params = dict(zip(covar_names, s))
                        A1_s = params.get("A1")
                        A2_s = params.get("A2")
                        A3_s = params.get("A3") if params.get("A3") is not None else 0

                        p_clone = sim.particles[f"{idx+1}"]
                        if param_dict["DT"] != 0:
                            r_vec = np.array([asym_data[f"{idx+1}"][j,:3]])
                            v_vec = np.array([asym_data[f"{idx+1}"][j,3:]])
                        else:
                            r_vec = np.array([p_mean.x, p_mean.y, p_mean.z])
                            v_vec = np.array([p_mean.vx, p_mean.vy, p_mean.vz])
                        
                        r_vec = np.array([p_clone.x, p_clone.y, p_clone.z])
                        v_vec = np.array([p_clone.vx, p_clone.vy, p_clone.vz])

                        r_norm = np.linalg.norm(r_vec)
                        A_vec = np.array([A1_s, A2_s, A3_s])

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

    runtime = end - start

    sim_info["Info"] ={
                "Body": full_name,
                "Data": url,
                "Filter": request_filter,
                "Requester": request_dict,
                "classes": classes,
                "Assumption": "1) Standard H2O g(r), no mention from JPL on model, 2) Covariance DT not used (but is sampled)",
                "Int": integrator,
                "dt": sim.dt,
                "CPU_time": runtime,
                "N_clones": N_samples,
                "Perturbations": {
                                'NBP': NBP,
                                'NGA': NGA,
                                'Rel': Rel
                                },
                "Covar_labels": covar_names
        }
    sim_info["Info"]["perturbing_bodies"] = planets if NBP else [None]
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
    with open(f"{directory_name}/info_{integrator}_{N_samples}_{dt}.pkl", "wb") as f:
        pickle.dump(sim_info, f)
    with open(f"{directory_name}/info_{integrator}_{N_samples}_{dt}.txt", "w") as f:
        pprint.pprint(sim_info, stream=f, indent=2, width=80)
