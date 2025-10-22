# loading packages
import numpy as np
import math

# Import systems
import sys
import time as timer
import pickle
import os
import glob
import json
import pprint

# load sim packages
from scipy import constants as const
from tudatpy.astro import time_representation

# Import python files
sys.path.append('/Users/pieter/IAA/Coding')
import Utilities as Util       

np.set_printoptions(linewidth=160)
comets = ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/orbit_analysis_2033.00-2037.00"

# environment
primary = "Sun"
perturbing_bdoies = ["Mercury",
                     "Venus",
                     "Earth", #"Luna",
                     "Mars", "Phobos","Deimos",
                     "Jupiter", "Io", "Europa", "Ganymede", "Callisto", 
                     "Saturn", "Titan","Rhea","Iapetus","Dione","Tethys","Enceladus","Mimas",
                     "Uranus", "Miranda","Ariel","Umbriel","Titania","Oberon", 
                     "Neptune", "Triton"
                     ]

env = perturbing_bdoies

# sim details
Integrator = "trace"
timestep = 0.1
N_samples = 1000

# perturbations
NBP = False
Rel = False

files_dict = {}
for comet in comets:
    files_dict[comet] = {"covar": [], "total": []}
    
    covar_files = sorted(glob.glob(os.path.join(base_path, f"covar_{comet}_*.txt")))
    total_files = sorted(glob.glob(os.path.join(base_path, f"total_{comet}_*.json")))
    
    paired_total_files = []
    for covar_file in covar_files:
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".json", "")
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".txt", "")

        matching_total = [tf for tf in total_files if covar_date in tf]
        if matching_total:
            paired_total_files.append(matching_total[0])
    
    files_dict[comet]["covar"] = covar_files
    files_dict[comet]["total"] = paired_total_files

for comet, data in files_dict.items():
    sim_count = 1
    for covar_file, total_file in zip(data['covar'], data['total']):
        start = timer.perf_counter()
        with open(total_file) as f:
            total_data = json.loads(f.read())

        body = total_data.get("ids")[0]

        # saving directory
        rel_dir = os.path.dirname(os.path.abspath(__file__))
        directory_name = f"{rel_dir}/data_{body}"

        # saving dictionary 
        data_to_write = {
            "Nominal_trajectory":0,
            "Nominal_trajectory_times":0,
            "perturbing_bodies": {},
            "Nominal_trajectory_fit":0,
            "Monte_trajectory": {},
            "Monte_trajectory_times": {},
            "Initial_condition": 0,
            "Sampled_data": 0,
        }

        # ---------------------------------
        # Simulator
        covariance, _= Util.extract_traditional_covariance(covar_file)
        # _, start_time_epoch_covariance= Util.extract_traditional_covariance(covar_file)

        # start_time_epoch_oscullating = total_data["objects"][body]["elements"].get("epoch")
        start_time_first_obs = total_data["objects"][body]["observations"].get("earliest")

        end_time = total_data["objects"][body]["elements"].get("Tp_iso").replace("Z", "")

        start_time_MJD = time_representation.julian_day_to_modified_julian_day(start_time_first_obs)
        end_time_MJD =  time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.iso_string_to_epoch(str(end_time))))

        # end_time_MJD_int = math.ceil(end_time_MJD)+5

        sim = Util.create_sim(primary,start_time_MJD,Integrator,timestep)
        # times = np.arange(start_time_MJD, end_time_MJD_int+timestep, timestep)
        times = np.linspace(start_time_MJD, end_time_MJD, 10000)

        data_to_write['Nominal_trajectory_times'] = times
        data_to_write['Monte_trajectory_times'] = times

        # ---------------------------------
        # adding perturbations
        def add_NBP(ref_sim, start_time, bodies):
            start_time_JD = start_time + 2400000.5
            JD_str = f'JD{start_time_JD}'
            for perturbing in bodies:
                ref_sim.add(perturbing,date=JD_str,hash=f"{perturbing}")
                data_to_write["perturbing_bodies"][perturbing] = np.zeros((len(times), 6))

        if NBP:
            add_NBP(sim,start_time_MJD,env)
        if Rel:
            Util.add_Rel(sim)

        # ---------------------------------
        # add mean orbit
        SBDB = Util.SBDB(body,end_time_MJD)
        e,a,q,i,om,w,Tp_mjd = SBDB
        sim.add(
            primary=sim.particles[0],
            m=0.0,
            a=q/(1.-e),
            e=e,
            inc=i,
            Omega=om,
            omega=w,
            T=Tp_mjd,
            hash='SBDB'
        )

        # add_SBDB(sim,SBDB)
        data_to_write["Nominal_trajectory"] = np.zeros((len(times), 6))

        # ---------------------------------
        # add fitted reference
        initial_conidtions = Util.initial_conditions(total_data,body)
        e,a,q,i,om,w,Tp_mjd = initial_conidtions
        sim.add(
            primary=sim.particles[0],
            m=0.0,
            a=q/(1.-e),
            e=e,
            inc=i,
            Omega=om,
            omega=w,
            T=Tp_mjd,
            hash='FIT'
        )
        
        data_to_write["Nominal_trajectory_fit"] = np.zeros((len(times), 6))

        # ---------------------------------
        # add pertubed conditions
        # covar state is in Tp ,e, q , 1/a , i  , omega,  Omega
        # initial conditions is in: e,a,q,i,om,w,Tp_mjd
        e,a,q,i,om,w,Tp_mjd = initial_conidtions
        # Tp_jd = time_representation.modified_julian_day_to_julian_day(Tp_mjd)
        sampled_intial = np.array([Tp_mjd,e,q,1/a,np.rad2deg(i),np.rad2deg(w),np.rad2deg(om)])
        samples = np.random.multivariate_normal(sampled_intial, covariance, size=N_samples)
        def add_clones(ref_sim):
            valid_idx = 0
            valid_ids = []
            for idx, s in enumerate(samples):
                Tp_mjd, e, q, inv_a, i_deg, w_deg, om_deg = s
                if e <= 0 or e >= 2:
                    continue
                if (q/(1-e) > 0 and e > 1) or (q/(1-e) < 0 and e < 1):
                    continue
                valid_idx += 1
                hash_id = f"{valid_idx}"
                ref_sim.add(
                    primary=ref_sim.particles[0],
                    m=0.0,
                    a=q/(1-e),
                    e=e,
                    inc=np.deg2rad(i_deg),
                    Omega=np.deg2rad(om_deg),
                    omega=np.deg2rad(w_deg),
                    T=Tp_mjd,
                    hash=hash_id
                )

                data_to_write["Monte_trajectory"][valid_idx] = np.zeros((len(times), 6))
                valid_ids.append(valid_idx)

            return valid_ids

        valid_ids=add_clones(sim)

        print(
            f"Integrating \n"
            f"integrator: {Integrator}\n"
            f"Samples: {len(valid_ids)}\n"
            )
        sim.move_to_com()
        for j, time in enumerate(times):
            sim.integrate(time)
            p_reference = sim.particles["SBDB"]
            p_reference_fit = sim.particles["FIT"]
            data_to_write["Nominal_trajectory"][j] = [p_reference.x*const.au, p_reference.y*const.au, p_reference.z*const.au,p_reference.vx*const.au/const.day, p_reference.vy*const.au/const.day, p_reference.vz*const.au/const.day]
            data_to_write["Nominal_trajectory_fit"][j] = [p_reference_fit.x*const.au, p_reference_fit.y*const.au, p_reference_fit.z*const.au,p_reference_fit.vx*const.au/const.day, p_reference_fit.vy*const.au/const.day, p_reference_fit.vz*const.au/const.day]
            if NBP:
                for perturbing in env:
                    p_body = sim.particles[f"{perturbing}"]
                    data_to_write["perturbing_bodies"][perturbing][j] = [p_body.x*const.au, p_body.y*const.au, p_body.z*const.au,p_body.vx*const.au/const.day, p_body.vy*const.au/const.day, p_body.vz*const.au/const.day]
            for idx in valid_ids:
                p_clone = sim.particles[f"{idx}"]
                data_to_write["Monte_trajectory"][idx][j] = [p_clone.x*const.au, p_clone.y*const.au, p_clone.z*const.au,p_clone.vx*const.au/const.day, p_clone.vy*const.au/const.day, p_clone.vz*const.au/const.day]   
        end = timer.perf_counter()
        runtime = end-start

        sim_info = {
            "Data author": "Pedro Lacerda",
            "Covar data": covar_file[45:],
            "Full data": total_file[45:],
            "Body": body,
            "Simulator": "Rebound",
            "Integrator": Integrator,
            "Sim_time": {
                'Start_iso': total_data["objects"][body]["observations"].get("earliest iso"),
                'End_iso': total_data["objects"][body]["elements"].get("Tp_iso"),
                'last obs': total_data["objects"][body]["observations"].get("latest iso"),
                'Start_MJD': start_time_MJD,
                'End_MJD': end_time_MJD,
                }, 
            "timestep": timestep*const.day,
            "CPU_Time": runtime,
            "N_clones": N_samples,
            "N_valid_clones": len(valid_ids),
            "Origin": 'SSB',
            "Orientation": 'ECLIPJ2000',
            "Environment": env,
            "Planetary positions":'Horizons',
            "Perturbations": {
                'NBP': f'{NBP}',
                'Relativity': f'{Rel}',
                },
            "used_obs": total_data["objects"][body]["observations"].get("used"),
            }

        data_to_write["Initial_conidtions"]      = initial_conidtions
        data_to_write["Sampled_data"]            = samples

        print("done")

        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")

        subdir = f"{directory_name}/Simulation_{sim_count}"

        try:
            os.mkdir(subdir)
            print(f"Directory '{subdir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{subdir}' already exists.")

        with open(f"{subdir}/Rebound_Simulation_data.pkl", "wb") as f:
            pickle.dump(data_to_write, f)
        with open(f"{subdir}/Rebound_Simulation_Info.pkl", "wb") as f:
            pickle.dump(sim_info, f)
        with open(f"{subdir}/Rebound_Simulation_Info.txt", "w") as f:
            pprint.pprint(sim_info, stream=f, indent=2, width=80, sort_dicts=False)
        
        sim_count+=1