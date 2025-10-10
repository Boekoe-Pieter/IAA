# loading packages
import numpy as np

# Import systems
import sys
import time as timer
import pickle
import re
import os
import glob
import json
import pprint

# load sim packages
from scipy import constants as const
from astropy.time import Time
from tudatpy.data.sbdb import SBDBquery
from tudatpy.astro import time_representation, element_conversion

# Import python files
sys.path.append('/Users/pieter/IAA/Coding')
import Utilities as Util       


np.set_printoptions(linewidth=160)
comets = ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/orbit_analysis_2033.00-2037.00"

files_dict = {}
for comet in comets:
    files_dict[comet] = {"covar": [], "total": []}
    
    covar_files = sorted(glob.glob(os.path.join(base_path, f"covar_{comet}_*.json")))
    total_files = sorted(glob.glob(os.path.join(base_path, f"total_{comet}_*.json")))
    
    paired_total_files = []
    for covar_file in covar_files:
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".json", "")
        
        matching_total = [tf for tf in total_files if covar_date in tf]
        if matching_total:
            paired_total_files.append(matching_total[0])
    
    files_dict[comet]["covar"] = covar_files
    files_dict[comet]["total"] = paired_total_files

for comet, data in files_dict.items():
    sim_count = 1
    for covar_file, total_file in zip(data['covar'], data['total']):
        with open(covar_file) as f:
            covar_states = json.loads(f.read())
        with open(total_file) as f:
            total_data = json.loads(f.read())

        body = total_data.get("ids")[0]
        target_sbdb = SBDBquery(body,full_precision=True)
        e = target_sbdb["orbit"]['elements'].get('e')
        a = target_sbdb["orbit"]['elements'].get('a').value 
        q = target_sbdb["orbit"]['elements'].get('q').value 
        i = np.deg2rad(target_sbdb["orbit"]['elements'].get('i').value)
        om = np.deg2rad(target_sbdb["orbit"]['elements'].get('om').value)
        w = np.deg2rad(target_sbdb["orbit"]['elements'].get('w').value)
        M = np.deg2rad(target_sbdb["orbit"]['elements'].get('ma').value)

        # saving directory
        rel_dir = os.path.dirname(os.path.abspath(__file__))
        directory_name = f"{rel_dir}/data_{body}"

        # environment
        primary = "Sun"
        planets = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

        # sim details
        N_time = 10000
        Integrator = "trace"
        timestep = 1
        N_samples = 1000

        # perturbations
        NBP = False
        Rel = False

        # saving dictionary 
        data_to_write = {
            "Nominal_trajectory":0,
            "Nominal_trajectory_times":0,
            "Monte_trajectory": {},
            "Monte_trajectory_times": {},
            "Initial_condition": 0,
            "Sampled_data": 0,
        }


        # ---------------------------------
        # Initial conditions
        initial_conidtions = np.array(covar_states["state_vect"])
        covariance = np.array(covar_states["covar"])
        samples = np.random.multivariate_normal(initial_conidtions, covariance, size=N_samples)

        # ---------------------------------
        # Simulator
        # start_time = total_data["objects"][body]["observations"].get("earliest iso")
        start_time = covar_states["epoch"]
        end_time = total_data["objects"][body]["elements"].get("Tp_iso")

        start_time_MJD = time_representation.julian_day_to_modified_julian_day(start_time)
        end_time_MJD =  Time(end_time, format='isot', scale='utc').mjd

        sim = Util.create_sim(primary,start_time_MJD,Integrator,timestep)
        times = np.arange(start_time_MJD, end_time_MJD+timestep, timestep)
        data_to_write['Nominal_trajectory_times'] = times
        data_to_write['Monte_trajectory_times'] = times

        # ---------------------------------
        # adding perturbations
        if NBP:
            Util.add_NBP(sim,start_time_MJD,planets)
        if Rel:
            Util.add_Rel(sim)

        # ---------------------------------
        # add mean orbit
        def add_reference(ref_sim):
            ref_sim.add(
                primary=ref_sim.particles[0],
                m=0.0,
                a=(q) / (1 - e),
                e=e,
                inc=i,
                Omega=om,
                omega=w,
                T=end_time_MJD,
                hash='0'
            )

        add_reference(sim)
        data_to_write["Nominal_trajectory"] = np.zeros((len(times), 6))

        # ---------------------------------
        # add pertubed conditions
        def add_clones(ref_sim):
            for idx, s in enumerate(samples):
                Sx, Sy, Sz, Svx, Svy, Svz = s
                hash_id = f"{idx+1}"  
                ref_sim.add(
                    m = 0.0,
                    x = Sx,
                    y = Sy,  
                    z = Sz,
                    vx = Svx,
                    vy = Svy,
                    vz = Svz,
                    hash=hash_id
                )

                data_to_write["Monte_trajectory"][idx+1] = np.zeros((len(times), 6))
        
        add_clones(sim)

        print(
            f"Integrating \n"
            f"integrator: {Integrator}\n"
            f"Samples: {N_samples}\n"
            )
        # print(sim.status())
        sim.move_to_com()
        start = timer.perf_counter()
        for j, time in enumerate(times):
            sim.integrate(time)
            p_reference = sim.particles["0"]
            data_to_write["Nominal_trajectory"][j] = [p_reference.x*const.au, p_reference.y*const.au, p_reference.z*const.au,p_reference.vx*const.au/const.day, p_reference.vy*const.au/const.day, p_reference.vz*const.au/const.day]
            for idx in range(1, N_samples+1):
                p_clone = sim.particles[f"{idx}"]
                data_to_write["Monte_trajectory"][idx][j] = [p_clone.x*const.au, p_clone.y*const.au, p_clone.z*const.au,p_clone.vx*const.au/const.day, p_clone.vy*const.au/const.day, p_clone.vz*const.au/const.day]   
        end = timer.perf_counter()
        runtime = end-start

        sim_info = {
            "Data author": "Pedro Lacerda",
            "Covar data": covar_file[45:],
            "Full data": total_file[45:],
            "Body": body,
            "Simulator": "TUDAT",
            "Integrator": Integrator,
            "Sim_time": {
                'Start_iso': total_data["objects"][body]["observations"].get("earliest iso"),
                'End_iso': total_data["objects"][body]["elements"].get("Tp_iso"),
                'last obs': total_data["objects"][body]["observations"].get("latest iso"),
                'Start_MJD': start_time,
                'End_MJD': end_time,
                }, 
            "timestep": timestep,
            "CPU_Time": runtime,
            "N_clones": N_samples,
            "origin": 'CoM',
            "Orientation": 'ECLIPJ2000',
            "Environment": 'Sun',
            "Planetary positions":'SPICE DE-421 & TUDAT Standard kernels (https://py.api.tudat.space/en/latest/interface/spice.html#tudatpy.interface.spice.load_standard_kernels)',
            "Perturbations": {
                'NBP': 'Off',
                'Relativity': "Off",
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