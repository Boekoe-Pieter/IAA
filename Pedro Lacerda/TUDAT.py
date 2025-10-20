# loading packages
import numpy as np
import matplotlib.pyplot as plt

# Import systems
import pickle
import os
import json
import pprint
import glob
import requests
import re

# load sim packages
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, environment, propagation, parameters_setup,simulator
from tudatpy.astro import time_representation, element_conversion
from tudatpy import constants
from tudatpy.data.sbdb import SBDBquery

spice.load_standard_kernels()
spice.load_kernel("Pedro Lacerda/de421.bsp")
np.set_printoptions(linewidth=160)


# ----------------------------------------------------------------------
# General Sim data
N_samples = 10

Integrator = 'rkf89'
Integrator_type = 'Fixed step, Higherorder'
Timestep_global = constants.JULIAN_DAY

# ----------------------------------------------------------------------
# Define data paths
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
    sim = 1
    for covar_file, total_file in zip(data['covar'], data['total']):
        with open(covar_file) as f:
            covar_states = json.loads(f.read())
        with open(total_file) as f:
            total_data = json.loads(f.read())

        # data to save
        data_to_write = {
            "Nominal_trajectory":0,
            "Nominal_trajectory_times":0,
            "Monte_trajectory": {},
            "Monte_trajectory_times": {},
            "Initial_condition": 0,
            "Sampled_data": 0,
            "CPU_time_list": 0
        }

        # ----------------------------------------------------------------------
        # Retrieve body data
        # ----------------------------------------------------------------------

        # target body
        body = total_data.get("ids")[0]

        start_time = covar_states["epoch"]
        end_time = total_data["objects"][body]["elements"].get("Tp")
        start_time_SJD = time_representation.julian_day_to_seconds_since_epoch(start_time)
        end_time_SJD = time_representation.julian_day_to_seconds_since_epoch(end_time)

        #SBDB Method
        target_sbdb = SBDBquery(body,full_precision=True)

        e = target_sbdb["orbit"]['elements'].get('e')
        a = target_sbdb["orbit"]['elements'].get('a').value 
        q = target_sbdb["orbit"]['elements'].get('q').value 
        i = target_sbdb["orbit"]['elements'].get('i').value
        om = target_sbdb["orbit"]['elements'].get('om').value
        w = target_sbdb["orbit"]['elements'].get('w').value
        M = target_sbdb["orbit"]['elements'].get('ma').value
        n = target_sbdb["orbit"]['elements'].get('n').value
        Tp = total_data["objects"][body]["elements"].get("Tp")
        Shifter = time_representation.julian_day_to_seconds_since_epoch(Tp)
        "Convert the mean anomaly given by SBDB from the given date to the wanted starting position"
        delta_t = Shifter-start_time_SJD
        M0=M-n*delta_t/constants.JULIAN_DAY #M in degrees, n in deg/day delta_t in seconds automatically uses hyperbolic if e>1
        true_anomaly_start = element_conversion.mean_to_true_anomaly(e, np.deg2rad(M0))

        # Monte carlo sampling
        initial_conditions_AU = np.array(covar_states["state_vect"])
        initial_conditions_m = np.zeros_like(initial_conditions_AU)
        initial_conditions_m[:3] = initial_conditions_AU[:3] * constants.ASTRONOMICAL_UNIT                          # position
        initial_conditions_m[3:] = initial_conditions_AU[3:] * constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY     # velocity
        data_to_write['Initial_condition'] = initial_conditions_m

        covariance = np.array(covar_states["covar"])
        samples = np.random.multivariate_normal(initial_conditions_AU, covariance, size=N_samples)
        samples_m = np.zeros_like(samples)
        samples_m[:, :3] = samples[:, :3] * constants.ASTRONOMICAL_UNIT       
        samples_m[:, 3:] = samples[:, 3:] * constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY
        data_to_write['Sampled_data'] = samples_m

        # saving directory
        rel_dir = os.path.dirname(os.path.abspath(__file__))
        directory_name = f"{rel_dir}/data_{body}"

        # ----------------------------------------------------------------------
        # Create environment
        # ----------------------------------------------------------------------

        global_frame_origin = "SSB"
        global_frame_orientation = "ECLIPJ2000"
        central_bodies = [global_frame_origin]

        bodies_to_create = [
            "Sun",
            "Mercury",
            "Venus",

            "Earth",
            "Moon",

            "Mars",
            'Phobos',
            'Deimos',

            "Jupiter",
            'Europa',
            'Ganymede',
            'Io',
            'Callisto',

            "Saturn",
            'Titan',
            'Enceladus',

            "Uranus",
            "Neptune",
        ]

        # Create system of bodies
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation
        )

        # add comet
        bodies = environment_setup.create_system_of_bodies(body_settings)
        bodies.create_empty_body(body)
        bodies_to_propagate = [body]

        # ----------------------------------------------------------------------
        # Define accelerations
        # ----------------------------------------------------------------------
        accelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                # propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
            ],

            # "Mercury": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Venus": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Earth": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Moon": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],

            # "Mars": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Phobos": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Deimos": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],

            # "Jupiter": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            #     propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
            # ],
            # "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
            # "Europa": [propagation_setup.acceleration.point_mass_gravity()],
            # "Callisto": [propagation_setup.acceleration.point_mass_gravity()],
            # "Io": [propagation_setup.acceleration.point_mass_gravity()],

            # "Saturn": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Enceladus": [propagation_setup.acceleration.point_mass_gravity()],
            # "Titan": [propagation_setup.acceleration.point_mass_gravity()],

            # "Uranus": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            # "Neptune": [
            #     propagation_setup.acceleration.point_mass_gravity(),
            # ],
            
        }

        acceleration_settings = {}
        acceleration_settings[body] = accelerations

        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # ----------------------------------------------------------------------
        # Define integrator and propagator
        # ----------------------------------------------------------------------
        initial_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=bodies.get("Sun").gravitational_parameter,
            semi_major_axis=a*constants.ASTRONOMICAL_UNIT,
            eccentricity=e,
            inclination=np.deg2rad(i),
            argument_of_periapsis=np.deg2rad(w),
            longitude_of_ascending_node=np.deg2rad(om),
            true_anomaly=true_anomaly_start,
        )
          
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = Timestep_global,
            coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_89,
            order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )  

        termination_condition = propagation_setup.propagator.time_termination(end_time_SJD)

        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_position("Mercury",   "Sun"),
            propagation_setup.dependent_variable.relative_position("Venus",     "Sun"),
            propagation_setup.dependent_variable.relative_position("Earth",     "Sun"),
            propagation_setup.dependent_variable.relative_position("Mars",      "Sun"),
            propagation_setup.dependent_variable.relative_position("Jupiter",   "Sun"),
            propagation_setup.dependent_variable.relative_position("Saturn",    "Sun"),
            propagation_setup.dependent_variable.relative_position("Uranus",    "Sun"),
            propagation_setup.dependent_variable.relative_position("Neptune",   "Sun"),
            ]

        propagator_settings = propagation_setup.propagator.translational(
            central_bodies=central_bodies,
            acceleration_models=acceleration_models,
            bodies_to_integrate=bodies_to_propagate,
            initial_states=initial_state,
            initial_time=start_time_SJD,
            integrator_settings=integrator_settings,
            termination_settings=termination_condition,
            output_variables=dependent_variables_to_save,)

        dynamics_simulator = simulator.create_dynamics_simulator(
            bodies, propagator_settings
        )

        state_hist = dynamics_simulator.propagation_results.state_history
        data_to_write['Nominal_trajectory'] = np.vstack(list(state_hist.values()))
        data_to_write['Nominal_trajectory_times'] = np.vstack(list(state_hist.keys()))

        cpu_time = np.zeros((N_samples,1))
        
        # ----------------------------------------------------------------------
        # Perform monte carlo
        # ----------------------------------------------------------------------
        for i, sampled_conditions in enumerate(samples_m):
            propagator_settings_monte = propagation_setup.propagator.translational(
                central_bodies=central_bodies,
                acceleration_models=acceleration_models,
                bodies_to_integrate=bodies_to_propagate,
                initial_states=sampled_conditions,
                initial_time=start_time_SJD,
                integrator_settings=integrator_settings,
                termination_settings=termination_condition,
            )

            dynamics_simulator_monte = simulator.create_dynamics_simulator(
                bodies, propagator_settings_monte
            )

            state_hist = dynamics_simulator_monte.propagation_results.state_history
            data_to_write['Monte_trajectory'][i] = np.vstack(list(state_hist.values()))
            data_to_write['Monte_trajectory_times'][i] = np.vstack(list(state_hist.keys()))

            CPU_time = dynamics_simulator_monte.cumulative_computation_time_history
            Function_evaluations = dynamics_simulator_monte.cumulative_number_of_function_evaluations
            Total_cpu_time = list(CPU_time.values())[-1]
            Total_Function_evaluations = list(Function_evaluations.values())[-1]
            cpu_time[i]= Total_cpu_time

        data_to_write["CPU_time_list"] = np.array(cpu_time)

        sim_info = {
            "Data author": "Pedro Lacerda",
            "Covar data": covar_file[45:],
            "Full data": total_file[45:],
            "Body": body,
            "Simulator": "TUDAT",
            "Integrator": Integrator,
            "Integrator_type": Integrator_type,
            "Sim_time": {
                'Start_iso': total_data["objects"][body]["observations"].get("earliest iso"),
                'End_iso': total_data["objects"][body]["elements"].get("Tp_iso"),
                'last obs': total_data["objects"][body]["observations"].get("latest iso"),
                'Start_epoch': start_time,
                'End_epoch': end_time,
                'Start_TDB': start_time_SJD,
                'End_TDB': end_time_SJD,
                }, 
            "timestep": Timestep_global,
            "CPU_Time": sum(cpu_time)[0],
            "N_clones": N_samples,
            "origin": global_frame_origin,
            "Orientation": global_frame_orientation,
            "Environment": bodies_to_create,
            "Planetary positions":'SPICE DE-421 & TUDAT Standard kernels (https://py.api.tudat.space/en/latest/interface/spice.html#tudatpy.interface.spice.load_standard_kernels)',
            "Perturbations": {
                'NBP': 'Off',
                'Relativity': "Off",
                },
            "used_obs": total_data["objects"][body]["observations"].get("used"),
            }


            # "Environment": bodies_to_create,
            # "Planetary positions":'SPICE DE-421 & TUDAT Standard kernels (https://py.api.tudat.space/en/latest/interface/spice.html#tudatpy.interface.spice.load_standard_kernels)',
            # "Perturbations": {
            #     'NBP': 'Point mass',
            #     'Relativity': "Schwarzschild (Sun & Jupiter)",
            #     },
            # "used_obs":      total_data["objects"][body]["observations"].get("used"),
            #     }
        
        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")

        subdir = f"{directory_name}/Simulation_{sim}"

        try:
            os.mkdir(subdir)
            print(f"Directory '{subdir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{subdir}' already exists.")

        with open(f"{subdir}/TUDAT_Simulation_data.pkl", "wb") as f:
            pickle.dump(data_to_write, f)
        with open(f"{subdir}/TUDAT_Simulation_Info.pkl", "wb") as f:
            pickle.dump(sim_info, f)
        with open(f"{subdir}/TUDAT_Simulation_Info.txt", "w") as f:
            pprint.pprint(sim_info, stream=f, indent=2, width=80, sort_dicts=False)
        
        sim+=1

# orbit_dat = total_data["objects"][body]["elements"]
# e = orbit_dat.get('e')
# a = orbit_dat.get('a') * constants.ASTRONOMICAL_UNIT
# q = orbit_dat.get('q') * constants.ASTRONOMICAL_UNIT
# i = np.deg2rad(orbit_dat.get('i'))
# AoP = np.deg2rad(orbit_dat.get('arg_per'))
# RAAN = np.deg2rad(orbit_dat.get('asc_node'))