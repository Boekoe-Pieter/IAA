"""
Week 8 is dedicated to the orbit fitting of NGA orbits via reference observations of an orbit with NGA's.
"""

# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, environment, propagation, parameters_setup, simulator
from tudatpy.astro import time_representation, element_conversion
from tudatpy.estimation import observations, observable_models, observations_setup, observable_models_setup, estimation_analysis
from tudatpy import constants

# other useful modules
import numpy as np
import random
import pprint
import math

# other libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import pickle

# python files
import Helper_file
from Plotter import Observations_Plotter as ObsPlot
from Plotter import estimation_plotter as EstPlot
from Plotter import statistics_plotter as StatPlot

np.set_printoptions(linewidth=160)

# ----------------------------------------------------------------------
# Define saving directories
# ----------------------------------------------------------------------
sys.path.append('/Users/pieter/IAA/Coding')
Spice_files_path = '/Users/pieter/IAA/Coding/Spice_files/'
rel_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Define all the planetary and satellite spice kernals
# ----------------------------------------------------------------------
spice.clear_kernels()
spice.load_standard_kernels()
spice.load_kernel("Coding/Spice_files/gm_Horizons.pck")
spice.load_kernel("Coding/Spice_files/de441_part-1.bsp")
spice.load_kernel("Coding/Spice_files/de441_part-2.bsp")
spice.load_kernel("Coding/Spice_files/sb441-n16.bsp")
spice.load_kernel("Coding/Spice_files/ura184_part-1.bsp")
spice.load_kernel("Coding/Spice_files/ura184_part-2.bsp")
spice.load_kernel("Coding/Spice_files/ura184_part-3.bsp")
spice.load_kernel("Coding/Spice_files/nep097.bsp")
spice.load_kernel("Coding/Spice_files/plu060.bsp")

# ----------------------------------------------------------------------
# Define iterations, timesteps and frames
# ----------------------------------------------------------------------

# samples
Orbit_samples = 1000
Observation_step_size = 20
np.random.seed(42)

# number of iterations for our estimation
number_of_pod_iterations = 8

# timestep of 1 hours for our estimation
timestep_global = 24*3600

# avoid interpolation errors:
time_buffer = 31*86400 
time_buffer_end = 31*86400 

# define the frame origin and orientation.
global_frame_origin = "Sun"
global_frame_orientation = "ECLIPJ2000"

# NGA on or off for orbit propagation
NGA_Flag = True
NGA_Est = True
# ----------------------------------------------------------------------
# Define the Environment
# ----------------------------------------------------------------------
bodies_to_create = [
    "Sun",

    "Mercury",

    "Venus",

    "Earth",
        "Moon",

    "Mars",
        "Phobos",
        "Deimos",
    
    "Jupiter",
        "Io",
        "Europa",
        "Ganymede",
        "Callisto",

    "Saturn",
        "Titan",
        "Rhea",
        "Iapetus",
        "Dione",
        "Tethys",
        "Enceladus",
        "Mimas",

    "Uranus",
        "Miranda",
        "Ariel",
        "Umbriel",
        "Titania",
        "Oberon", 

    "Neptune",
        "Triton",

    "Ceres",
    "Vesta", 
    "Pluto",
]

# Create system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# ----------------------------------------------------------------------
# Retrieve filtered comet SPKID's from SBDB  
# ----------------------------------------------------------------------
# classes = ["HYP","PAR","COM"]
# request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF", "A2|DF", "DT|ND"]}'

# SBDB_request = Helper_file.sbdb_query(classes,request_filter)

all_results = ["C2013US10"] #"C2001Q4","C2008A1","C2013US10"]

for body in all_results:
    # saving dictionary
    data_to_write = {
        "Spice_Reference_trajectory": 0,

        "Truth_Reference_trajectory":0,
        "Truth_Reference_trajectory_times":0,

        "N_body_trajectories": {},

        "Estimated_Reference_trajectory":{},
        "Estimated_Reference_trajectory_times":{},
        "Covariance_matrix":{},
        "NGA_Est":{},

        "Montecarlo_trajectory": {},
        "Montecarlo_trajectory_times": {},
        "observation_times": {},

        "environment": 0,
        "Sim_info": {}
        }
    
    # ----------------------------------------------------------------------
    # Retrieve comet information from SBDB
    # ----------------------------------------------------------------------
    target_body = body

    comet_designation, comet_time_information, Oscullating_elements, non_gravitational_parameters = Helper_file.sbdb_query_info(
        target_body,
        full_precision=True)
    
    spkid,name,designator = comet_designation

    first_obs,last_obs = comet_time_information

    e,a,q,i,om,w,M,n,Tp,epoch= Oscullating_elements

    A1,A2,A3,DT = non_gravitational_parameters

    NGA_array= np.array(
        [A1.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A1 is not None else 0,
        A2.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A2 is not None else 0,
        A3.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A3 is not None else 0,
        ]
        )

    directory_name = f"{rel_dir}/plots_{designator}"
    
    print("--------------------------------------------------------------------")
    print(f"SPKid: {spkid} for body {name}")
    print("First obs:", first_obs)
    print("Last obs:", last_obs)
    print("--------------------------------------------------------------------"\
          "\n")

    # saving directory for comet
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    addition = "/NGA_Est"
    # saving directory for NGA flag
    try:
        os.mkdir(f"{directory_name}{addition}")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}' already exists.")
    try:
        os.mkdir(f"{directory_name}{addition}/Observation")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}/Observation' already exists.")
    try:
        os.mkdir(f"{directory_name}{addition}/Estimation")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}/Estimation' already exists.")
    try:
        os.mkdir(f"{directory_name}{addition}/Fit_to_Truth")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}/Estimation' already exists.")
    try:
        os.mkdir(f"{directory_name}{addition}/Clone_divergence")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}/Estimation' already exists.")

    # ----------------------------------------------------------------------
    # Convert calander dates and JD to Epochs
    # ----------------------------------------------------------------------
    y1, m1, d1 = map(int, first_obs.split('-'))
    y2, m2, d2 = map(int, last_obs.split('-'))

    # Convert to epoch
    SSE_start = time_representation.date_time_components_to_epoch(y1, m1, d1, 0,0,0)
    SSE_end_cal = time_representation.date_time_components_to_epoch(y2, m2, d2, 0,0,0)

    SSE_tp = time_representation.julian_day_to_seconds_since_epoch(float(Tp))

    SSE_end = SSE_end_cal

    SSE_start_buffer = SSE_start - time_buffer
    SSE_end_buffer = SSE_end + time_buffer_end

    JD_start = time_representation.seconds_since_epoch_to_julian_day(SSE_start_buffer)
    JD_end = time_representation.seconds_since_epoch_to_julian_day(SSE_end_buffer)

    JD_start = "JD" + str(JD_start)
    JD_end = "JD" + str(JD_end)

    # ----------------------------------------------------------------------
    # Create SPK file via JPL Horizons & load comet specific kernel
    # ----------------------------------------------------------------------
    Helper_file.Horizons_SPK(Spice_files_path, 
                             spkid, 
                             JD_start, 
                             JD_end)
    
    spice.load_kernel(f"Coding/Spice_files/{spkid}.bsp")

    # ----------------------------------------------------------------------
    # Add current comet to the body settings
    # ----------------------------------------------------------------------
    body_settings.add_empty_settings(str(spkid))

    bodies = environment_setup.create_system_of_bodies(body_settings)
    central_bodies = [global_frame_origin]

    # ----------------------------------------------------------------------
    # Define accelerations, integration and propagation settings
    # ----------------------------------------------------------------------
    bodies_to_propagate = [str(spkid)]

    # Spice Initial
    initial_state_reference = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_start,
    )

    acceleration_models_reference = Helper_file.Accelerations(spkid,
                                                    bodies, 
                                                    bodies_to_propagate, 
                                                    central_bodies,
                                                    NGA_array,
                                                    NGA_Flag=NGA_Flag)

    print(np.linalg.norm(initial_state_reference[:3])/constants.ASTRONOMICAL_UNIT)

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position(body, "Sun")
        for body in bodies_to_create]
    
    integrator_settings = Helper_file.integrator_settings(timestep_global,
                                                          variable=False)
    
    propagator_settings = Helper_file.propagator_settings(integrator_settings,
                                                          central_bodies,
                                                          acceleration_models_reference,
                                                          bodies_to_propagate,
                                                          initial_state_reference,
                                                          SSE_start,
                                                          SSE_end,
                                                          dependent_variables_to_save)

    Reference_orbit_simulator = simulator.create_dynamics_simulator(
                        bodies, propagator_settings
                    )
    
    Reference_orbit_results = Reference_orbit_simulator.propagation_results.state_history
    Reference_orbit_dependent = Reference_orbit_simulator.propagation_results.dependent_variable_history

    data_to_write["Truth_Reference_trajectory"] = np.vstack(np.array(list(Reference_orbit_results.values())))
    data_to_write["Truth_Reference_trajectory_times"] = np.vstack(np.array(list(Reference_orbit_results.keys())))

    times = np.arange(SSE_start, SSE_end, timestep_global)
    if not np.isclose(times[-1], SSE_end, rtol=1e-10, atol=1e-10):
        times = np.append(times, SSE_end)

    ephemeris_spice = np.zeros((len(times), 6))
    for i, time in enumerate(times):
        ephemeris_spice[i] = spice.get_body_cartesian_state_at_epoch(
            str(spkid), central_bodies[0], global_frame_orientation, "NONE", time
        )

    data_to_write["Spice_Reference_trajectory"]=np.vstack(ephemeris_spice)

    for i, disturber in enumerate(bodies_to_create):
        Trajectory_info = np.vstack(list(Reference_orbit_dependent.values()))
        start = i * 3
        end = start + 3
        data_to_write["N_body_trajectories"][disturber] = Trajectory_info[:, start:end]

    comet_states = np.vstack(list(Reference_orbit_results.values()))
    comet_pos = comet_states[:, :3]/constants.ASTRONOMICAL_UNIT

    print("--------------------------------------------------------------------")
    print(f"propagated final state: {min(np.linalg.norm(comet_pos,axis=1))}")
    
    Final_state_spice = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_end,
    )
    print(f"Spice final state: {np.linalg.norm(np.array(Final_state_spice)[:3])/constants.ASTRONOMICAL_UNIT}")
    print(f"Difference: {np.linalg.norm((comet_pos[-1]-Final_state_spice[:3]/constants.ASTRONOMICAL_UNIT))*constants.ASTRONOMICAL_UNIT/1000}")
    print("--------------------------------------------------------------------"\
          )

    # ----------------------------------------------------------------------
    # Define Observatory
    # ----------------------------------------------------------------------
    "We define the LSST as our observatory, and is located in the center of the earth as we are not interested in the effect of one tellescope"
    "but the effect of N observations. Sim time is one observation per day as per LSST and noise is added as per LSST (source from Pedro Lacerda in ESTEC)"
    "We remove the timebuffer from the observation times as we are only interested in the actual time frame"
    station_altitude =  0          # m
    Longitude        =  [0,0,0]    # W
    Latitude         =  [0,0,0]    # N

    longitude_deg =  (Longitude[0] + Longitude[1]/60 + Longitude[2]/3600)  
    latitude_deg  =  (Latitude[0]  + Latitude[1]/60 + Latitude[2]/3600) 

    Station_latitude  = np.deg2rad(latitude_deg)
    Station_longitude = np.deg2rad(longitude_deg)

    # Add ground station
    environment_setup.add_ground_station(
        bodies.get_body("Earth"),
        "LSST",
        [Station_latitude, Station_longitude, station_altitude],
        element_conversion.cartesian_position_type)

    # Define the uplink link ends for one-way observable
    link_ends = dict()
    link_ends[observable_models_setup.links.receiver] = observable_models_setup.links.body_reference_point_link_end_id("Earth", "LSST")
    link_ends[observable_models_setup.links.transmitter] = observable_models_setup.links.body_origin_link_end_id(str(spkid))

    # Create observation settings for each link/observable
    link_definition = observable_models_setup.links.LinkDefinition(link_ends)
    observation_settings_list = [observable_models_setup.model_settings.angular_position(link_definition)]

    # ----------------------------------------------------------------------
    # Define propagator and integrator settings for estimation type
    # ----------------------------------------------------------------------
    initial_state_estimation = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_start_buffer,
    )

    dependent_variables_to_save = []
    def NGA(time: float) -> np.ndarray:
        state = bodies.get(str(spkid)).state
        r_vec = state[:3]

        v_vec = state[3:]

        r_norm = np.linalg.norm(r_vec)

        m = 2.15
        n = 5.093
        k = 4.6142
        r0 = 2.808*constants.ASTRONOMICAL_UNIT
        alpha = 0.1113

        A_vec = get_custom_parameter()

        g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)

        C_rtn2eci = rtn_to_eci(r_vec, v_vec)

        F_vec_rtn = g * A_vec
        F_vec_inertial = C_rtn2eci @ F_vec_rtn

        return F_vec_inertial  

    def rtn_to_eci(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        C = np.vstack((r_hat, t_hat, h_hat)).T
        return C

    accelerations = {
        "Sun": [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
        ],

        "Mercury": [propagation_setup.acceleration.point_mass_gravity()],

        "Venus": [propagation_setup.acceleration.point_mass_gravity()],
        
        "Earth": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Moon": [propagation_setup.acceleration.point_mass_gravity()],

        "Mars": [propagation_setup.acceleration.point_mass_gravity()],
        "Phobos": [propagation_setup.acceleration.point_mass_gravity()],
        "Deimos": [propagation_setup.acceleration.point_mass_gravity()],

        "Jupiter": [propagation_setup.acceleration.point_mass_gravity(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
        "Io": [propagation_setup.acceleration.point_mass_gravity()],
        "Europa": [propagation_setup.acceleration.point_mass_gravity()],
        "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
        "Callisto": [propagation_setup.acceleration.point_mass_gravity()],

        "Saturn": [propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
        "Titan": [propagation_setup.acceleration.point_mass_gravity()],
        "Rhea": [propagation_setup.acceleration.point_mass_gravity()],
        "Iapetus": [propagation_setup.acceleration.point_mass_gravity()],
        "Dione": [propagation_setup.acceleration.point_mass_gravity()],
        "Tethys": [propagation_setup.acceleration.point_mass_gravity()],
        "Enceladus": [propagation_setup.acceleration.point_mass_gravity()],
        "Mimas": [propagation_setup.acceleration.point_mass_gravity()],

        "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
        "Miranda": [propagation_setup.acceleration.point_mass_gravity()],
        "Ariel": [propagation_setup.acceleration.point_mass_gravity()],
        "Umbriel": [propagation_setup.acceleration.point_mass_gravity()],
        "Titania": [propagation_setup.acceleration.point_mass_gravity()],
        "Oberon": [propagation_setup.acceleration.point_mass_gravity()],

        "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
        "Triton": [propagation_setup.acceleration.point_mass_gravity()],
    
        "Ceres": [propagation_setup.acceleration.point_mass_gravity()],
        "Vesta": [propagation_setup.acceleration.point_mass_gravity()],
        "Pluto": [propagation_setup.acceleration.point_mass_gravity()],

        str(spkid): [propagation_setup.acceleration.custom_acceleration(NGA)]
        }
    acceleration_settings = {str(spkid): accelerations}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    
    propagator_settings_estimator = Helper_file.propagator_settings(
                                                          integrator_settings,
                                                          central_bodies,
                                                          acceleration_models,
                                                          bodies_to_propagate,
                                                          initial_state_estimation,
                                                          SSE_start_buffer,
                                                          SSE_end_buffer,
                                                          dependent_variables_to_save)
    
    # ----------------------------------------------------------------------
    # Define estimator for the actual wanted dynamics
    # ----------------------------------------------------------------------
    def compute_current_custom_parameter_partial(time: float,state: float) -> np.ndarray:
        state = bodies.get(str(spkid)).state
        r_vec = state[:3]
        v_vec = state[3:]
        r_norm = np.linalg.norm(r_vec)

        m = 2.15
        n = 5.093
        k = 4.6142
        r0 = 2.808*constants.ASTRONOMICAL_UNIT
        alpha = 0.1113

        g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)
        C_rtn2eci = rtn_to_eci(r_vec, v_vec)

        current_custom_parameter_partial = g * C_rtn2eci  

        return current_custom_parameter_partial
    
    A1,A2,A3 = NGA_array
    A1 = 2.195*10**-8*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2
    A2 = 0.006*10**-8*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2
    A3 = 0*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2
    print(NGA_array)
    A_vec = np.array([A1,A2,A3])
    custom_parameter = A_vec

    def get_custom_parameter():
        return custom_parameter #Get the parameter values

    def set_custom_parameter(estimated_value):
        global custom_parameter
        custom_parameter = np.array(estimated_value)  #Update the guess

    parameter_settings = parameters_setup.initial_states(propagator_settings_estimator, bodies)
    
    parameter_settings.append(
        parameters_setup.custom_parameter(
            'NGA', 3, get_custom_parameter,
            set_custom_parameter
        )
    )

    parameter_settings[-1].custom_partial_settings = [
        parameters_setup.custom_analytical_partial(
            compute_current_custom_parameter_partial,
            str(spkid),
            str(spkid),
            propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
        )
    ]

    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings_estimator
    )

    estimator = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings_estimator,
        integrate_on_creation=True)
    
    # ----------------------------------------------------------------------
    # Define observation campaign
    # ----------------------------------------------------------------------
    "We want a limited version, from 3 AU until peri so we keep simulating until we reach the mimimum observations at 3AU"
    rh = 3
    # Reference orbit
    States = np.array(list(Reference_orbit_results.values()))
    Times = np.array(list(Reference_orbit_results.keys()))

    # Find arc times
    States_AU = np.linalg.norm(States[:,:3],axis=1)/constants.ASTRONOMICAL_UNIT #AU

    mask = States_AU <= rh
    idx = np.argmax(mask)
    Start_arc_time = Times[idx]

    Minimum_obs = len(np.arange(SSE_start_buffer + time_buffer, Start_arc_time, constants.JULIAN_DAY))

    Full_observation_times = np.arange(SSE_start_buffer + time_buffer, SSE_end_buffer - time_buffer_end, constants.JULIAN_DAY)
    max_observations = len(Full_observation_times)
    observation_counts = list(range(max_observations, Minimum_obs, -Observation_step_size))

    # Full_observation_times = np.arange(SSE_start_buffer + time_buffer, SSE_end_buffer - time_buffer_end, constants.JULIAN_DAY)
    # max_observations = len(Full_observation_times)
    # observation_counts = list(range(max_observations, 0, -Observation_step_size))

    # ----------------------------------------------------------------------
    # Perform Estimation for each observation campaign
    # ----------------------------------------------------------------------

    #nth observation sim
    sim = 1
    for n_obs in observation_counts:
        try:
            print(f"\n"\
                f"Estimating with {n_obs} observations")
            # populate the dictionaries
            
            current_times = Full_observation_times[:n_obs]
            observation_simulation_settings_RADEC = observations_setup.observations_simulation_settings.tabulated_simulation_settings(
                observable_models_setup.model_settings.angular_position_type,
                link_definition,
                current_times
            )
            observation_simulation_settings = [observation_simulation_settings_RADEC] 

            # Add noise levels as Gaussian noise to the observation
            LSST = 0.1 # arcsec
            noise_level = (0.1/3600) * np.pi/180  # radians
            observations_setup.random_noise.add_gaussian_noise_to_observable(
                [observation_simulation_settings_RADEC],
                noise_level,
                observable_models_setup.model_settings.angular_position_type
            )

            # ----------------------------------------------------------------------
            # Create Reference Observations
            # ----------------------------------------------------------------------
            propagator_settings_truth = Helper_file.propagator_settings(
                                                                integrator_settings,
                                                                central_bodies,
                                                                acceleration_models_reference,
                                                                bodies_to_propagate,
                                                                initial_state_estimation,
                                                                SSE_start_buffer,
                                                                SSE_end_buffer,
                                                                dependent_variables_to_save)

            parameter_settings_truth = parameters_setup.initial_states(propagator_settings_truth, bodies)
            parameters_to_estimate_truth = parameters_setup.create_parameter_set(
                parameter_settings_truth, bodies, propagator_settings_truth
            )

            truth_estimator = estimation_analysis.Estimator(
                bodies,
                parameters_to_estimate_truth,
                observation_settings_list,
                propagator_settings_truth,
                integrate_on_creation=True
            )

            simulated_observations = observations_setup.observations_wrapper.simulate_observations(
                observation_simulation_settings,
                truth_estimator.observation_simulators,
                bodies
                )

            # ----------------------------------------------------------------------
            # Precise orbit determination
            # ----------------------------------------------------------------------
            "Precise orbit determination of the comets trajectory by weighting the observations, we perurb the initial conditions to better see"
            "the POD can anlayse the dynamical model against the observations."

            # Define weighting of the observations in the inversion
            weights_per_observable = { observations.observations_processing.observation_parser(
                observable_models_setup.model_settings.angular_position_type ): noise_level ** -2}
            
            simulated_observations.set_constant_weight_per_observation_parser(weights_per_observable)

            truth_parameters = parameters_to_estimate.parameter_vector

            perturbed_parameters = truth_parameters.copy()
            for i in range(3):
                perturbed_parameters[i] += 1000000000.0
                perturbed_parameters[i+3] += 100

            parameters_to_estimate.parameter_vector = perturbed_parameters

            pod_input = estimation_analysis.EstimationInput(
                observations_and_times=simulated_observations,
                convergence_checker= estimation_analysis.estimation_convergence_checker(
                    maximum_iterations=number_of_pod_iterations,
                ),
            )

            pod_input.define_estimation_settings(
                reintegrate_variational_equations=True,
                save_state_history_per_iteration=True,    
            )
            
            pod_output = estimator.perform_estimation(pod_input)
        except Exception as e:
            print("\n")
            print(f"Estimation failed: {e}")
            print("Stopping execution.")
            break
        
        if np.any(np.abs(np.array(custom_parameter)) > 1e-5):
            print("\n")
            print(f"A1,A2,A3 is larger than 1e-5, unrealistic value. Stopping simulation.")
            break 

        data_to_write['Montecarlo_trajectory'].setdefault(sim, {})
        data_to_write['Montecarlo_trajectory_times'].setdefault(sim, {})
        data_to_write['observation_times'].setdefault(sim, {})
        data_to_write['Covariance_matrix'].setdefault(sim, {})
        data_to_write['NGA_Est'].setdefault(sim, {})



        times = np.arange(SSE_start, SSE_end, timestep_global)
        if not np.isclose(times[-1], SSE_end, rtol=1e-10, atol=1e-10):
            times = np.append(times, SSE_end)

        ephemeris_estimated = np.zeros((len(times), 6))
        for i, time in enumerate(times):
            ephemeris_estimated[i] = bodies.get(str(spkid)).ephemeris.cartesian_state(time)

        data_to_write["Estimated_Reference_trajectory"][sim] = np.vstack(ephemeris_estimated)
        data_to_write["Estimated_Reference_trajectory_times"][sim] = np.vstack(times)

        print("--------------------------------------------------------------------")
        state_est_start = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)
        print("Estimated start state m & m/s and norm in AU")
        print(np.array(state_est_start))
        print(np.linalg.norm(np.array(state_est_start)[:3])/constants.ASTRONOMICAL_UNIT)

        state_est_end = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_end)
        print("Estimated final state m & m/s and norm in AU")
        print(np.array(state_est_end))
        print(np.linalg.norm(np.array(state_est_end)[:3])/constants.ASTRONOMICAL_UNIT)
        print("--------------------------------------------------------------------"\
            )

        # ----------------------------------------------------------------------
        # Covariance estimation
        # ----------------------------------------------------------------------

        # Create input object for covariance analysis
        covariance_input = estimation_analysis.CovarianceAnalysisInput(
            simulated_observations)

        # Set methodological options
        covariance_input.define_covariance_settings(
            reintegrate_variational_equations=True)

        # Perform the covariance analysis
        covariance_output = estimator.compute_covariance(covariance_input)

        # ----------------------------------------------------------------------
        # Plot observations & estimation 
        # ----------------------------------------------------------------------
        print("--------------------------------------------------------------------")
        print("Plotting observations and estimation")
        obs = ObsPlot(sim,name, simulated_observations, directory_name, addition)
        obs.RADEC_overtime()
        obs.skyplot()
        obs.aitoff()

        est = EstPlot(sim,name, number_of_pod_iterations, pod_output,
                    simulated_observations, covariance_output,
                    parameters_to_estimate, directory_name, addition)
        est.correlation()
        est.residuals(simulated_observations)
        est.formal_erros()

        info_dict_synobs = {
            'Name': name,
            'SPK-id': spkid,

            "Integrator": "RK12",
            "Type": "Fixed",
            "Order": "Higher",

            "Global Timestep": timestep_global,
            "interpolation buffer start(days)": time_buffer/86400,
            "interpolation buffer end(days)": time_buffer_end/86400,

            "Observations": len(current_times),
            "Orbit_clones": Orbit_samples,
            "POD iterations": number_of_pod_iterations,
            "station noise (rad)": noise_level,
            "station noise (arcsec)": LSST,

            "NGAs": {
                "A1": A1,
                "A2": A2,
                "A3": A3
            } if NGA_Flag else None,

            "Reference_state": initial_state_reference,
            "Reference_estimation": initial_state_estimation,

            "NBP": bodies_to_create,
        }
        print(custom_parameter)
        with open(f"{directory_name}{addition}/Info_syntheticobs.txt", "w") as f:
            pprint.pprint(info_dict_synobs, stream=f, indent=2, width=80, sort_dicts=False)
        print(custom_parameter)

        # ----------------------------------------------------------------------
        # Perform monte carlo
        # ----------------------------------------------------------------------
        initial_covariance = covariance_output.covariance
        data_to_write["Covariance_matrix"][sim] = initial_covariance

        initial_state = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)

        NGA_Parameters = custom_parameter
        state_vector = np.concatenate((initial_state,NGA_Parameters))

        trajectory_parameters = state_vector.copy()
        samples = np.random.multivariate_normal(trajectory_parameters, initial_covariance, size=Orbit_samples)
        
        dependent_variables_to_save = []
        subdir = f"{directory_name}/Simulation_{sim}"
        print("--------------------------------------------------------------------")
        print("Performing Monte Carlo")

        for i, sampled_conditions in enumerate(samples):
            acceleration_models_monte = Helper_file.Accelerations(spkid,
                                                bodies, 
                                                bodies_to_propagate, 
                                                central_bodies,
                                                np.array(sampled_conditions[6:]),
                                                NGA_Flag=NGA_Flag)
            
            propagator_settings_sample = Helper_file.propagator_settings(
                                                                integrator_settings,
                                                                central_bodies,
                                                                acceleration_models_monte,
                                                                bodies_to_propagate,
                                                                np.array(sampled_conditions[:6]),
                                                                SSE_start,
                                                                SSE_end,
                                                                dependent_variables_to_save)

            dynamics_simulator_sample = simulator.create_dynamics_simulator(
                            bodies, propagator_settings_sample
                        )

            state_hist_sample = dynamics_simulator_sample.propagation_results.state_history
            data_to_write['Montecarlo_trajectory'][sim][i] = np.vstack(list(state_hist_sample.values()))
            data_to_write['Montecarlo_trajectory_times'][sim][i] = np.vstack(list(state_hist_sample.keys()))
            data_to_write["observation_times"][sim] = current_times
            data_to_write["NGA_Est"][sim] = NGA_Parameters
            data_to_write["NGA_JPL"][sim] = NGA_array
            data_to_write["NGA_Input"][sim] = A_vec


            data_to_write["Sim_info"][sim] = {
            "Orbit_samples": Orbit_samples,
            "N_obs": n_obs,
            "Integrator": "rkf_56",
            "Time_step": timestep_global,
            "Body": target_body
            }

            progress = (i + 1) / len(samples)
            bar_length = 30
            bar = '=' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
            sys.stdout.write(f"\rProgress: [{bar}] {progress * 100:.1f}%")
            sys.stdout.flush()

        data_to_write['environment'] = bodies_to_create

        sim += 1
    
    with open(f"{directory_name}{addition}/Data_NGA_Est.pkl", "wb") as f:
        pickle.dump(data_to_write, f)
    with open(f"Coding/W79 Estimation/Sim_data/Data_NGA_Est_{body}.pkl", "wb") as f:
        pickle.dump(data_to_write, f)

    stat = StatPlot(data_to_write,info_dict_synobs,directory_name,addition)
    stat.plot_3D()
    stat.boxplot()
    stat.fit()
    stat.ref_to_spice()
    stat.clone_divergence()

