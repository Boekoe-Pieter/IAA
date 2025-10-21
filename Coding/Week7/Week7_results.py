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

# python files
import Helper_file
from Plotter import observations_Plotter as obs_plot
from Plotter import estimation_plotter as est_plot
from Plotter import statistics_plotter as stat_plot

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
perform_montecarlo = False
Orbit_samples = 1000

# number of iterations for our estimation
number_of_pod_iterations = 6

# timestep of 1 hours for our estimation
timestep_global = 48*3600

# avoid interpolation errors:
time_buffer = 31*86400 
time_buffer_end = 31*86400 

# define the frame origin and orientation.
global_frame_origin = "Sun"
global_frame_orientation = "ECLIPJ2000"

# NGA on or off for orbit propagation
NGA_Flag = False

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

    "Ceres",
    "Vesta",
    
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

all_results = ["C2001Q4"] #,"C2008A1","C2013US10"]

for body in all_results:
    # ----------------------------------------------------------------------
    # Retrieve comet information from SBDB
    # ----------------------------------------------------------------------
    target_body = body

    comet_designation, comet_time_information, Oscullating_elements, non_gravitational_parameters = Helper_file.sbdb_query_info(
        target_body,
        full_precision=True)
    
    spkid,name,designator = comet_designation

    first_obs,last_obs = comet_time_information

    e,a,q,i,om,w,M,n,Tp = Oscullating_elements

    A1,A2,A3,DT = non_gravitational_parameters

    NGA_array= np.array(
        [A1.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A1 is not None else 0,
        A2.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A2 is not None else 0,
        A3.value*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A3 is not None else 0,
        DT.value if DT is not None else 0,]
        )

    directory_name = f"{rel_dir}/plots_{designator}"

    print(f"SPKid: {spkid} for body {name}")
    print("First obs:", first_obs)
    print("Last obs:", last_obs)

    # ----------------------------------------------------------------------
    # Convert calander dates and JD to Epochs
    # ----------------------------------------------------------------------
    y1, m1, d1 = map(int, first_obs.split('-'))
    y2, m2, d2 = map(int, last_obs.split('-'))

    # Convert to epoch
    SSE_start = time_representation.date_time_components_to_epoch(y1, m1, d1, 0,0,0)
    SSE_end_cal = time_representation.date_time_components_to_epoch(y2, m2, d2, 0,0,0)

    SSE_tp = time_representation.julian_day_to_seconds_since_epoch(float(Tp))

    SSE_end = SSE_tp

    SSE_start_buffer = SSE_start - time_buffer
    SSE_end_buffer = SSE_end + time_buffer_end

    JD_start = time_representation.seconds_since_epoch_to_julian_day(SSE_start_buffer)
    JD_end = time_representation.seconds_since_epoch_to_julian_day(SSE_end_buffer)

    J2000_start = "JD" + str(JD_start)
    J2000_end = "JD" + str(JD_end)

    # ----------------------------------------------------------------------
    # Create SPK file via JPL Horizons & load comet specific kernel
    # ----------------------------------------------------------------------
    Helper_file.Horizons_SPK(Spice_files_path, 
                             spkid, 
                             J2000_start, 
                             J2000_end)
    
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
    initial_state_reference = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_start,
    )

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position(body, "Sun")
        for body in bodies_to_create]
    
    bodies_to_propagate = [str(spkid)]
    
    acceleration_models = Helper_file.Accelerations(spkid,
                                                    bodies, 
                                                    bodies_to_propagate, 
                                                    central_bodies,
                                                    NGA_array,
                                                    NGA_Flag=NGA_Flag)
    
    integrator_settings = Helper_file.integrator_settings(timestep_global,
                                                          variable=False)
    
    propagator_settings = Helper_file.propagator_settings(integrator_settings,
                                                          central_bodies,
                                                          acceleration_models,
                                                          bodies_to_propagate,
                                                          initial_state_reference,
                                                          SSE_start,
                                                          SSE_end,
                                                          dependent_variables_to_save)

    Reference_orbit_simulator = simulator.create_dynamics_simulator(
                        bodies, propagator_settings
                    )
    
    Reference_orbit_results = Reference_orbit_simulator.propagation_results.state_history
    comet_states = np.vstack(list(Reference_orbit_results.values()))
    comet_pos = comet_states[:, :3]/constants.ASTRONOMICAL_UNIT
    
    print(f"propagated final state: {min(np.linalg.norm(comet_pos,axis=1))}")
    print(f'final time {np.vstack(list(Reference_orbit_results.keys()))[-1]}')

    Final_state_spice = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_tp,
    )
    
    print(f"Spice final state: {np.linalg.norm(np.array(Final_state_spice)[:3])/constants.ASTRONOMICAL_UNIT}")
    print(f"difference {(np.linalg.norm(comet_pos[-1])-np.linalg.norm(np.array(Final_state_spice)[:3])/constants.ASTRONOMICAL_UNIT)*constants.ASTRONOMICAL_UNIT/1000}")
    
    print(f'Time difference {np.vstack(list(Reference_orbit_results.keys()))[-1]-SSE_end}')

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
        element_conversion.cartesian_position_type
    )

    # Define the uplink link ends for one-way observable
    link_ends = dict()
    link_ends[observable_models_setup.links.receiver] = observable_models_setup.links.body_reference_point_link_end_id("Earth", "LSST")
    link_ends[observable_models_setup.links.transmitter] = observable_models_setup.links.body_origin_link_end_id(str(spkid))

    # Create observation settings for each link/observable
    link_definition = observable_models_setup.links.LinkDefinition(link_ends)
    observation_settings_list = [observable_models_setup.model_settings.angular_position(link_definition)]

    observation_times = np.arange(SSE_start, SSE_end, constants.JULIAN_DAY)
    observation_simulation_settings_RADEC = observations_setup.observations_simulation_settings.tabulated_simulation_settings(
        observable_models_setup.model_settings.angular_position_type,
        link_definition,
        observation_times
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
    # Define propagator and integrator settings for estimation
    # ----------------------------------------------------------------------
    initial_state_estimation = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_start_buffer,
    )

    dependent_variables_to_save = []
        
    acceleration_models = Helper_file.Accelerations(spkid,
                                                    bodies, 
                                                    bodies_to_propagate, 
                                                    central_bodies,
                                                    NGA_array,
                                                    NGA_Flag=NGA_Flag)
    
    propagator_settings_estimator = Helper_file.propagator_settings(integrator_settings,
                                                          central_bodies,
                                                          acceleration_models,
                                                          bodies_to_propagate,
                                                          initial_state_estimation,
                                                          SSE_start_buffer,
                                                          SSE_end_buffer,
                                                          dependent_variables_to_save)
    
    # ----------------------------------------------------------------------
    # Define parameters to estimate
    # ----------------------------------------------------------------------
    parameter_settings = parameters_setup.initial_states(propagator_settings_estimator, bodies)

    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings_estimator
    )

    estimator = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings_estimator,
        integrate_on_creation=True)

    simulated_observations = observations_setup.observations_wrapper.simulate_observations(
        observation_simulation_settings,
        estimator.observation_simulators,
        bodies)

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
        perturbed_parameters[i] += 100.0
        perturbed_parameters[i+3] += 10

    parameters_to_estimate.parameter_vector = perturbed_parameters

    pod_input = estimation_analysis.EstimationInput(
        observations_and_times=simulated_observations,
        convergence_checker= estimation_analysis.estimation_convergence_checker(
            maximum_iterations=number_of_pod_iterations,
        ),
    )

    pod_input.define_estimation_settings(reintegrate_variational_equations=True)

    pod_output = estimator.perform_estimation(pod_input)
    
    residual_history = pod_output.residual_history
    
    state_est = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)
    print("Estimated start state m & m/s and norm in AU")
    print(np.array(state_est))
    print(np.linalg.norm(np.array(state_est)[:3])/constants.ASTRONOMICAL_UNIT)

    state_est = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_end)
    print("Estimated final state m & m/s and norm in AU")
    print(np.array(state_est))
    print(np.linalg.norm(np.array(state_est)[:3])/constants.ASTRONOMICAL_UNIT)


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
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    addition = "/NGA_True" if NGA_Flag else "/NGA_False"
    try:
        os.mkdir(f"{directory_name}{addition}")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}' already exists.")
    
    obs_plot = obs_plot(body, simulated_observations)

    obs_plot.RADEC_overtime(directory_name=directory_name, addition=addition)
    obs_plot.skyplot(directory_name=directory_name, addition=addition)
    obs_plot.aitoff(directory_name=directory_name, addition=addition)

    est_plot = est_plot(body, number_of_pod_iterations, pod_output,
                                simulated_observations, covariance_output,
                                parameters_to_estimate, estimator.state_transition_interface)
    
    est_plot.residuals(directory_name=directory_name, addition=addition,simulated_observations=simulated_observations)
    est_plot.correlation(directory_name=directory_name, addition=addition)
    est_plot.formal_erros(directory_name=directory_name, addition=addition)

    info_dict_syncobs = {
        'Name': name,
        'SPK-id': spkid,

        "Integrator": "RK1210",
        "Type": "Fixed",
        "Order": "Higher",

        "Global Timestep": timestep_global,
        "interpolation buffer start(days)": time_buffer/86400,
        "interpolation buffer end(days)": time_buffer_end/86400,

        "Observations": len(observation_times),
        "POD iterations": number_of_pod_iterations,
        "station noise (rad)": noise_level,
        "station noise (arcsec)": LSST,

        "NGAs": {
            "A1 [m/s^2]": A1,
            "A2 [m/s^2]": A2,
            "A3 [m/s^2]": A3
        } if NGA_Flag else None,

        "Reference_state": initial_state_reference,
        "Reference_estimation": initial_state_estimation,

        "NBP": bodies_to_create,
    }

    with open(f"{directory_name}{addition}/Info_syntheticobs.txt", "w") as f:
        pprint.pprint(info_dict_syncobs, stream=f, indent=2, width=80, sort_dicts=False)
    
    # ----------------------------------------------------------------------
    # Perform monte carlo
    # ----------------------------------------------------------------------
    initial_covariance = covariance_output.covariance
    initial_state = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)

    np.random.seed(42)

    trajectory_parameters = initial_state.copy()
    samples = np.random.multivariate_normal(trajectory_parameters, initial_covariance, size=Orbit_samples)
    
    dependent_variables_to_save = []

    for i, sampled_conditions in enumerate(samples):
        propagator_settings_sample = Helper_file.propagator_settings(
                                                            integrator_settings,
                                                            central_bodies,
                                                            acceleration_models,
                                                            bodies_to_propagate,
                                                            sampled_conditions,
                                                            SSE_start,
                                                            SSE_end,
                                                            dependent_variables_to_save)

        dynamics_simulator_sample = simulator.create_dynamics_simulator(
                        bodies, propagator_settings_sample
                    )

        state_hist_sample = dynamics_simulator_sample.propagation_results.state_history



#     # ----------------------------------------------------------------------
#     # Saving information dict
#     info_dict_syncobs = {
#         'Name': target_sbdb['object']['fullname'],
#         'SPK-id': target_sbdb['object']['spkid'],
#         'Type': target_sbdb['object']['orbit_class'].get("code"),

#         "integrator": "variable RK78",
#         "Global Timestep": timestep_global,
#         "interpolation buffer start(days)": time_buffer/86400,
#         "interpolation buffer end(days)": time_buffer_end/86400,

#         "Observations": len(ra_deg),
#         "POD iterations": number_of_pod_iterations,
#         "station noise (rad)": noise_level,
#         "station noise (arcsec)": LSST,

#         "NGAs": {
#             "A1 [m/s^2]": A1,
#             "A2 [m/s^2]": A2,
#             "A3 [m/s^2]": A3
#         },

#         "Initial conditions": {       
#             'First_obs': target_sbdb['orbit']["first_obs"],
#             "last_obs": target_sbdb['orbit']["last_obs"],
#             "tp": target_sbdb["orbit"]["elements"]["tp"],
#             "q": target_sbdb["orbit"]["elements"]["q"],
#             "e": target_sbdb["orbit"]["elements"]["e"],
#             "w": target_sbdb["orbit"]["elements"]["w"],
#             "om": target_sbdb["orbit"]["elements"]["om"],
#             "i": target_sbdb["orbit"]["elements"]["i"],
#             "a": target_sbdb["orbit"]["elements"]["a"],
#         },

#         "NBP": bodies_to_create,
#     }

#     with open(f"{directory_name}{addition}/Info_syntheticobs.txt", "w") as f:
#         pprint.pprint(info_dict_syncobs, stream=f, indent=2, width=80, sort_dicts=False)


#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------
#     # ----------------------------------------------------------------------

#     "this part of the code is for the covariance sampling and propagation"
#     if perform_montecarlo:
#         # create directory
#         addition = '/Montecarlo'
#         try:
#             os.mkdir(f"{directory_name}{addition}")
#             print(f"Directory '{directory_name}{addition}' created successfully.")
#         except FileExistsError:
#             print(f"Directory '{directory_name}{addition}' already exists.")

#         initial_covariance = covariance_output.covariance                                   # covariance from observations
#         initial_state = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)         # Estimated initial state from sythetic observations

#         np.random.seed(42)                                                                  # the answer to everything
#         trajectory_parameters = initial_state.copy()
#         samples = np.random.multivariate_normal(trajectory_parameters, initial_covariance, size=Orbit_samples)
        
#         integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
#             time_step = timestep_global,
#             coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78,
#             order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )           # RK8 Fixed step integrator, to prevent interpolation
        
#         termination_condition = propagation_setup.propagator.time_termination(SSE_end)

#         def NGA(time: float) -> np.ndarray:
#             state = bodies.get(str(spkid)).state
#             r_vec = state[:3]
#             v_vec = state[3:]

#             r_norm = np.linalg.norm(r_vec)

#             m = 2.15
#             n = 5.093
#             k = 4.6142
#             r0 = 2.808*constants.ASTRONOMICAL_UNIT
#             alpha = 0.1113

#             A1, A2, A3 = param_dict['A1'],param_dict['A2'],param_dict['A3']
#             A_vec = np.array([A1, A2, A3])

#             g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)

#             C_rtn2eci = rtn_to_eci(r_vec, v_vec)

#             F_vec_rtn = g * A_vec
#             F_vec_inertial = C_rtn2eci @ F_vec_rtn

#             return F_vec_inertial  

#         def rtn_to_eci(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
#             r_hat = r_vec / np.linalg.norm(r_vec)
#             h_vec = np.cross(r_vec, v_vec)
#             h_hat = h_vec / np.linalg.norm(h_vec)
#             t_hat = np.cross(h_hat, r_hat)

#             C = np.vstack((r_hat, t_hat, h_hat)).T
#             return C


#         # Full model with NGA
#         NGAaccelerations = {
#             "Sun": [
#                 propagation_setup.acceleration.point_mass_gravity(),
#                 propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
#             ],
#             str(spkid): [propagation_setup.acceleration.custom_acceleration(NGA)]
#         }

#         # Gravity-only model
#         Gravityaccelerations = {
#             "Sun": [
#                 propagation_setup.acceleration.point_mass_gravity(),
#             ]
#         }

#         models = [Gravityaccelerations] #, NGAaccelerations]

#         data_to_write = {
#             "Nominal_trajectory":0,
#             "Nominal_trajectory_times":0,
#             "Monte_trajectory": {},
#             "Monte_trajectory_times": {},
#             "Initial_condition": 0,
#             "Sampled_data": 0,
#             "CPU_time_list": 0
#         }

#         for model in models:
#             acceleration_settings[str(spkid)] = model
#             acceleration_models = propagation_setup.create_acceleration_models(
#                     bodies, acceleration_settings, bodies_to_propagate, central_bodies
#                 )
            
#             for sample in samples:
#                 propagator_settings = propagation_setup.propagator.translational(
#                     central_bodies=central_bodies,
#                     acceleration_models=acceleration_models,
#                     bodies_to_integrate=bodies_to_propagate,
#                     initial_states=sample,
#                     initial_time=SSE_start,
#                     integrator_settings=integrator_settings,
#                     termination_settings=termination_condition,
#                 )

#                 dynamics_simulator_montecarlo = simulator.create_dynamics_simulator(
#                     bodies, propagator_settings
#                 )

#                 states = dynamics_simulator_montecarlo.propagation_results.state_history
                

    