# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, environment, propagation, parameters_setup, simulator
from tudatpy.astro import time_representation, element_conversion
from tudatpy.estimation import observations, observable_models, observations_setup, observable_models_setup, estimation_analysis
from tudatpy import constants

# import SBDB interface
from tudatpy.data.sbdb import SBDBquery

# other useful modules
import numpy as np
import random
import pprint

# other libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import spiceypy as spicepy
import os

# Horizons requests
import sys
import json
import base64
import requests

# ----------------------------------------------------------------------
# Define saving directories
# ----------------------------------------------------------------------
sys.path.append('/Users/pieter/IAA/Coding')
Spice_files = '/Users/pieter/IAA/Coding/Spice_files/'
rel_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Define iterations, timesteps and frames
# ----------------------------------------------------------------------

# samples
perform_montecarlo = False
Orbit_samples = 1000

# number of iterations for our estimation
number_of_pod_iterations = 6

# timestep of 1 hours for our estimation
timestep_global = 1*3600 

# avoid interpolation errors:
time_buffer = 1*86400 
time_buffer_end = 7*86400 

# define the frame origin and orientation.
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"

# ----------------------------------------------------------------------
# Retrieve data from SBDB
# ----------------------------------------------------------------------
url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

classes = ["HYP","PAR","COM"]
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
    target_mpc_code = body
    target_sbdb = SBDBquery(target_mpc_code,full_precision=True)

    spkid = target_sbdb['object']['spkid']
    name  = target_sbdb['object']['fullname']
    designator = target_sbdb['object']['des']
    directory_name = f"{rel_dir}/plots_{designator}"

    print(f"SPKid: {spkid} for body {name}")

    first_obs = target_sbdb['orbit']['first_obs']
    last_obs  = target_sbdb['orbit']['last_obs']

    q_time = target_sbdb["orbit"]['elements']['tp'].value

    print("First obs:", first_obs)
    print("Last obs:", last_obs)

    y1, m1, d1 = map(int, first_obs.split('-'))
    y2, m2, d2 = map(int, last_obs.split('-'))

    # Convert to epoch
    SSE_start = time_representation.date_time_components_to_epoch(y1, m1, d1, 0,0,0)
    SSE_end_cal = time_representation.date_time_components_to_epoch(y2, m2, d2, 0,0,0)
    SSE_end = time_representation.TDB_to_TT(q_time,[0,0,0])
    SSE_end = time_representation.julian_day_to_seconds_since_epoch(SSE_end)

    SSE_start_buffer = SSE_start - time_buffer
    SSE_end_buffer = SSE_end + time_buffer_end

    Epoch_start = time_representation.seconds_since_epoch_to_julian_day(SSE_start_buffer)
    Epoch_end = time_representation.seconds_since_epoch_to_julian_day(SSE_end_buffer)

    J2000_start = "JD" + str(time_representation.seconds_since_epoch_to_julian_day(SSE_start_buffer))
    J2000_end = "JD" + str(SSE_end_buffer) #time_representation.seconds_since_epoch_to_julian_day(SSE_end_buffer))


    # NGA DATA
    A1 = target_sbdb["orbit"]["model_pars"].get("A1")
    A2 = target_sbdb["orbit"]["model_pars"].get("A2")
    A3 = target_sbdb["orbit"]["model_pars"].get("A3")
    DT = target_sbdb["orbit"]["model_pars"].get("DT") 

    A1, A2, A3, DT = (
        A1.value if A1 is not None else 0,
        A2.value if A2 is not None else 0,
        A3.value if A3 is not None else 0,
        DT.value if DT is not None else 0,
        )

    param_dict = {
        "A1": A1*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2,
        "A2": A2*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2,
        "A3": A3*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2,
        "DT": DT*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2,
    }

    # ----------------------------------------------------------------------
    # Create SPK file via JPL Horizons from the observations of
    # ----------------------------------------------------------------------
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
    url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, J2000_start, J2000_end)

    response = requests.get(url)
    Horizons = json.loads(response.text)

    # If the request was valid...
    if (response.status_code == 200):
        #
        # If the SPK file was generated, decode it and write it to the output file:
        if "spk" in Horizons:
            #
            # If a suggested SPK file basename was provided, use it:
            if "spk_file_id" in Horizons:
                spk_filename = Spice_files + Horizons["spk_file_id"] + ".bsp"
            try:
                f = open(spk_filename, "wb")
            except OSError as err:
                print("Unable to open SPK file '{0}': {1}".format(spk_filename, err))
            #
            # Decode and write the binary SPK file content:
            f.write(base64.b64decode(Horizons["spk"]))
            f.close()
            print("wrote SPK content to {0}".format(spk_filename))
    
    spice.clear_kernels()
    spice.load_standard_kernels()
    spice.load_kernel(f"Coding/Spice_files/{spkid}.bsp")

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

    body_settings.add_empty_settings(str(spkid))
    # body_settings.get(str(spkid)).ephemeris_settings = environment_setup.ephemeris.direct_spice(
    #     global_frame_origin,
    #     global_frame_orientation,
    #     str(spkid)
    # )

    bodies = environment_setup.create_system_of_bodies(body_settings)

    central_bodies = [global_frame_origin]

    # ----------------------------------------------------------------------
    # Define accelerations on the body of interest
    # ----------------------------------------------------------------------
    "Creating the reference orbit will have all accelerations acting on the comet,"
    "note: the NGA's are JPL fitted THESE CANT BE REUSED WHEN ESTIMATING THE ORBIT"

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

        A1, A2, A3 = param_dict['A1'],param_dict['A2'],param_dict['A3']
        A_vec = np.array([A1, A2, A3])

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

        "Mercury": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Venus": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Earth": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Moon": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Mars": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Phobos": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Deimos": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Jupiter": [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
        ],
        "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
        "Europa": [propagation_setup.acceleration.point_mass_gravity()],
        "Callisto": [propagation_setup.acceleration.point_mass_gravity()],
        "Io": [propagation_setup.acceleration.point_mass_gravity()],

        "Saturn": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Enceladus": [propagation_setup.acceleration.point_mass_gravity()],
        "Titan": [propagation_setup.acceleration.point_mass_gravity()],

        "Uranus": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Neptune": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        
        str(spkid): [propagation_setup.acceleration.custom_acceleration(NGA)]
    }

    bodies_to_propagate = [str(spkid)]
    acceleration_settings = {}
    acceleration_settings[str(spkid)] = accelerations

    # create the acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = timestep_global,
            coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_89,
            order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )  

    termination_condition = propagation_setup.propagator.time_termination(SSE_end_buffer)

    initial_state = spice.get_body_cartesian_state_at_epoch(
        str(spkid),
        global_frame_origin,
        global_frame_orientation,
        "NONE",
        SSE_start_buffer,
    )

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Mercury", "Sun"),
        propagation_setup.dependent_variable.relative_position("Venus", "Sun"),
        propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
        propagation_setup.dependent_variable.relative_position("Mars", "Sun"),
        propagation_setup.dependent_variable.relative_position("Jupiter", "Sun"),
        propagation_setup.dependent_variable.relative_position("Saturn", "Sun"),
        propagation_setup.dependent_variable.relative_position("Uranus", "Sun"),
        propagation_setup.dependent_variable.relative_position("Neptune", "Sun"),
        propagation_setup.dependent_variable.keplerian_state(str(spkid), "Sun"),
        propagation_setup.dependent_variable.central_body_fixed_cartesian_position(str(spkid), "Sun"),
        propagation_setup.dependent_variable.relative_position(str(spkid), "Sun"),
        propagation_setup.dependent_variable.relative_velocity(str(spkid), "Sun"),                                
        ]

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=SSE_start_buffer,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save,
    )

    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )

    propagated_state_history = dynamics_simulator.state_history

    body_ephemeris = environment_setup.ephemeris.tabulated(
        propagated_state_history,
        global_frame_origin,
        global_frame_orientation
    )

    body_settings.get(str(spkid)).ephemeris_settings = body_ephemeris

    # ----------------------------------------------------------------------
    # Define Observatory
    # ----------------------------------------------------------------------
    "We define the LSST as our observatory, and is located in the center of the earth as we are not interested in the effect of one tellescope"
    "but the effect of N observations. Sim time is one observation per day as per LSST and noise is added as per LSST (source from Pedro Lacerda in ESTEC)"
    "We remove the timebuffer from the observation times as we are only interested in the actual time frame"
    "TODO: variable for the number of observations"
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

    observation_times = np.arange(SSE_start_buffer + time_buffer, SSE_end_buffer - time_buffer_end, constants.JULIAN_DAY)
    
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
    # Define accelerations on the body of interest
    # ----------------------------------------------------------------------
    "We re-define the accelerations on the comet, and redefine the NGA defenition as we now want to estimate the A1,A2,A3 parameters"
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

        A_vec = custom_parameter

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

        "Mercury": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        
        "Venus": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        
        "Earth": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Moon": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Mars": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Phobos": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Deimos": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        "Jupiter": [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
        ],
        "Ganymede": [
            propagation_setup.acceleration.point_mass_gravity()],
        "Europa": [
            propagation_setup.acceleration.point_mass_gravity()],
        "Callisto": [
            propagation_setup.acceleration.point_mass_gravity()],
        "Io": [
            propagation_setup.acceleration.point_mass_gravity()],

        "Saturn": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Enceladus": [
            propagation_setup.acceleration.point_mass_gravity()],
        "Titan": [
            propagation_setup.acceleration.point_mass_gravity()],

        "Uranus": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],
        "Neptune": [
            propagation_setup.acceleration.point_mass_gravity(),
        ],

        str(spkid): [
            propagation_setup.acceleration.custom_acceleration(NGA)] 
        }

    bodies_to_propagate = [str(spkid)]
    acceleration_settings = {}
    acceleration_settings[str(spkid)] = accelerations
    
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.central_body_fixed_cartesian_position(str(spkid), "Sun"),
        propagation_setup.dependent_variable.relative_position(str(spkid), "Sun"),
        propagation_setup.dependent_variable.relative_velocity(str(spkid), "Sun"),
        ]
    
    # create the acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=SSE_start,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save,
    )

    # ----------------------------------------------------------------------
    # Define the parameters to be estimated
    # ----------------------------------------------------------------------
    "Here we define the parameters to be estimated,the initial state and NGAs A1,A2,A3"
    def compute_current_custom_parameter_partial() -> np.ndarray:
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
    
    custom_parameter = np.array([A1, A2, A3])  # JPL as Guess

    def get_custom_parameter():
        return custom_parameter #Get the parameter values

    def set_custom_parameter(estimated_value):
        global custom_parameter
        custom_parameter = np.array(estimated_value)  #Update the guess

    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    
    parameter_settings.append(
        parameters_setup.custom_parameter(
            'NGA', 3, get_custom_parameter,
            set_custom_parameter
        )
    )

    parameter_settings[-1].custom_partial_settings = [
        parameters_setup.custom_analytical_partial(
            compute_current_custom_parameter_partial(),
            str(spkid), str(spkid),
            propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
        )
    ]

    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    estimator = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings,
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
        perturbed_parameters[i] += 1000.0
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

    # ----------------------------------------------------------------------
    # Covariance estimation
    # ----------------------------------------------------------------------
    "Create the covariance matrix at the initial position"

    # Create input object for covariance analysis
    covariance_input = estimation_analysis.CovarianceAnalysisInput(
        simulated_observations)

    # Set methodological options
    covariance_input.define_covariance_settings(
        reintegrate_variational_equations=True)

    # Perform the covariance analysis
    covariance_output = estimator.compute_covariance(covariance_input)

    # Print the covariance matrix
    formal_errors = covariance_output.formal_errors
    print(f'Formal Errors:\n{formal_errors}')

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    " This part of the code plots all the observations"

    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    addition = '/Observations'
    try:
        os.mkdir(f"{directory_name}{addition}")
        print(f"Directory '{directory_name}{addition}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}{addition}' already exists.")
    
    # observatin data
    Epochs = simulated_observations.get_observation_times()
    RA = simulated_observations.get_observations()[0::2]
    DEC = simulated_observations.get_observations()[1::2]

    # ----------------------------------------------------------------------
    # RA/DEC over time
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    for epochs, obs in zip(simulated_observations.get_observation_times(),
                        simulated_observations.get_observations()):
        ra_deg = np.rad2deg(obs[0::2])
        time_days = (np.array(epochs) - epochs[0]) / (3600.0*24)
        plt.scatter(time_days, ra_deg, s=2, label="Batch")
    
    plt.xlabel("Time [days]")
    plt.ylabel("Right Ascension [deg]")
    plt.grid(True)
    plt.title("Simulated RA")

    plt.subplot(1,2,2)
    for epochs, obs in zip(simulated_observations.get_observation_times(),
                        simulated_observations.get_observations()):
        dec_deg = np.rad2deg(obs[1::2])
        time_days = (np.array(epochs) - epochs[0]) / (3600.0*24)
        plt.scatter(time_days, dec_deg, s=2, label="Batch")
    
    plt.xlabel("Time [days]")
    plt.ylabel("Declination [deg]")
    plt.grid(True)
    plt.title("Simulated Dec")

    plt.tight_layout()
    plt.savefig(f"{directory_name}{addition}/RADEC_time.pdf", dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # RA/DEC Skyplot
    plt.figure(figsize=(7,7))
    sc = plt.scatter(ra_deg, dec_deg, c=time_days, cmap='viridis', s=2)
    plt.xlabel("Right Ascension [deg]")
    plt.ylabel("Declination [deg]")
    plt.title(f"Sky Track, SKPID:{spkid},Nobs: {len(ra_deg)}")
    plt.gca().invert_xaxis()  
    plt.grid(True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Time [days]")
    plt.savefig(f"{directory_name}{addition}/RADEC_skyplot.pdf", dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Aitoff Projection
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    plt.figure(figsize=(10,5))
    ax = plt.subplot(111, projection='aitoff')

    sc = ax.scatter(ra_rad, dec_rad, c=time_days, cmap='viridis', s=5)

    ax.set_title("Comet Sky Track", va='bottom')
    plt.grid(True)

    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label("Time [days]")

    plt.tight_layout()
    plt.savefig(f"{directory_name}{addition}/RADEC_Aitoff.pdf", dpi=300)

    # ----------------------------------------------------------------------
    # 3D Plot
    dep_hist = dynamics_simulator.propagation_results.dependent_variable_history

    epochs = np.array(list(dep_hist.keys()))
    dep_vals = np.vstack(list(dep_hist.values()))/constants.ASTRONOMICAL_UNIT

    mercury_pos = dep_vals[:, 0:3]
    venus_pos   = dep_vals[:, 3:6]
    earth_pos   = dep_vals[:, 6:9]
    mars_pos    = dep_vals[:, 9:12]
    jupiter_pos = dep_vals[:, 12:15]
    saturn_pos  = dep_vals[:, 15:18]
    uranus_pos  = dep_vals[:, 18:21]
    neptune_pos = dep_vals[:, 21:24]

    state_hist = dynamics_simulator.propagation_results.state_history
    comet_states = np.vstack(list(state_hist.values()))
    comet_pos = comet_states[:, :3]/constants.ASTRONOMICAL_UNIT

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(mercury_pos[:,0], mercury_pos[:,1], mercury_pos[:,2], label="Mercury")
    ax.plot(venus_pos[:,0],   venus_pos[:,1],   venus_pos[:,2],   label="Venus")
    ax.plot(earth_pos[:,0],   earth_pos[:,1],   earth_pos[:,2],   label="Earth")
    ax.plot(mars_pos[:,0],    mars_pos[:,1],    mars_pos[:,2],    label="Mars")
    ax.plot(jupiter_pos[:,0], jupiter_pos[:,1], jupiter_pos[:,2], label="Jupiter")
    ax.plot(saturn_pos[:,0],  saturn_pos[:,1],  saturn_pos[:,2],  label="Saturn")
    ax.plot(uranus_pos[:,0],  uranus_pos[:,1],  uranus_pos[:,2],  label="Uranus")
    ax.plot(neptune_pos[:,0], neptune_pos[:,1], neptune_pos[:,2], label="Neptune")

    ax.plot(comet_pos[:,0], comet_pos[:,1], comet_pos[:,2], "k-", linewidth=1, label=f"{name}")
    ax.scatter(comet_pos[0,0], comet_pos[0,1], comet_pos[0,2], color="green", marker="o", s=10)
    ax.scatter(comet_pos[-1,0], comet_pos[-1,1], comet_pos[-1,2], color="red", marker="x", s=10)

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.legend()

    max_range = np.array([comet_pos[:,0].max()-comet_pos[:,0].min(),
                        comet_pos[:,1].max()-comet_pos[:,1].min(),
                        comet_pos[:,2].max()-comet_pos[:,2].min()]).max() / 2.0
    mid_x = (comet_pos[:,0].max()+comet_pos[:,0].min()) * 0.5
    mid_y = (comet_pos[:,1].max()+comet_pos[:,1].min()) * 0.5
    mid_z = (comet_pos[:,2].max()+comet_pos[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.savefig(f"{directory_name}{addition}/3D_plot.pdf", dpi=300)
    # plt.show()
    plt.close()

    # ----------------------------------------------------------------------
    # Plotting of the residuals

    # Number of columns and rows for our plot
    number_of_columns = 2

    number_of_rows = (
        int(number_of_pod_iterations / number_of_columns)
        if number_of_pod_iterations % number_of_columns == 0
        else int((number_of_pod_iterations + 1) / number_of_columns)
    )

    fig, axs = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(9, 3.5 * number_of_rows),
        sharex=True,
        sharey=False,
    )

    # We cheat a little to get an approximate year out of our times (which are in seconds since J2000)
    residual_times = (
        np.array(simulated_observations.concatenated_times) / (86400 * 365.25) + 2000
    )

    # plot the residuals, split between RA and DEC types
    for idx, ax in enumerate(fig.get_axes()):
        ax.grid()
        # we take every second
        ax.scatter(
            residual_times[::2],
            residual_history[
                ::2,
                idx,
            ],
            marker="+",
            s=60,
            label="Right Ascension",
        )
        ax.scatter(
            residual_times[1::2],
            residual_history[
                1::2,
                idx,
            ],
            marker="+",
            s=60,
            label="Declination",
        )
        ax.set_ylabel("Observation Residual [rad]")
        ax.set_title("Iteration " + str(idx + 1))

    plt.tight_layout()

    # add the year label for the x-axis
    for col in range(number_of_columns):
        axs[int(number_of_rows - 1), col].set_xlabel("Year")

    axs[0, 0].legend()
    plt.savefig(f"{directory_name}{addition}/Residuals.pdf", dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # correlation plot
    body = name
    corr_matrix = covariance_output.correlations

    covar_names = ["x", 'y', 'z', 'vx', 'vy', 'vz'] #, 'A1', 'A2', 'A3']

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(covar_names)), labels=covar_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(covar_names)), labels=covar_names)

    for i in range(len(covar_names)):
        for j in range(len(covar_names)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", color="w", fontsize=8)

    cb = plt.colorbar(im)
    cb.set_label("Correlation coefficient")

    ax.set_xlabel("Estimated Parameter")
    ax.set_ylabel("Estimated Parameter")
    fig.suptitle(f"Correlation matrix for estimated parameters of {body}")

    fig.tight_layout()
    plt.savefig(f"{directory_name}{addition}/Corr_matrix.pdf", dpi=300)
    plt.close()


    # ----------------------------------------------------------------------
    # correlation plot final
    initial_covariance = covariance_output.covariance    
    final_covariance = estimation_analysis.propagate_covariance(initial_covariance,estimator.state_transition_interface,[SSE_end])

    # Extract the covariance matrix
    cov_matrix = list(final_covariance.values())[0]

    # Compute correlation matrix
    D = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(D, D)

    covar_names = ["x", "y", "z", "vx", "vy", "vz"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(covar_names)), labels=covar_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(covar_names)), labels=covar_names)

    for i in range(len(covar_names)):
        for j in range(len(covar_names)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", color="w", fontsize=8)

    cb = plt.colorbar(im)
    cb.set_label("Correlation coefficient")

    ax.set_xlabel("Estimated Parameter")
    ax.set_ylabel("Estimated Parameter")
    fig.suptitle("Correlation matrix for estimated parameters")

    fig.tight_layout()
    plt.close()

    # ----------------------------------------------------------------------
    # Formal Errors and Covariance Matrix
    x_star = parameters_to_estimate.parameter_vector # Nominal solution (center of the ellipsoid)
    # Create input object for covariance analysis
    covariance_input = estimation_analysis.CovarianceAnalysisInput(
        simulated_observations)

    # # Set methodological options
    covariance_input.define_covariance_settings(
        reintegrate_variational_equations=True)
    covariance_output = estimator.compute_covariance(covariance_input)
    initial_covariance = covariance_output.covariance  # Covariance matrix
    print(f'Initial_covariance:\n\n{initial_covariance}\n')

    state_transition_interface = estimator.state_transition_interface
    output_times = observation_times

    diagonal_covariance = np.diag(formal_errors**2)
    print(f'Formal Error Matrix:\n\n{diagonal_covariance}\n')

    sigma = 3  # Confidence level
    original_eigenvalues, original_eigenvectors = np.linalg.eig(diagonal_covariance)
    original_diagonal_eigenvalues, original_diagonal_eigenvectors = np.linalg.eig(diagonal_covariance)
    print(f'Estimated state and parameters:\n\n {parameters_to_estimate.parameter_vector}\n')
    print(f'Eigenvalues of Covariance Matrix:\n\n {original_eigenvalues}\n')
    print(f'Eigenvalues of Formal Errors Matrix:\n\n {original_diagonal_eigenvalues}\n')

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(original_eigenvalues)[::-1]
    diagonal_sorted_indices = np.argsort(original_diagonal_eigenvalues)[::-1]

    eigenvalues = original_eigenvalues[sorted_indices]
    eigenvectors = original_eigenvectors[:, sorted_indices]

    diagonal_eigenvalues = original_diagonal_eigenvalues[diagonal_sorted_indices]
    diagonal_eigenvectors = original_diagonal_eigenvectors[:, diagonal_sorted_indices]

    # Output results
    print(f"Sorted Eigenvalues (variances along principal axes):\n\n{eigenvalues}\n")
    print(f"Sorted Formal Error Matrix Eigenvalues (variances along principal axes):\n\n{diagonal_eigenvalues}\n")
    print(f"Sorted Eigenvectors (directions of principal axes):\n\n{eigenvectors}\n")
    print(f"Sorted Formal Error Matrix Eigenvectors (directions of principal axes):\n\n{diagonal_eigenvectors}\n")

    COV_sub = initial_covariance[np.ix_(np.sort(sorted_indices)[:3], np.sort(sorted_indices)[:3])]  #Covariance restriction to first 3 (spatial) eigenvectors
    diagonal_COV_sub = diagonal_covariance[np.ix_(np.sort(diagonal_sorted_indices)[:3], np.sort(diagonal_sorted_indices)[:3])]  #Covariance restriction to first 3 (spatial) eigenvectors

    x_star_sub = x_star[sorted_indices[:3]] #Nominal solution subset
    diagonal_x_star_sub = x_star[diagonal_sorted_indices[:3]] #Nominal solution subset

    # Eigenvalue decomposition of the submatrix
    eigenvalues, eigenvectors = np.linalg.eig(COV_sub)
    diagonal_eigenvalues, diagonal_eigenvectors = np.linalg.eig(diagonal_COV_sub)

    # Ensure eigenvalues are positive
    if np.any(eigenvalues <= 0):
        raise ValueError(f"$Covariance$ submatrix is not positive definite. Eigenvalues must be positive.\n")
    if np.any(diagonal_eigenvalues <= 0):
        raise ValueError(f"$Formal Errors$ submatrix is not positive definite. Eigenvalues must be positive.\n")

    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi,50)
    phi, theta = np.meshgrid(phi, theta)

    # Generate points on the unit sphere and multiply each direction by the corresponding eigenvalue
    x_ell= np.sqrt(eigenvalues[0])*  np.sin(phi) * np.cos(theta)
    y_ell = np.sqrt(eigenvalues[1])* np.sin(phi) * np.sin(theta)
    z_ell = np.sqrt(eigenvalues[2])* np.cos(phi)

    # Generate points on the unit sphere and multiply each direction by the corresponding diagonal_eigenvalue
    diagonal_x_ell = np.sqrt(diagonal_eigenvalues[0])*np.sin(phi) * np.cos(theta)
    diagonal_y_ell = np.sqrt(diagonal_eigenvalues[1])*np.sin(phi) * np.sin(theta)
    diagonal_z_ell = np.sqrt(diagonal_eigenvalues[2])*np.cos(phi)

    ell = np.stack([x_ell, y_ell, z_ell], axis=0)
    diagonal_ell = np.stack([diagonal_x_ell, diagonal_y_ell, diagonal_z_ell], axis=0)

    #Rotate the Ellipsoid(s). This is done by multiplying ell and diagonal_ell by the corresponding eigenvector matrices
    ellipsoid_boundary_3_sigma = 3 * np.tensordot(eigenvectors, ell, axes=1)
    ellipsoid_boundary_1_sigma = 1 * np.tensordot(eigenvectors, ell, axes=1)
    diagonal_ellipsoid_boundary_3_sigma = 3 * np.tensordot(diagonal_eigenvectors, diagonal_ell, axes=1)
    diagonal_ellipsoid_boundary_1_sigma = 1 * np.tensordot(diagonal_eigenvectors, diagonal_ell, axes=1)

    # Plot the ellipsoid in 3D
    fig = plt.figure(figsize=(15, 8))
    fig.tight_layout()

    ax = fig.add_subplot(121, projection='3d')
    diagonal_ax =fig.add_subplot(122, projection='3d')

    ax.plot_surface(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], color='cyan', alpha=0.4, label = '3-sigma (covariance)')
    ax.plot_surface(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], color='blue', alpha=0.4, label = '1-sigma (covariance)')

    diagonal_ax.plot_surface(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[1], diagonal_ellipsoid_boundary_3_sigma[2], color='red', alpha=0.2, label = '3-sigma (formal errors)')
    diagonal_ax.plot_surface(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[1], diagonal_ellipsoid_boundary_1_sigma[2], color='black', alpha=0.2, label = '1-sigma (formal errors)')

    ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[2], 'r+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
    ax.plot(ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], 'r+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
    ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], 'r+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))

    ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[2], 'b+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
    ax.plot(ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], 'b+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
    ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], 'b+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))

    diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[2], 'r+', alpha=0.1, zdir='y', zs=2*np.max(diagonal_ellipsoid_boundary_3_sigma[1]))
    diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[1], diagonal_ellipsoid_boundary_1_sigma[2], 'r+',alpha=0.1, zdir='x', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[0]))
    diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[1], 'r+',alpha=0.1, zdir='z', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[2]))

    diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[2], 'b+', alpha=0.1, zdir='y', zs=2*np.max(diagonal_ellipsoid_boundary_3_sigma[1]))
    diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[1], diagonal_ellipsoid_boundary_3_sigma[2], 'b+',alpha=0.1, zdir='x', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[0]))
    diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[1], 'b+',alpha=0.1, zdir='z', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[2]))
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max([x_range, y_range, z_range])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    ax.set_xlabel(r'$(x-x^*)$')
    ax.set_ylabel(r'$(y-y^*)$')
    ax.set_zlabel(r'$(z-z^*)$')
    ax.set_title('3D Confidence Ellipsoid and Projections')
    ax.legend(loc = 'upper right')

    diagonal_ax.set_xlabel(r'$(x-x^*)$')
    diagonal_ax.set_ylabel(r'$(y-y^*)$')
    diagonal_ax.set_zlabel(r'$(z-z^*)$')
    diagonal_ax.set_title('Formal Errors and Projections')
    diagonal_ax.legend(loc = 'upper right')


    plt.legend()
    plt.savefig(f"{directory_name}{addition}/Confidence.pdf", dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Plot difference to spice
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    spice_states = []
    estimation_states = []

    # retrieve the states for a list of times.
    times = np.linspace(SSE_start, SSE_end, 1000)
    times_plot = times / (86400 * 365.25) + 2000  # approximate
    for time in times:
        # from spice
        state_spice = spice.get_body_cartesian_state_at_epoch(
            str(spkid), central_bodies[0], global_frame_orientation, "NONE", time
        )
        spice_states.append(state_spice)

        # from estimation
        state_est = bodies.get(str(spkid)).ephemeris.cartesian_state(time)
        estimation_states.append(state_est)

        # Error in kilometers
        error = (np.array(spice_states) - np.array(estimation_states)) / 1000

    # plot
    ax.plot(times_plot, error[:, 0], label="x")
    ax.plot(times_plot, error[:, 1], label="y")
    ax.plot(times_plot, error[:, 2], label="z")

    ax.grid()
    ax.legend(ncol=1)

    plt.tight_layout()

    ax.set_ylabel("Cartesian Error [km]")
    ax.set_xlabel("Year")

    fig.suptitle(f"Error vs SPICE over time for {name}")
    fig.set_tight_layout(True)
    plt.savefig(f"{directory_name}{addition}/Error_to_spice.pdf", dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # Saving information dict
    info_dict_syncobs = {
        'Name': target_sbdb['object']['fullname'],
        'SPK-id': target_sbdb['object']['spkid'],
        'Type': target_sbdb['object']['orbit_class'].get("code"),

        "integrator": "variable RK78",
        "Global Timestep": timestep_global,
        "interpolation buffer start(days)": time_buffer/86400,
        "interpolation buffer end(days)": time_buffer_end/86400,

        "Observations": len(ra_deg),
        "POD iterations": number_of_pod_iterations,
        "station noise (rad)": noise_level,
        "station noise (arcsec)": LSST,

        "NGAs": {
            "A1 [m/s^2]": A1,
            "A2 [m/s^2]": A2,
            "A3 [m/s^2]": A3
        },

        "Initial conditions": {       
            'First_obs': target_sbdb['orbit']["first_obs"],
            "last_obs": target_sbdb['orbit']["last_obs"],
            "tp": target_sbdb["orbit"]["elements"]["tp"],
            "q": target_sbdb["orbit"]["elements"]["q"],
            "e": target_sbdb["orbit"]["elements"]["e"],
            "w": target_sbdb["orbit"]["elements"]["w"],
            "om": target_sbdb["orbit"]["elements"]["om"],
            "i": target_sbdb["orbit"]["elements"]["i"],
            "a": target_sbdb["orbit"]["elements"]["a"],
        },

        "NBP": bodies_to_create,
    }

    with open(f"{directory_name}{addition}/Info_syntheticobs.txt", "w") as f:
        pprint.pprint(info_dict_syncobs, stream=f, indent=2, width=80, sort_dicts=False)


    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    "this part of the code is for the covariance sampling and propagation"
    if perform_montecarlo:
        # create directory
        addition = '/Montecarlo'
        try:
            os.mkdir(f"{directory_name}{addition}")
            print(f"Directory '{directory_name}{addition}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}{addition}' already exists.")

        initial_covariance = covariance_output.covariance                                   # covariance from observations
        initial_state = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)         # Estimated initial state from sythetic observations

        np.random.seed(42)                                                                  # the answer to everything
        trajectory_parameters = initial_state.copy()
        samples = np.random.multivariate_normal(trajectory_parameters, initial_covariance, size=Orbit_samples)
        
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = timestep_global,
            coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78,
            order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )           # RK8 Fixed step integrator, to prevent interpolation
        
        termination_condition = propagation_setup.propagator.time_termination(SSE_end)

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

            A1, A2, A3 = param_dict['A1'],param_dict['A2'],param_dict['A3']
            A_vec = np.array([A1, A2, A3])

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


        # Full model with NGA
        NGAaccelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
            ],
            str(spkid): [propagation_setup.acceleration.custom_acceleration(NGA)]
        }

        # Gravity-only model
        Gravityaccelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
            ]
        }

        models = [Gravityaccelerations] #, NGAaccelerations]

        data_to_write = {
            "Sim_time": {
                'Start': SSE_start,
                'End': SSE_end,
                }, 
            "Traj_NGA": {},
            "Traj_NGAf": {},
            "mean_data": initial_state,
            "Initial_covariance": initial_covariance,
        }

        for model in models:
            acceleration_settings[str(spkid)] = model
            acceleration_models = propagation_setup.create_acceleration_models(
                    bodies, acceleration_settings, bodies_to_propagate, central_bodies
                )
            
            for sample in samples:
                propagator_settings = propagation_setup.propagator.translational(
                    central_bodies=central_bodies,
                    acceleration_models=acceleration_models,
                    bodies_to_integrate=bodies_to_propagate,
                    initial_states=sample,
                    initial_time=SSE_start,
                    integrator_settings=integrator_settings,
                    termination_settings=termination_condition,
                )

                dynamics_simulator_montecarlo = simulator.create_dynamics_simulator(
                    bodies, propagator_settings
                )

                states = dynamics_simulator_montecarlo.propagation_results.state_history
                
                print(states)

    