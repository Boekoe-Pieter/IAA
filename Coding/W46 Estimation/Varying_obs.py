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
import pickle

# other libraries
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
perform_montecarlo = True
Orbit_samples = 100
Observation_step_size = 10

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

# for sbclass in classes:
#     request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF", "A2|DF", "DT|ND"]}'
#     request_dict = {
#         'fields': 'spkid',
#         'sb-class': sbclass, 
#         'sb-cdata': request_filter,
#     }
#     response = requests.get(url, params=request_dict)
#     if response.ok:
#         all_results.extend(response.json().get("data", []))
all_results = ["C2001Q4","C2008A1","C2013US10"]

for body in all_results:
    target_mpc_code = body
    target_sbdb = SBDBquery(target_mpc_code,full_precision=True)
    print(target_sbdb)
    spkid = target_sbdb['object']['spkid']
    name  = target_sbdb['object']['fullname']
    designator = target_sbdb['object']['des']

    directory_name = f"{rel_dir}/data_{body}"

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
    "We define the LSST as our observatory, and is located in the center of the earth as we are not interested in the effect of one telescope"
    "but the effect of N observations. Sim time is one observation per day as per LSST and noise is added as per LSST (source from Pedro Lacerda ESTEC)"
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

    # ----------------------------------------------------------------------
    # Define accelerations on the body of interest
    # ----------------------------------------------------------------------
    "We re-define the accelerations on the comet, and redefine the NGA defenition as we now want to estimate the A1,A2,A3 parameters"
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

        # str(spkid): [
        #     propagation_setup.acceleration.custom_acceleration(NGA)] 
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

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=SSE_start_buffer,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
    )

    # ----------------------------------------------------------------------
    # Define the parameters to be estimated
    # ----------------------------------------------------------------------
    "Here we define the parameters to be estimated, such as the states and NGAs as last the synthetic observations are created"
    
    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    # Create the estimator
    estimator = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings,
        integrate_on_creation=True)

    Full_observation_times = np.arange(SSE_start_buffer + time_buffer, SSE_end_buffer - time_buffer_end, constants.JULIAN_DAY)
    max_observations = len(Full_observation_times)
    observation_counts = list(range(Observation_step_size, max_observations + Observation_step_size, Observation_step_size))

    # ----------------------------------------------------------------------
    # Loop over all observations
    # ----------------------------------------------------------------------
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    sim = 1

    for n_obs in observation_counts:
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

        # Simulate required observations
        simulated_observations = observations_setup.observations_wrapper.simulate_observations(
            observation_simulation_settings,
            estimator.observation_simulators,
            bodies)

        # ----------------------------------------------------------------------
        # Precise orbit determination
        # ----------------------------------------------------------------------
        "Precise orbit determination of the comets trajectory by weighting the observations, we perurb the initial conditions to make sure"
        "the POD can anlayse the dynamical model against the observations."

        # Define weighting of the observations in the inversion
        weights_per_observable = { observations.observations_processing.observation_parser(
            observable_models_setup.model_settings.angular_position_type ): noise_level ** -2}
        
        simulated_observations.set_constant_weight_per_observation_parser(weights_per_observable)

        # Save the true parameters to later analyse the error
        truth_parameters = parameters_to_estimate.parameter_vector

        # Perturb the initial state estimate from the truth
        perturbed_parameters = truth_parameters.copy( )
        for i in range(3):
            perturbed_parameters[i] += 10.0
            perturbed_parameters[i+3] += 0.01

        parameters_to_estimate.parameter_vector = perturbed_parameters

        # provide the observation collection as input, and limit number of iterations for estimation.
        pod_input = estimation_analysis.EstimationInput(
            observations_and_times=simulated_observations,
            convergence_checker= estimation_analysis.estimation_convergence_checker(
                maximum_iterations=number_of_pod_iterations,
            ),
        )

        # Set methodological options
        pod_input.define_estimation_settings(reintegrate_variational_equations=True)

        # Perform the estimation
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

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        "this part of the code is for the covariance sampling and propagation"
        if perform_montecarlo:
            subdir = f"{directory_name}/Simulation_{sim}"
            try:
                os.mkdir(subdir)
                print(f"Directory '{subdir}' created successfully.")
            except FileExistsError:
                print(f"Directory '{subdir}' already exists.")

            print("performing monte carlo")
            initial_covariance = covariance_output.covariance                                   # covariance from observations
            initial_state = bodies.get(str(spkid)).ephemeris.cartesian_state(SSE_start)         # Estimated initial state from sythetic observations
            print(np.linalg.norm(initial_state)/constants.ASTRONOMICAL_UNIT)
            np.random.seed(42)                                                                  # the answer to everything
            trajectory_parameters = initial_state.copy()
            samples = np.random.multivariate_normal(trajectory_parameters, initial_covariance, size=Orbit_samples)
            
            integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
                    time_step = timestep_global,
                    coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_89,
                    order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )
            
            termination_condition = propagation_setup.propagator.time_termination(SSE_end)

            # Gravity-only model
            Gravityaccelerations = {
                "Sun": [
                    propagation_setup.acceleration.point_mass_gravity(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
                ]
            }

            models = [Gravityaccelerations]

            # ----------------------------------------------------------------------------
            "Saving dictionaries"
            data_to_write = {
                "SBDB_reference_trajectory":0,
                "SBDB_trajectory_times":0,

                "Estimated_reference_trajectory":0,
                "Estimated_trajectory_times":0,

                "Monte_trajectory": {},
                "Monte_trajectory_times": {},
                "Initial_condition": 0,
                "Sampled_data": 0}
            
            sim_info = {
                "Body": body,
                "Simulator": "TUDAT",
                "Integrator": "RK89",
                "Sim_time": {
                    'Start_iso': first_obs,
                    'End_iso': last_obs,
                    'last obs': current_times[-1],
                    'Start_TDB': SSE_start,
                    'End_TDB': SSE_end,
                    }, 

                "timestep": timestep_global,
                "N_clones": Orbit_samples,
                "origin": global_frame_origin,
                "Orientation": global_frame_orientation,
                "Environment": bodies_to_create,
                "Planetary positions":'TUDAT Standard kernels & Horizons comet SPKID (https://py.api.tudat.space/en/latest/interface/spice.html#tudatpy.interface.spice.load_standard_kernels)',
                "Perturbations": {
                    'NBP': 'Off',
                    'Relativity': "Off",
                    },
                "used_obs": n_obs,
            }

                
                # ----------------------------------------------------------------------------
            
            # ----------------------------------------------------------------------------
            "SBDB Reference orbit, calculated at the start"
            # print("Creating reference orbit")
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

            termination_condition = propagation_setup.propagator.time_termination(SSE_end)

            initial_state = spice.get_body_cartesian_state_at_epoch(
                str(spkid),
                global_frame_origin,
                global_frame_orientation,
                "NONE",
                SSE_start,
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
                                        ]

            propagator_settings = propagation_setup.propagator.translational(
                central_bodies=central_bodies,
                acceleration_models=acceleration_models,
                bodies_to_integrate=bodies_to_propagate,
                initial_states=initial_state,
                initial_time=SSE_start,
                integrator_settings=integrator_settings,
                termination_settings=termination_condition,
            )

            dynamics_simulator = simulator.create_dynamics_simulator(
                bodies, propagator_settings
            )

            propagated_state_history = dynamics_simulator.state_history
            data_to_write['SBDB_reference_trajectory'] = np.vstack(list(propagated_state_history.values()))
            data_to_write['SBDB_trajectory_times'] = np.vstack(list(propagated_state_history.keys()))

            for model in models:
                acceleration_settings[str(spkid)] = model
                acceleration_models = propagation_setup.create_acceleration_models(
                        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

                # ----------------------------------------------------------------------------
                "Fitted Reference orbit"
                propagator_settings = propagation_setup.propagator.translational(
                        central_bodies=central_bodies,
                        acceleration_models=acceleration_models,
                        bodies_to_integrate=bodies_to_propagate,
                        initial_states=initial_state,
                        initial_time=SSE_start,
                        integrator_settings=integrator_settings,
                        termination_settings=termination_condition,)

                dynamics_simulator_estimated_state = simulator.create_dynamics_simulator(
                    bodies, propagator_settings)

                state_hist = dynamics_simulator_estimated_state.propagation_results.state_history
                data_to_write['Estimated_reference_trajectory'] = np.vstack(list(state_hist.values()))
                data_to_write['Estimated_trajectory_times'] = np.vstack(list(state_hist.keys()))

                for i, sampled_conditions in enumerate(samples):
                    propagator_settings = propagation_setup.propagator.translational(
                        central_bodies=central_bodies,
                        acceleration_models=acceleration_models,
                        bodies_to_integrate=bodies_to_propagate,
                        initial_states=sampled_conditions,
                        initial_time=SSE_start,
                        integrator_settings=integrator_settings,
                        termination_settings=termination_condition,)

                    dynamics_simulator_montecarlo = simulator.create_dynamics_simulator(
                        bodies, propagator_settings
                    )

                    state_hist = dynamics_simulator_montecarlo.propagation_results.state_history
                    
                    data_to_write['Monte_trajectory'][i] = np.vstack(list(state_hist.values()))
                    data_to_write['Monte_trajectory_times'][i] = np.vstack(list(state_hist.keys()))

                    CPU_time = dynamics_simulator_montecarlo.cumulative_computation_time_history
                    Function_evaluations = dynamics_simulator_montecarlo.cumulative_number_of_function_evaluations
                    Total_cpu_time = list(CPU_time.values())[-1]
                    Total_Function_evaluations = list(Function_evaluations.values())[-1]
            
            
            with open(f"{subdir}/Simulation_data.pkl", "wb") as f:
                pickle.dump(data_to_write, f)
            with open(f"{subdir}/Simulation_Info.pkl", "wb") as f:
                pickle.dump(sim_info, f)
            sim += 1
