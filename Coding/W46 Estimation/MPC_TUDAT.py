# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import constants
from tudatpy.astro import time_conversion

# import MPC interface
from tudatpy.data.mpc import BatchMPC

# import SBDB interface
from tudatpy.data.sbdb import SBDBquery

# other useful modules
import numpy as np
# import datetime

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

# Basic spice kernel
spice.load_standard_kernels()

# ----------------------------------------------------------------------
# Define saving directories
sys.path.append('/Users/pieter/IAA/Coding')
Spice_files = '/Users/pieter/IAA/Coding/Spice_files/'
rel_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Define iterations, timesteps and frames

# number of iterations for our estimation
number_of_pod_iterations = 6 #Optimum??

# timestep of 1 hours for our estimation
timestep_global = 1* 3600 #Analyze??

# 1 month time buffer used to avoid interpolation errors:
time_buffer =  1 * 31 * 86400 #Analyze??

# define the frame origin and orientation.
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

# ----------------------------------------------------------------------
# Retrieve data from SBDB
target_mpc_code = "C/2001 Q4"

target_sbdb = SBDBquery(target_mpc_code)

spkid = target_sbdb['object']['spkid']
name  = target_sbdb['object']['fullname']
designator = target_sbdb['object']['des']
print(f"SPKid: {spkid} for body {name}")

first_obs = target_sbdb['orbit']['first_obs']
last_obs  = target_sbdb['orbit']['last_obs']

print("First obs:", first_obs)
print("Last obs:", last_obs)

# ----------------------------------------------------------------------
# Load observations from the Minor Planet Centre
observations_start = first_obs
observations_end = last_obs

mpc_codes = [target_mpc_code]                     
batch = BatchMPC()
batch.get_observations(mpc_codes)
batch.filter(
    epoch_start=observations_start,
    epoch_end=observations_end,
)
batch.summary()

# Retrieve the first and final observation epochs and add the buffer
epoch_start_nobuffer = batch.epoch_start
epoch_end_nobuffer = batch.epoch_end

epoch_start_buffer = epoch_start_nobuffer - time_buffer
epoch_end_buffer = epoch_end_nobuffer + time_buffer


J2000_start = "JD" + str(time_conversion.seconds_since_epoch_to_julian_day(epoch_start_buffer))
J2000_end = "JD" + str(time_conversion.seconds_since_epoch_to_julian_day(epoch_end_buffer))

# ----------------------------------------------------------------------
# Create SPK file via JPL Horizons from the observations of 
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

spicepy.furnsh("Coding/Spice_files/1000351.bsp")

# ----------------------------------------------------------------------
# Define the Environment
bodies_to_create = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Moon",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

# Create system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

bodies = environment_setup.create_system_of_bodies(body_settings)

# ----------------------------------------------------------------------
# Retrieve body name from MPC and make TUDAT compatible 

# retrieve body name
bodies_to_propagate = batch.MPC_objects
central_bodies = [global_frame_origin]

# Transform the MPC observations into a tudat compatible format.
observation_collection = batch.to_tudat(bodies=bodies, included_satellites=None)

# ----------------------------------------------------------------------
# From the observations, define their positions

# set create angular_position settings for each link in the list.
observation_settings_list = list()
link_list = list(
    observation_collection.get_link_definitions_for_observables(
        observable_type=observation.angular_position_type
    )
)

for link in link_list:
    # add optional bias settings here
    observation_settings_list.append(
        observation.angular_position(link, bias_settings=None)
    )
# ----------------------------------------------------------------------
# Define accelerations on the body of interest
accelerations = {
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
    ],
    "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
    "Venus": [propagation_setup.acceleration.point_mass_gravity()],
    "Earth": [propagation_setup.acceleration.point_mass_gravity()],
    "Moon": [propagation_setup.acceleration.point_mass_gravity()],
    "Mars": [propagation_setup.acceleration.point_mass_gravity()],
    "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
    "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
    "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
    "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
}

# Set up the accelerations settings for each body
acceleration_settings = {}
for body in batch.MPC_objects:
    acceleration_settings[str(body)] = accelerations

# create the acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# ----------------------------------------------------------------------
# Define accelerations on the body of interest

# benchmark state for later comparison retrieved from SPICE
initial_states = spice.get_body_cartesian_state_at_epoch(
    spkid,
    global_frame_origin,
    global_frame_orientation,
    "NONE",
    epoch_start_buffer,
)
print(np.linalg.norm(initial_states[:3])/constants.ASTRONOMICAL_UNIT)

# Add random offset for initial guess
rng = np.random.default_rng(seed=42) 

initial_position_offset = 1e6 * 1000
initial_velocity_offset = 100

initial_guess = initial_states.copy()
initial_guess[0:3] += (2 * rng.random(3) - 1) * initial_position_offset
initial_guess[3:6] += (2 * rng.random(3) - 1) * initial_velocity_offset

print("Error between the real initial state and our initial guess:")
print(initial_guess - initial_states)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    epoch_start_buffer,
    timestep_global,
    propagation_setup.integrator.CoefficientSets.rkf_78,
    timestep_global,
    timestep_global,
    1.0,
    1.0,
)

# Terminate at the time of oldest observation
termination_condition = propagation_setup.propagator.time_termination(epoch_end_buffer)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies,
    acceleration_models=acceleration_models,
    bodies_to_integrate=bodies_to_propagate,
    initial_states=initial_guess,
    initial_time=epoch_start_buffer,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition,
)

# Setup parameters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(
    propagator_settings, bodies
)

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(
    parameter_settings, bodies, propagator_settings
)

# Set up the estimator
estimator = numerical_simulation.Estimator(
    bodies=bodies,
    estimated_parameters=parameters_to_estimate,
    observation_settings=observation_settings_list,
    propagator_settings=propagator_settings,
    integrate_on_creation=True,
)

# provide the observation collection as input, and limit number of iterations for estimation.
pod_input = estimation.EstimationInput(
    observations_and_times=observation_collection,
    convergence_checker=estimation.estimation_convergence_checker(
        maximum_iterations=number_of_pod_iterations,
    ),
)

# Set methodological options
pod_input.define_estimation_settings(reintegrate_variational_equations=True)

# Perform the estimation
pod_output = estimator.perform_estimation(pod_input)

# retrieve the estimated initial state.
results_final = pod_output.parameter_history[:, -1]

vector_error_initial = (np.array(initial_guess) - initial_states)[0:3]
error_magnitude_initial = np.sqrt(np.square(vector_error_initial).sum()) / 1000

vector_error_final = (np.array(results_final) - initial_states)[0:3]
error_magnitude_final = np.sqrt(np.square(vector_error_final).sum()) / 1000

print(
    f"{name} initial guess radial error to spice: {round(error_magnitude_initial, 2)} km"
)
print(
    f"{name} final radial error to spice: {round(error_magnitude_final, 2)} km"
)

residual_history = pod_output.residual_history

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
    np.array(observation_collection.concatenated_times) / (86400 * 365.25) + 2000
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

plt.savefig("residuals")
plt.show()

# Correlation can be retrieved using the CovarianceAnalysisInput class:
covariance_input = estimation.CovarianceAnalysisInput(observation_collection)
covariance_output = estimator.compute_covariance(covariance_input)

correlations = covariance_output.correlations
estimated_param_names = ["x", "y", "z", "vx", "vy", "vz"]


fig, ax = plt.subplots(1, 1, figsize=(9, 7))

im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

# add numbers to each of the boxes
for i in range(len(estimated_param_names)):
    for j in range(len(estimated_param_names)):
        text = ax.text(
            j, i, round(correlations[i, j], 2), ha="center", va="center", color="w"
        )

cb = plt.colorbar(im)

ax.set_xlabel("Estimated Parameter")
ax.set_ylabel("Estimated Parameter")

fig.suptitle(f"Correlations for estimated parameters for {name}")

fig.set_tight_layout(True)
plt.savefig("Covar")
plt.show()