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
Observation_step_size = 165
np.random.seed(42)

# number of iterations for our estimation
number_of_pod_iterations = 6

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

body = "C2001Q4" 

# saving dictionary
data_to_write = {
    "Spice_Reference_trajectory": 0,

    "Truth_Reference_trajectory":0,
    "Truth_Reference_trajectory_times":0,

    "N_body_trajectories": {},

    "Estimated_Reference_trajectory":{},
    "Estimated_Reference_trajectory_times":{},

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
    DT.value if DT is not None else 0,]
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
addition = "/NGA_True" if NGA_Flag else "/NGA_False"
# saving directory for NGA flag
try:
    os.mkdir(f"{directory_name}{addition}")
    print(f"Directory '{directory_name}{addition}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}{addition}' already exists.")

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

acceleration_models = Helper_file.Accelerations(spkid,
                                                bodies, 
                                                bodies_to_propagate, 
                                                central_bodies,
                                                NGA_array,
                                                NGA_Flag=NGA_Flag)

print(np.linalg.norm(initial_state_reference[:3])/constants.ASTRONOMICAL_UNIT)

# SBDB Oscullating Initial
# initial_state_reference = Helper_file.initial_state(SSE_start,Oscullating_elements,bodies)

# print(np.linalg.norm(initial_state_reference[:3])/constants.ASTRONOMICAL_UNIT)

# acceleration_models = Helper_file.twoBP(spkid,
#                                         bodies, 
#                                         bodies_to_propagate, 
#                                         central_bodies)

dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position(body, "Sun")
    for body in bodies_to_create]

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

