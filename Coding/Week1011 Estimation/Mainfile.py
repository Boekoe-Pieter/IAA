"""
This folder is dedicated for well comented and better to navigate python files.
Methodology for and equations can be found here: 
The installation of TUDAT is required: https://docs.tudat.space/en/latest/getting-started/installation.html

This files purpose is to only act as an input file.

The following tasks can be performed, and need to be specified:
1) Orbit fitting for synthetic observations, for specified observation campaign.       True/False
2) Orbit fitting with the estimation of Marsden style A1, A2 and A3.                   True/False
3) Arcwise fitting for specified arclengths.                                           True/False
4) Montecarlo sampling of the covariance matrix for each fitted orbit.                 True/False

Additional inputs:
1) Save images. np.array([False,False,False,False]) True/False
    array of True/False for: 
    - Clone Divergence per fit
    - Estimation (Confidence, Correlation & Residual) per fit
    - The fit orbit to truth orbit per fit
    - observations (RA/DEC, Aitoff, Skyplot and against time)

2) Save datafiles (pkl).    True/False

Notes for the simulation inputs:
1) Specify the Environment
    note: Bodies in the environment do not automatically mean they act as perturbers! In TUDAT it needs to be specified
    note: The code uses Spice kernels, not all bodies are in the standard TUDAT kernels and must be downloaded: 
            https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
    note: For the comets, an SPK file is generated via Horizons.
2) Integration:
    note: Since the code will compare trajectories to each other, and to avoid interpolation errors, 
            it has been opted to only use fixed step integrators found here:
            one should be wary of integration errors over long period of times with lower step integrators.

Initial conditions are taken from JPL SBDB Query via users specified filter or by specified bodies, 
    this data will be saved on the computer to not get a timeout from the API
"""
# --------------------------------------------
# Load datamanegement
import numpy as np
import os
# --------------------------------------------
# Load helper file
from Helperfile import management

if __name__ == '__main__':
    rel_dir = os.path.dirname(os.path.abspath(__file__))

    # specify your own spice file path
    Spice_files_path = '/Users/pieter/IAA/Coding/Spice_files/'

    # ------------------------------
    # Specify which tasks to perform
    # ------------------------------
    Perform_Orbit_fitting       = True
    Perform_Marsden_estimation  = True
    Perform_Arcwise_fitting     = True
    Perform_montecarlo          = True

    Save_images                 = True
    Save_data                   = True

    # ------------------------------
    # Specify Observation specifics
    # ------------------------------

    # Station
    """Defaults to cartesian if altitude = 0 and Long/Lat = 0, else it will use geodetic position
        https://py.api.tudat.space/en/latest/dynamics/environment_setup/ground_station.html#tudatpy.dynamics.environment_setup.ground_station
        DSN and EVN can also be added if needed
    """

    Station_noise    =  0.1         # arcseconds
    station_altitude =  0           # m
    Longitude        =  [0,0,0]     # E/W   [deg] for west put negative
    Latitude         =  [0,0,0]     # N     [deg]

    longitude_deg =  (Longitude[0] + Longitude[1]/60 + Longitude[2]/3600)  
    latitude_deg  =  (Latitude[0]  + Latitude[1]/60 + Latitude[2]/3600) 

    Station_latitude  = np.deg2rad(latitude_deg)
    Station_longitude = np.deg2rad(longitude_deg)

    station =  np.array([Station_noise, Station_latitude, Station_longitude, station_altitude])
 
    # Observation Campaign
    Observation_step_size = 20

    # Precise Orbit Iterations
    number_of_pod_iterations = 4
    Weighting = Station_noise**(-2)

    # ------------------------------
    # Specify Observation specifics
    # ------------------------------
    Start_first_arc     = 3     #AU
    End_first_arc       = 1.5   #AU
    Step_Arc            = -0.5  #AU
    
    # ------------------------------
    # Specify Monte carlo specifics
    # ------------------------------
    Samples     =   1000
    Seed        =   42

    # ------------------------------
    # Specify Simulation specifics
    # ------------------------------
    # timestep
    Integrator              = "RKF45"
    OrderToIntegrate        = "Higher"
    timestep_global         = 24*3600
    # Current applied integrators, can be found and modified  in Classes -> Environment -> Create_intergrator_settings
        
    # Avoid interpolation errors during POD:
    time_buffer             = 31*86400 
    time_buffer_end         = 31*86400 

    # Define the frame origin and orientation.
    global_frame_origin      = "Sun"
    global_frame_orientation = "ECLIPJ2000"

    # Specify which comets or Specify filter
    filter_ = True

    if filter_:
        """
        Filter settings are explained here: https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html
        """
        Comet_types = "HYP"
        request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF", "DT|ND"]}' # ND not defined
    else:
        Comet_types = None
        request_filter = ["C2001Q4","C2008A1","C2013US10"]

    # Environment
    bodies_to_create = [
        "Sun",

        # "Mercury",

        # "Venus",

        # "Earth",
        #     "Moon",

        # "Mars",
        #     "Phobos",
        #     "Deimos",
        
        # "Jupiter",
        #     "Io",
        #     "Europa",
        #     "Ganymede",
        #     "Callisto",

        # "Saturn",
        #     "Titan",
        #     "Rhea",
        #     "Iapetus",
        #     "Dione",
        #     "Tethys",
        #     "Enceladus",
        #     "Mimas",

        # "Uranus",
        #     "Miranda",
        #     "Ariel",
        #     "Umbriel",
        #     "Titania",
        #     "Oberon", 

        # "Neptune",
        #     "Triton",

        # "Ceres",
        # "Vesta", 
        # "Pluto",
        ]









    # ------------------------------------------------------------------------------------------------------------------------
    # GATHER REQUESTS AND PERFORM 
    # ------------------------------------------------------------------------------------------------------------------------
    Tasks = np.array([Perform_Orbit_fitting, 
                      Perform_Marsden_estimation, 
                      Perform_Arcwise_fitting, 
                      Perform_montecarlo,
                      Save_images,
                      Save_data])
    
    Estimation = np.array([
        #general
        number_of_pod_iterations,
        Observation_step_size,
        time_buffer,
        time_buffer_end,
        
        #arcwise
        Start_first_arc,
        End_first_arc,
        Step_Arc,
    ])

    montecarlo = np.array([
            Samples,
            Seed
            ])
    
    Simulation = np.array([
        Integrator,
        OrderToIntegrate,
        timestep_global,
        global_frame_origin,
        global_frame_orientation,
        request_filter,
        Comet_types
    ])

    Run = management(Tasks,station,Estimation,montecarlo,Simulation,bodies_to_create,rel_dir,Spice_files_path)
    Run.perform_requested()














