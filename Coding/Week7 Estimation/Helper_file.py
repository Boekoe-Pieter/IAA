# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, environment, propagation, parameters_setup, simulator
from tudatpy.astro import time_representation, element_conversion
from tudatpy.estimation import observations, observable_models, observations_setup, observable_models_setup, estimation_analysis
from tudatpy import constants

# import SBDB interface
from tudatpy.data.sbdb import SBDBquery

import requests
import json
import base64

import numpy as np

def sbdb_query(classes,request_filter):
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    all_results = []

    for sbclass in classes:
        request_dict = {
            'fields': 'spkid',
            'sb-class': sbclass, 
            'sb-cdata': request_filter,
        }
        response = requests.get(url, params=request_dict)
        if response.ok:
            all_results.extend(response.json().get("data", []))
    return all_results

def sbdb_query_info(target_body,full_precision=False):
    target_sbdb = SBDBquery(target_body,full_precision=full_precision)

    # get body name information
    spkid = target_sbdb['object']['spkid']
    name  = target_sbdb['object']['fullname']
    designator = target_sbdb['object']['des']
    comet_designation = np.array([spkid,name,designator])

    # get body time information
    first_obs = target_sbdb['orbit']['first_obs']
    last_obs  = target_sbdb['orbit']['last_obs']
    comet_time_information = np.array([first_obs,last_obs])

    # get oscullating elements
    e = target_sbdb["orbit"]['elements'].get('e')
    a = target_sbdb["orbit"]['elements'].get('a').value 
    q = target_sbdb["orbit"]['elements'].get('q').value 
    i = target_sbdb["orbit"]['elements'].get('i').value
    om = target_sbdb["orbit"]['elements'].get('om').value
    w = target_sbdb["orbit"]['elements'].get('w').value
    M = target_sbdb["orbit"]['elements'].get('ma').value
    n = target_sbdb["orbit"]['elements'].get('n').value
    Tp = target_sbdb["orbit"]['elements'].get('tp').value
    
    Oscullating_elements = np.array([e,a,q,i,om,w,M,n,Tp])

    # get NGA parameters
    A1 = target_sbdb["orbit"]["model_pars"].get("A1")
    A2 = target_sbdb["orbit"]["model_pars"].get("A2")
    A3 = target_sbdb["orbit"]["model_pars"].get("A3")
    DT = target_sbdb["orbit"]["model_pars"].get("DT")

    non_gravitational_parameters = np.array([A1,A2,A3,DT])
    return comet_designation, comet_time_information, Oscullating_elements, non_gravitational_parameters

def Horizons_SPK(Spice_files_path,spkid, J2000_start, J2000_end):
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
                spk_filename = Spice_files_path + Horizons["spk_file_id"] + ".bsp"
            try:
                f = open(spk_filename, "wb")
            except OSError as err:
                print("Unable to open SPK file '{0}': {1}".format(spk_filename, err))
            #
            # Decode and write the binary SPK file content:
            f.write(base64.b64decode(Horizons["spk"]))
            f.close()
            print("wrote SPK content to {0}".format(spk_filename))

def Accelerations(spkid,bodies, bodies_to_propagate, central_bodies,NGA_array,NGA_Flag=False,):
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

        A1,A2,A3,DT = NGA_array
        A_vec = np.array([A1,A2,A3])

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
    
    if NGA_Flag:
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

    else:
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
            }
        
        acceleration_settings = {str(spkid): accelerations}   

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    return acceleration_models

def integrator_settings(timestep_global,variable=False):
    if variable:
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            timestep_global,
            timestep_global,
            propagation_setup.integrator.CoefficientSets.rkf_1210,
            timestep_global,
            timestep_global,
            1.0,
            1.0,
        )
    else:
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
                time_step = timestep_global,
                coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_1210,
                order_to_use = propagation_setup.integrator.OrderToIntegrate.higher )  
        
    return integrator_settings

def initial_state(Start_time,osculating_elements,bodies):
    e,a,q,i,om,w,M,n,Tp = osculating_elements

    Shifter = time_representation.julian_day_to_seconds_since_epoch(Tp)
    delta_t = Shifter-Start_time
    M0=M-n*delta_t/constants.JULIAN_DAY #M in degrees, n in deg/day delta_t in seconds automatically uses hyperbolic if e>1
    true_anomaly_start = element_conversion.mean_to_true_anomaly(e, np.deg2rad(M0))

    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=bodies.get("Sun").gravitational_parameter,
        semi_major_axis=a*constants.ASTRONOMICAL_UNIT,
        eccentricity=e,
        inclination=np.deg2rad(i),
        argument_of_periapsis=np.deg2rad(w),
        longitude_of_ascending_node=np.deg2rad(om),
        true_anomaly=true_anomaly_start,
    )

    return initial_state

def propagator_settings(integrator_settings,central_bodies,acceleration_models,bodies_to_propagate,initial_state,start_time,end_time,dependent_variables_to_save):
    termination_condition = propagation_setup.propagator.time_termination(end_time,
                                                                          True)

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=start_time,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save,
    )

    return propagator_settings

    
    