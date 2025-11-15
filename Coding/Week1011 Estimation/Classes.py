# Tudat imports for propagation and estimation
from tudatpy.interface  import spice
from tudatpy.dynamics   import environment_setup, propagation_setup, environment, propagation, parameters_setup, simulator
from tudatpy.astro      import time_representation, element_conversion
from tudatpy.estimation import observations, observable_models, observations_setup, observable_models_setup, estimation_analysis
from tudatpy            import constants

# data handling modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# system libraries
import os
import sys
import requests
import json
import base64

class Comet_data:
    def __init__(self,request_filter,Comet_types,directory,filename,Spice_files_path):
        self.request_filter     =   request_filter
        self.Comet_types        =   Comet_types
        self.rel_dir            =   directory
        self.filename           =   filename
        self.Spice_files_path   =   Spice_files_path
    
    def SBDB_Query_request(self):
        url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

        request_dict = {
            'fields'    :   'spkid,full_name,e,a,q,i,om,w,A1,A2,A3,DT,first_obs,last_obs,tp',
            'sb-class'  :   self.Comet_types,
            'full-prec' :   True, 
            'sb-cdata'  :   self.request_filter,
        }

        print("\n--------------------------------------------------------------------")
        print(f"Calling SBDBQuery...\n")
        response = requests.get(url, params=request_dict)
        json_resp = response.json()
        count = json_resp['count']
        Comet_data = pd.DataFrame(json_resp["data"], columns=json_resp["fields"])
        Comet_data.to_pickle(f"{self.rel_dir}/{self.filename}")
    
    def Generate_spice(self,data,time_buffer,time_buffer_end):
        spkid,_,_,_,_,_,_,_,_,_,_,_,first_obs,last_obs,tp = data
        JD_start, JD_end,  _, _, _, _, _ = self.Time_settings(first_obs,last_obs,tp,time_buffer,time_buffer_end)

        url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
        url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
        url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, JD_start, JD_end)

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
                    spk_filename = self.Spice_files_path + Horizons["spk_file_id"] + ".bsp"
                try:
                    f = open(spk_filename, "wb")
                except OSError as err:
                    print("Unable to open SPK file '{0}': {1}".format(spk_filename, err))
                #
                # Decode and write the binary SPK file content:
                f.write(base64.b64decode(Horizons["spk"]))
                f.close()
                print("wrote SPK content to {0}".format(spk_filename))

class TudatEnvironmentBuilder:
    """"
        This class is created to:
        1) Create the bodies settings
        2) dynamics settings
        3) create the propagator settings
    """
    def __init__(self,bodies_to_create, global_frame_origin, global_frame_orientation, all_comet_data, Integrator, OrderToIntegrate, timestep_global):
        self.bodies_to_create = bodies_to_create
        self.global_frame_origin = global_frame_origin
        self.global_frame_orientation = global_frame_orientation
        self.all_comet_data = all_comet_data

        self.Integrator = Integrator
        self.Order_to_integrate = OrderToIntegrate
        self.timestep_global = timestep_global

    def create_bodies(self):
        # initialize all the bodies 
        Create_body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation
        )

        self.body_settings = Create_body_settings

    def load_spice_kernals(self):
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

    def Create_integrator_settings(self):
        Integrator_lookup = {
            "RKF45":  propagation_setup.integrator.CoefficientSets.rkf_45,

            "RKF56":  propagation_setup.integrator.CoefficientSets.rkf_56,
           
            "RKF78": propagation_setup.integrator.CoefficientSets.rkf_78,
           
            "RKDP78": propagation_setup.integrator.CoefficientSets.rkdp_87,
        
            "RKF89": propagation_setup.integrator.CoefficientSets.rkf_89,
            
            "RKF108": propagation_setup.integrator.CoefficientSets.rkf_108,
            
            "RKF1210": propagation_setup.integrator.CoefficientSets.rkf_1210,
            
            "RKF1412": propagation_setup.integrator.CoefficientSets.rkf_1412
        }

        Order_lookup = {
            "Lower": propagation_setup.integrator.OrderToIntegrate.lower,

            "Higher": propagation_setup.integrator.OrderToIntegrate.higher
        }

        if self.Integrator not in Integrator_lookup:
            raise ValueError(f"Unknown integrator method: {self.Integrator}")
        
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = float(self.timestep_global),
            coefficient_set = Integrator_lookup[self.Integrator],
            order_to_use = Order_lookup[self.Order_to_integrate])
        
        self.integrator_settings = integrator_settings

    def add_comet(self,spkid):
        self.spkid = spkid
        spice.load_kernel(f"Coding/Spice_files/{spkid}.bsp")

        # Add the comet
        self.body_settings.add_empty_settings(str(spkid))

        # Create the bodies
        bodies = environment_setup.create_system_of_bodies(self.body_settings)

        # Define the bodies, central body and which body to propagate
        self.bodies = bodies
        self.central_bodies = [self.global_frame_origin]
        self.body_to_propagate = [str(spkid)]

    def JPL_NGA_Array(self,spkid):
        row = self.all_comet_data[self.all_comet_data['spkid'] == spkid]
        A1 = row['A1'].values[0]
        A2 = row['A2'].values[0]
        A3 = row['A3'].values[0]

        self.JPL_NGA= np.array(
            [float(A1)*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A1 is not None else 0,
            float(A2)*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A2 is not None else 0,
            float(A3)*constants.ASTRONOMICAL_UNIT/constants.JULIAN_DAY**2 if A3 is not None else 0,
            ])
        
    def NGA(self,time: float) -> np.ndarray:
            state = self.bodies.get(str(self.spkid)).state
            r_vec = state[:3]
            v_vec = state[3:]

            r_norm = np.linalg.norm(r_vec)

            m = 2.15
            n = 5.093
            k = 4.6142
            r0 = 2.808*constants.ASTRONOMICAL_UNIT
            alpha = 0.1113

            A1,A2,A3 = self.JPL_NGA
            A_vec = np.array([A1,A2,A3])

            g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)

            F_vec_rtn = g * A_vec

            C_rtn2eci = element_conversion.rsw_to_inertial_rotation_matrix(state)

            F_vec_inertial = C_rtn2eci @ F_vec_rtn

            return F_vec_inertial  

    def NGA_Est(self):
        x=1

    def create_dynamics_struct(self,spkid):
        self.JPL_NGA_Array(spkid)

        accelerations = {
            body: [propagation_setup.acceleration.point_mass_gravity()]
            for body in self.bodies_to_create
        }

        # accelerations.update(str(spkid))
        accelerations[str(spkid)] = [propagation_setup.acceleration.custom_acceleration(self.NGA)]

        # accelerations on comet
        acceleration_settings = {str(spkid): accelerations}   

        self.acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies, acceleration_settings, self.body_to_propagate, self.central_bodies
        )

    def create_propagator_settings(self,spkid):
        row = self.all_comet_data[self.all_comet_data['spkid'] == spkid]
        first_obs   =   row['first_obs'].values[0]
        last_obs    =   row['last_obs'].values[0]
        tp          =   row['tp'].values[0]

        self.Comet_data = Comet_data()
        _, _, SSE_start, SSE_end_cal, SSE_tp, SSE_start_buffer, SSE_end_buffer = \
            self.Comet_data.Time_settings(first_obs, last_obs, tp, None, None)


        termination_condition = propagation_setup.propagator.time_termination(end_time,
                                                                          True)

        propagator_settings = propagation_setup.propagator.translational(
            central_bodie           =   self.central_bodies,
            acceleration_models     =   self.acceleration_models,
            bodies_to_integrate     =   self.body_to_propagate,
            initial_states          =   initial_state,
            initial_time            =   start_time,
            integrator_settings     =   self.integrator_settings,
            termination_settings    =   termination_condition,
            output_variables        =   dependent_variables_to_save,
        )

class Simulation:
    def __init__(self):
        x=1
    def Standard_sim(self):
        x=1
    def Marsden(self):
        x=1
        