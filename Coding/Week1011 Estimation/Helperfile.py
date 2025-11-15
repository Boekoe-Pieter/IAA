from Classes import Comet_data as Data_request
from Classes import Simulation as Sim
from Classes import TudatEnvironmentBuilder as create_structs

from tudatpy.astro import time_representation, element_conversion


import pandas as pd
class management:
    def __init__(self,Tasks,station,Estimation,Montecarlo,Simulation,Environment,rel_dir,Spice_files_path):
        self.Tasks = Tasks
        self.Simulation = Simulation
        self.Environment = Environment
        self.station = station
        self.Estimation = Estimation
        self.Montecarlo = Montecarlo

        self.rel_dir = rel_dir
        self.Spice_files_path = Spice_files_path

    def Time_settings(self,first_obs,last_obs,Tp,time_buffer, time_buffer_end):
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

        return JD_start, JD_end,  SSE_start, SSE_end_cal, SSE_tp, SSE_start_buffer, SSE_end_buffer

    def perform_requested(self):
        Perform_Orbit_fitting, Perform_Marsden_estimation, Perform_Arcwise_fitting, Perform_montecarlo, Save_images, Save_data = self.Tasks
        Integrator, OrderToIntegrate, timestep_global, global_frame_origin, global_frame_orientation, request_filter, Comet_types = self.Simulation
        number_of_pod_iterations, Observation_step_size, time_buffer, time_buffer_end, Start_first_arc, End_first_arc, Step_Arc = self.Estimation
        
        # ---------------------------------
        """
        First request all the data from JPL SBDB, we save this to the computer to not get a time out from the API
        """
        filename = "SBDB_Request_data.pkl"
        comet_data = Data_request(request_filter,Comet_types,self.rel_dir,filename,self.Spice_files_path)
        # comet_data.SBDB_Query_request()
        Comet_data = pd.read_pickle(f"{self.rel_dir}/{filename}")
        count = len(Comet_data)
        print(f"\nFound {count} valid comets\n")
        print(Comet_data)
        
        # ---------------------------------
        """
        Second create SPK via horizons, we save this to the computer to not get a time out from the API.
        The time range includes the time buffer as we need it for the estimation
        """
        # for i in range(count):
        #     comet_data.Generate_spice(Comet_data.loc[i],time_buffer, time_buffer_end)
        
        Dynamics_initialize = create_structs(
            self.Environment,
            global_frame_origin,
            global_frame_orientation,
            Comet_data,
            Integrator,
            OrderToIntegrate,
            timestep_global
        )

        Dynamics_initialize.load_spice_kernals()   
        Dynamics_initialize.create_bodies()        
        Dynamics_initialize.Create_integrator_settings()  

        for spkid in Comet_data["spkid"]:
            Dynamics_initialize.add_comet(spkid)
            Dynamics_initialize.create_dynamics_struct(spkid)
            Dynamics_initialize.create_propagator_settings(spkid)
            
            if Perform_Orbit_fitting:
                print("hi")
            if Perform_Marsden_estimation:
                print("hi")
            if Perform_Arcwise_fitting:
                print("hi")
            if Perform_montecarlo:
                print("hi")
