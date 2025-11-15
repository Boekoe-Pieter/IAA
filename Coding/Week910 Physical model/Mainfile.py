"""
This is the main file for the Thermol Phsyical Modeling, the code performs the following:

First) compute reasonable changes in orbital elements from the database
    For each comet + Marsden NGA in JPL -> integrate with only Sun (no external perturbation) 
    -> Output: Orbital Elements (t=t_end) - Orbital Elements (t=0). 
    
    Output: plot of displacement with time vs. PG orbit.
    Reconstruct Histogram_JPL in Delta a, Delta OMEGA, Delta omega

Second)
    Sample the 25 regions randomly: i.e. set them all to zero except for maybe around 5 that are set to 1.
    For each sample: plot sum(Q) over time and compare to brightening laws. 
    Discard samples where Q is outside the Q envelope whilst inside of 3AU. 

    For each sample: integrate with only Sun (no external perturbation) + interpolated NGA law using summed(RTN NGAs) 
    -> Output: Orbital Elements (t=t_end) - Orbital Elements (t=0). Output: plot of displacement with time vs. PG orbit.
    Discard samples that fall outside range of Histogram_JPL.

All comets will be filtered and called via the SBDBQuerry, made TUDAT free.
"""
# Simulator libraries
from scipy.stats import norm
import random
from astropy.time import Time
import random

# load in technical libraries
import requests
import json
import base64
import pickle
import os
import sys

# load in data management libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all the classes
from Classes import Simulator as Sim
from Classes import Physical_model as TPM
from Classes import plotting as plot

sys.path.append('/Users/pieter/IAA/Coding')
rel_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__': 

    #------------------------------------
    # Define Environment, Integrator & Datarate
    Primary = "Sun"         # -> Heliocentric
    Secondaries = None      # 2BP
    Integrator = "trace"    # IAS15, WHFast combined
    Timestep = 0.1          # Days
    datarate = 10000        # Output array length

    Max_region_sampling = 5 # Thermophysical areas
    Number_of_sampling_combs = 1000

    # seed = 42               # Random sampling
    seeds = np.array([42,1000,78,80176,129386,12374,1,420])
    Reference_Orbit = "C/2001 Q4"

    N_comets = "Large" # OR "Small" => relates to perihelion limit
    filename = f"Comet_data_{N_comets}.pkl" #save the comet data from the JPL call to not get banned for to many requests (:

    #------------------------------------
    # Calculate change in semi-major, 
    # Ascending node and 
    # Argument of Perihelion 
    # due to A1,A2,A3 for all jpl comets
    #------------------------------------

    #------------------------------------
    # Call the applicable comets
    if N_comets == "Large":
        request_filter = '{"AND":["A1|DF"]}'
    else:
        request_filter = '{"AND":["q|RG|0.80|1.20", "A1|DF"]}'
    
    if not os.path.exists(f"{rel_dir}/{filename}"):
        classes = ["HYP","PAR","COM"]

        url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

        request_dict = {
            'fields'    :   'full_name,e,a,q,i,om,w,A1,A2,A3,DT,first_obs,last_obs,tp',
            'sb-class'  :   "HYP,PAR,COM",
            'full-prec' :   True, 
            'sb-cdata'  :   request_filter,
        }

        print("--------------------------------------------------------------------")
        print(f"Calling SBDBQuery...\n")
        response = requests.get(url, params=request_dict)
        json_resp = response.json()
        count = json_resp['count']

        print("--------------------------------------------------------------------")
        print(f"Found {count} valid comets\n")
        Comet_data = pd.DataFrame(json_resp["data"], columns=json_resp["fields"])
        Comet_data.to_pickle(f"{rel_dir}/{filename}")

    else:
        print(f"File '{filename}' already exists — skipping download.")

    for seed in seeds:
        directory_name = f"{rel_dir}/plots_for_seed{seed}"

        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        
        random.seed(seed)
    
        #------------------------------------
        # Simulate every comet
        print("--------------------------------------------------------------------")
        print("Performing Orbit Simulation\n")
        Comet_data = pd.read_pickle(f"{rel_dir}/{filename}")
        count = len(Comet_data)

        # data arrays for JPL
        All_elements = np.zeros((count, 3))
        gr_check = np.zeros((count, datarate, 1))
        times = np.zeros((count, datarate, 1))
        names = np.empty(count, dtype=object)

        for i in range(count):
            print(f"\n\n{Comet_data.loc[i]}")
            Perform_Simulation = Sim(Primary, Integrator, Timestep, Comet_data.loc[i], datarate)
            Perform_Simulation.simulate_marsden()
            elements = Perform_Simulation.data_array
            difference = elements[-1]-elements[0]

            All_elements[i,:] = difference
            gr_check[i] = Perform_Simulation.gr
            times[i:] = Perform_Simulation.times_T_tp
            names[i] = Comet_data.loc[i,"full_name"]

            print("\n")
            

        plot_STD = plot(All_elements)
        plot_STD.plot_STD(saving=f"{directory_name}/Change_in_kepler_JPL.pdf")
        statistics = plot_STD.statistics
        plot_STD.plot_gr(gr_check,times,names,saving=f"{directory_name}/gr_check.pdf")
        
        
        # ------------------------------------
        # ------------------------------------
        # ------------------------------------
        # Thermophysical modeling 
        # ------------------------------------
        # ------------------------------------
        # ------------------------------------

        # ------------------------------------
        # Retrieve Reference Orbit
        C2001Q4 = Comet_data[Comet_data["full_name"].str.contains(Reference_Orbit, case=False, na=False)]
        Comet_indx = Comet_data[Comet_data["full_name"].str.contains(Reference_Orbit, case=False, na=False)].index[0]
        times = [C2001Q4["first_obs"].values[0] ,C2001Q4["last_obs"].values[0]]
        t = Time(times, format='fits', scale='utc')
        Start_time = t.mjd[0]
        End_time = t.mjd[1]

        times = [C2001Q4["tp"].values[0]]
        t = Time(times, format='jd', scale='utc')
        tp_MJD = t.mjd

        # time array for interpolation
        time_to_peri = np.arange(Start_time,End_time+Timestep,Timestep) - tp_MJD

        #------------------------------------
        # Thermophysical modeling sampling

        Physical_model_file = f"Coding/Week910 Physical model/OutputEAF=1_t_rh_Q_NGA.dat"
        with open(Physical_model_file, "rb") as f:
            Physical_model_data = pickle.load(f)
        NGA = Physical_model_data[3]

        num_rows, num_cols = NGA[0].shape
        regions = [i for i in range(num_cols)]

        Physical_model = TPM(Physical_model_data)
        Physical_model.create_pandas()

        sampled_regions_list = []
        valid_regions = []
        invalid_regions = []
        invalid_sims = []


        All_elements = np.zeros((Number_of_sampling_combs, 3))
        All_trajectories = np.zeros((Number_of_sampling_combs, datarate, 6))
        valid_mask = np.zeros(Number_of_sampling_combs, dtype=bool)

        Perform_Simulation = Sim(Primary, Integrator, Timestep, C2001Q4.loc[Comet_indx].T, datarate)
        for i in range(Number_of_sampling_combs):
            sampled_regions = random.sample(regions,Max_region_sampling)
            sampled_regions_list.append(sampled_regions)

            # Only pass valid region sampling such that Q is within the light curve & store invalid ones
            R,T,N,passed_regions,z = Physical_model.Q_validation(sampled_regions)
            if z == 0:
                invalid_regions.append(passed_regions)
                continue
                
            # If valid compute trajectory and change in Semi-major, AN, AoP
            Perform_Simulation.simulate_Physical(R,T,N,Physical_model_data[1],Physical_model_data[0],time_to_peri)
            elements = Perform_Simulation.data_array
            trajectory = Perform_Simulation.trajectory
            times = Perform_Simulation.times

            difference = elements[-1]-elements[0]   # semi_major, Ascending_Node, Argument_Perihelion

            # Only pass valid trajectories that are within the JPL sigmas & store invalid ones
            semi_major = statistics[0]              #median, STD
            Ascending_Node = statistics[1]          #median, STD
            Argument_Perihelion = statistics[2]     #median, STD

            invalid = (
                np.abs(difference[0]) > semi_major[1] or
                np.abs(difference[1]) > Ascending_Node[1] or
                np.abs(difference[2]) > Argument_Perihelion[1]
            )

            if invalid:
                invalid_sims.append(passed_regions)
                continue

            All_elements[i, :] = difference
            All_trajectories[i, :, :] = trajectory
            valid_mask[i] = True
            valid_regions.append(sampled_regions)

            progress = (i + 1) / Number_of_sampling_combs
            bar_length = 30
            bar = '=' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
            
            sys.stdout.write(f"\rProgress: [{bar}] {progress * 100:.1f}%")
            sys.stdout.flush()

        print(f"\n{len(invalid_regions)/Number_of_sampling_combs*100} % invalid region combinations")
        print(f"\n{len(invalid_sims)/(Number_of_sampling_combs-len(invalid_regions))*100} % invalid simulations")
        print(f"\n{(len(invalid_sims)+len(invalid_regions))/(Number_of_sampling_combs)*100} % invalid samples")

        print(f"\n{100-(len(invalid_sims)+len(invalid_regions))/(Number_of_sampling_combs)*100} % valid samples")

        # filter only valid combinations -> removing zeros from the array
        All_elements = All_elements[valid_mask]
        All_trajectories = All_trajectories[valid_mask]

        Marsden= Sim(Primary, Integrator, Timestep, C2001Q4.loc[Comet_indx].T, datarate)
        Marsden.simulate_marsden()
        Marsden_trajectory = Marsden.trajectory

        TwoBP = Sim(Primary, Integrator, Timestep, C2001Q4.loc[Comet_indx].T, datarate)
        TwoBP.simulate_2BP()
        TwoBP_trajectory = TwoBP.trajectory

        plotter = plot(All_elements)
        
        plotter.deviation_time(times-tp_MJD,All_trajectories,Marsden_trajectory,TwoBP_trajectory,saving=f"{directory_name}/Physical_model_deviation_time.pdf")
        plotter.deviation_AU(times-tp_MJD,All_trajectories,Marsden_trajectory,TwoBP_trajectory,saving=f"{directory_name}/Physical_model_deviation_AU.pdf")
        
        plotter.plot_STD(saving=f"{directory_name}/Change_in_kepler_Physical.pdf") 
        
        plotter.plot_line_kms(All_trajectories,Marsden_trajectory, title="One sigma velocity deviation, Thermophysical agaist Marsden", saving=f"{directory_name}/Difference_to_marsden_kms.pdf", saving2=f"{directory_name}/Difference_to_marsden_kms_zoomed.pdf")
        plotter.plot_line_km(All_trajectories,Marsden_trajectory, title="One sigma position deviation, Thermophysical agaist Marsden", saving=f"{directory_name}/Difference_to_marsden_km.pdf", saving2=f"{directory_name}/Difference_to_marsden_km_zoomed.pdf")
        
        plotter.plot_line_kms(All_trajectories,TwoBP_trajectory, title="One sigma velocity deviation, Thermophysical agaist 2BP",saving=f"{directory_name}/Difference_to_2BP_kms.pdf", saving2=f"{directory_name}/Difference_to_2BP_kms_zoomed.pdf")
        plotter.plot_line_km(All_trajectories,TwoBP_trajectory, title="One sigma position deviation, Thermophysical agaist 2BP",saving=f"{directory_name}/Difference_to_2BP_km.pdf", saving2=f"{directory_name}/Difference_to_2BP_km_zoomed.pdf")

        plotter.sample_stats(regions,valid_regions,invalid_regions,invalid_sims,title="(in)valid sample information",saving=f"{directory_name}/sample size.pdf")
        Physical_model.random_valid_Q(random.choice(valid_regions),saving=f"{directory_name}/Random_sample_Q.pdf")