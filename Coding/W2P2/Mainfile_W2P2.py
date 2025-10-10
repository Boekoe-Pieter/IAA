"""
Mainfile of week 1 PHASE 2, goal of coding:
    - limited data for q<=1.2
    - From comet interceptor mission meeting, we dont need Gravity fitted - NGA fitted + F_nga anymore
    - Adding 3BP
"""

# loading packages
import numpy as np
import rebound
import reboundx 
import pickle
from scipy import constants as const

# load python files
import Utilities as Util

#####################################################################
# DEFINE BODY SETTINGS ##############################################
#####################################################################
data = 'Coding/W2P1/unfiltered.txt'

# manually give the NGA data, CORRECT WAY:
# NGA_data = {
#   'Comet name 1': {
#       "Indicator1": {
#           'Gravity model': {},
#           'Model2': {'A1':, 'A2':, 'A3':,
#                       'm':, 'n':, 'k',
#                       'r':, 'alph':, 'tau':}
#                      },
#        "Indicator2":{....}
    
#   }
#   'Comet name 2': {.....}
# }
# Example: 'C/1885 X1': { "MODEL": {'A1':, 'A2':, 'A3': 'dT':}}

NGA_data = {
    'C/2001 Q4': {
        "Full": {   
            "n5": {"A1": 1.6506, "A2": 0.062406, "A3": 0.001412,
                   'm': 2.15, 'n': 5.093, 'k': 4.6142,
                   'r0': 2.808, 'alph': 0.1113, "tau": 0},
            "ng": {"A1": 1.6575, "A2": 0.087078, "A3": -0.010842,
                   'm': 2.15, 'n': 5.093, 'k': 4.6142,
                   'r0': 4.000, 'alph': 0.0510, "tau": 0}
        },
        "Pre": {    
            "p5": {"A1": 7.5133, "A2": -3.4627, "A3": -1.383,
                   'm': 2.15, 'n': 5.093, 'k': 4.6142,
                   'r0': 2.808, 'alph': 0.1113, "tau": 0},
            "pg": {"A1": 4.8347, "A2": -1.5452, "A3": -0.75232,
                   'm': 1.90, 'n': 5.093, 'k': 4.6142,
                   'r0': 4.000, 'alph': 0.0510, "tau": 0}
        }
    },

    'C/2002 O7': {
        "Pre Weight:True": {    
            "n5": {"A1": 34.312, "A2": -0.27275, "A3": 0,
                   'm':2.15,'n':5.093,'k':4.6142,
                   'r0':2.808,'alph':0.1113, "tau": 0},
            "c5": {"A1": 6.7831, "A2": 0.36982, "A3":-0.097038,
                   'm':2.00,'n':3.000,'k':2.6000,
                   'r0':10.000,'alph':0.0100, "tau": 0},    
        },
        "Pre Weight:False": {    
            "pn": {"A1": 192.82, "A2": 177.59, "A3": 0,
                   'm':2.15,'n':5.093,'k':4.6142,
                   'r0':2.808,'alph':0.1113, "tau": 0},
            "p1": {"A1": 8.3825, "A2": 1.4119, "A3":0,
                   'm':2.00,'n':3.000,'k':2.6000,
                   'r0':10.000,'alph':0.0100, "tau": 0},    
        }
    },

    'C/2002 T7': {
        "Full": {    
            "n5": {"A1": 0.37536, "A2": 0.30565, "A3": -0.15262,
                    'm':2.15,'n':5.093,'k':4.6142,
                    'r0':2.808,'alph':	0.1113, "tau": 0},
            "n6": {"A1": 0.047697, "A2": 0.31592, "A3":-0.17793,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": -13.1470},
            "ng": {"A1": 0.55776, "A2": 0.13539, "A3":-0.097038,
                    'm':2.15,'n':5.093,'k':4.6142,
                    'r0':1.500,'alph':	0.7255, "tau": 0},    
        },

        "Pre": {    
            "pn": {"A1": 2449.2, "A2": 	1338.1, "A3": -175.36,
                   'm':2.15,'n':5.093,'k':4.6142,
                   'r0':2.808,'alph':0.1113, "tau": 0},
            "pc": {"A1": -5.4729, "A2": 2.098, "A3":0.82991,
                   'm':2.00,'n':3.000,'k':2.6000,
                   'r0':10.000,'alph':0.0100, "tau": 0},   
            "p6": {"A1":5.5821, "A2": 1.6316, "A3":0.67917,
                   'm':2.15,'n':5.093,'k':4.6142,
                   'r0':2.808,'alph':0.1113, "tau": 0},  
        }   
    },

    'C/2003 K4': {
        "Full": {    
            "n5": {"A1": 0.85218, "A2": -0.4362, "A3": -0.066436,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": 0},
            "n6": {"A1": 0.38242, "A2": -0.024513, "A3": 0.078614,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": -89.9130},
        }
    },

    'C/2020 K8': {
        "Pre": {    
            "n5": {"A1": 0.07263, "A2": -0.13258, "A3": 0.32475,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": 0},
        }
    },

    'C/2018 V1': {
        "Pre": {    
            "n4": {"A1": -0.86633, "A2": 2.3655, "A3": 0,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": 0},
        }
    },

    'C/2017 T1': {
        "Full": {    
            "n5": {"A1": 4.3721, "A2": -1.2179, "A3": 	-0.16574,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": 0},
        },
        "Pre": {    
            "pn": {"A1": 8.3843, "A2": -7.6569, "A3": -8.0364,
                   'm':2.15,'n':5.093,'k':4.6142,'r0':2.808,
                   'alph':	0.1113, "tau": 0},
        }
    },
}
# in AU/Day**2, note it is multiplied by 10**-8 later on


Util.scatter(NGA_data)

#####################################################################
# CREATE SIM SETTINGS ###############################################
#####################################################################
integrator = "whfast"
N = 100000

primary = "Sun"
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

#####################################################################
# SIMULATING ########################################################
#####################################################################
comet_dict = Util.get_data(data)

data_dict = Util.select_comet(comet_dict, NGA_data)
bodies = list(data_dict.keys())

differences = {body: {} for body in bodies}
Save_times = {body: {} for body in bodies}

for body in bodies:
    print(f"analyzing {body}")
    data = data_dict[body]
    arcs = list(data_dict[body].keys())

    # Gravity fitted and NGA fitted + F_NGA
    cartesian_all       = {arc: {} for arc in arcs}
    F_NGA_all           = {arc: {} for arc in arcs}
    F_NGA_RTN_all       = {arc: {} for arc in arcs}
    cartesian_diff      = {arc: {} for arc in arcs}
    x_scales            = {arc: {} for arc in arcs}
    g_r                 = {arc: {} for arc in arcs}

    # PG NGA model
    NGA_PG_cartesian    = {arc: {} for arc in arcs}
    NGA_diff            = {arc: {} for arc in arcs}
    asymmetric_reference= {arc: {} for arc in arcs}

    for arc in arcs:
        print(f'analyzing {arc} data')
        models = list(data[arc].keys())
        for model in models:
            # initialize rebound sim
            sim = rebound.Simulation()
            sim.units = ('Days', 'AU', 'Msun')

            sim_NGA = rebound.Simulation()
            sim_NGA.units = ('Days', 'AU', 'Msun')

            sim_asym_ref = rebound.Simulation()
            sim_asym_ref.units = ('Days', 'AU', 'Msun')

            sim.integrator = integrator 
            sim_NGA.integrator = integrator 
            sim_asym_ref.integrator = integrator 

            # initialize reboundx
            rebx      = reboundx.Extras(sim)
            rebx_NGA  = reboundx.Extras(sim_NGA)
            rebx_sim_asym_ref  = reboundx.Extras(sim_asym_ref)

            # add relativity
            gr = rebx.load_force("gr")
            rebx.add_force(gr)
            gr_NGA = rebx_NGA.load_force("gr")
            rebx_NGA.add_force(gr_NGA)
            gr_asym_ref = rebx_sim_asym_ref.load_force("gr")
            rebx_sim_asym_ref.add_force(gr_asym_ref)
            
            c_au_per_day = const.c * const.day / const.au
            gr.params["c"] = c_au_per_day
            gr_NGA.params["c"] = c_au_per_day
            gr_asym_ref.params["c"] = c_au_per_day

            # retreive comet data
            current_comet_data = data[arc][model]  
            arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)
            
            # defining times
            start_time = arc1_mjd
            end_time = T_perihelium_mjd

            start_time_asym = start_time - current_comet_data['tau']
            end_time_asym = end_time - current_comet_data['tau']
            sim_asym_ref.t = start_time_asym

            sim.t = start_time
            sim_NGA.t = start_time
            
            # add solar system 
            JD = start_time + 2400000.5
            JD_str = 'JD'+str(JD)
            JD_nga = start_time_asym + 2400000.5
            JD_str_nga = 'JD'+str(JD_nga)

            sim.add(primary,date=JD_str)
            sim_NGA.add(primary,date=JD_str)
            sim_asym_ref.add(primary,date=JD_str_nga)

            p = sim.particles
            p_NGA = sim_NGA.particles
            p_sim_asym_ref = sim_asym_ref.particles

            for bodyS in planets:
                sim.add(bodyS,
                        date=JD_str)
                sim_NGA.add(bodyS,
                            date=JD_str)
                sim_asym_ref.add(bodyS,
                                 date=JD_str_nga)

            # add comets
            sim.add(
                    primary=p[0],
                    a=float(q)/(1-float(ecc)), 
                    e=float(ecc), 
                    inc=np.deg2rad(float(i)), 
                    Omega=np.deg2rad(float(RAAN)), 
                    omega=np.deg2rad(float(aop)),
                    T=T_perihelium_mjd,
                    hash=f"{arc}_{model}"  
                    )
            sim.move_to_com()
            
            sim_NGA.add(
                    primary=p_NGA[0],
                    a=float(q)/(1-float(ecc)), 
                    e=float(ecc), 
                    inc=np.deg2rad(float(i)), 
                    Omega=np.deg2rad(float(RAAN)), 
                    omega=np.deg2rad(float(aop)),
                    T=T_perihelium_mjd,
                    hash=f"{arc}_{model}"  
                    )
            sim_NGA.move_to_com()

            sim_asym_ref.add(
                    primary=p_sim_asym_ref[0],
                    a=float(q)/(1-float(ecc)), 
                    e=float(ecc), 
                    inc=np.deg2rad(float(i)), 
                    Omega=np.deg2rad(float(RAAN)), 
                    omega=np.deg2rad(float(aop)),
                    T=T_perihelium_mjd,
                    hash=f"{arc}_{model}"  
                    )
            sim_asym_ref.move_to_com()   
            
            times = np.linspace(start_time, end_time, N)
            times_NGA = np.linspace(start_time_asym, end_time_asym, N)

            # saving for NGA Fitted + F_NGA
            cartesian_all[arc][model]       = np.zeros((len(times), 6))
            F_NGA_all[arc][model]           = np.zeros((len(times), 3))
            F_NGA_RTN_all[arc][model]       = np.zeros((len(times), 3))
            g_r[arc][model]                 = np.zeros((len(times), 1))
            x_scales[arc][model]            = np.zeros((len(times), 1))

            # saving for NGA Fitted
            NGA_PG_cartesian[arc][model]    = np.zeros((len(times), 6))
            NGA_diff[arc][model]            = np.zeros((len(times_NGA), 6))

            asymmetric_reference[arc][model]   = np.zeros((len(times_NGA), 6))
                
            print(f"integrating orbit NGA Fitted as Gravity, model: {model}")
            for i, time in enumerate(times):
                NGA_PG_cartesian[arc][model][i] = [p_NGA[f"{arc}_{model}"].x, p_NGA[f"{arc}_{model}"].y, p_NGA[f"{arc}_{model}"].z,
                                                p_NGA[f"{arc}_{model}"].vx, p_NGA[f"{arc}_{model}"].vy, p_NGA[f"{arc}_{model}"].vz]
                                
                sim_NGA.integrate(time)

            if current_comet_data['tau'] != 0:  
                print(f"Creating asymmetric reference trajectory: {model}")
                for i, time in enumerate(times_NGA):
                    asymmetric_reference[arc][model][i] = [p_sim_asym_ref[f"{arc}_{model}"].x, p_sim_asym_ref[f"{arc}_{model}"].y, p_sim_asym_ref[f"{arc}_{model}"].z,
                                                    p_sim_asym_ref[f"{arc}_{model}"].vx, p_sim_asym_ref[f"{arc}_{model}"].vy, p_sim_asym_ref[f"{arc}_{model}"].vz]
                    
                    sim_asym_ref.integrate(time)      

            print(f"integrating orbit NGA Fitted + F_NGA accelerations, model: {model}")
            for i, time in enumerate(times):
                def NGA(reb_sim):
                    if current_comet_data['tau'] != 0:
                        r_vec = np.array(asymmetric_reference[arc][model][i,:3])
                        r_norm = np.linalg.norm(r_vec)
                        v_vec = np.array(asymmetric_reference[arc][model][i,3:])
                    else:
                        r_vec = np.array([p[f"{arc}_{model}"].x, p[f"{arc}_{model}"].y, p[f"{arc}_{model}"].z])
                        r_norm = np.linalg.norm(r_vec)
                        v_vec = np.array([p[f"{arc}_{model}"].vx, p[f"{arc}_{model}"].vy, p[f"{arc}_{model}"].vz])

                    NGA_x = data[arc][model]['A1'] * 1e-8
                    NGA_y = data[arc][model]['A2'] * 1e-8
                    NGA_z = data[arc][model]['A3'] * 1e-8
                    A_vec = np.array([NGA_x, NGA_y, NGA_z])

                    m = data[arc][model]['m']
                    n = data[arc][model]['n']
                    k = data[arc][model]['k']
                    r0 = data[arc][model]['r0']
                    alpha = data[arc][model]['alph']

                    g = alpha * (r_norm/r0)**(-m) * (1 + (r_norm/r0)**n)**(-k)
                    g_r[arc][model][i] = g

                    F_vec_rtn = g * A_vec
                    F_NGA_RTN_all[arc][model][i] = F_vec_rtn
                    C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

                    F_vec_inertial = C_rtn2eci @ F_vec_rtn
                    F_NGA_all[arc][model][i] = F_vec_inertial

                    p[f"{arc}_{model}"].ax += F_vec_inertial[0]
                    p[f"{arc}_{model}"].ay += F_vec_inertial[1]
                    p[f"{arc}_{model}"].az += F_vec_inertial[2]

                sim.additional_forces = NGA

                cartesian_all[arc][model][i] = [p[f"{arc}_{model}"].x, p[f"{arc}_{model}"].y, p[f"{arc}_{model}"].z,
                                                p[f"{arc}_{model}"].vx, p[f"{arc}_{model}"].vy, p[f"{arc}_{model}"].vz]

                x_scales[arc][model][i] = np.linalg.norm(np.array([p[f"{arc}_{model}"].x, p[f"{arc}_{model}"].y, p[f"{arc}_{model}"].z]))


                sim.integrate(time)

        difference_NGA = Util.compute_difference_NGA(arc, models, cartesian_all, NGA_PG_cartesian)

        NGA_diff[arc] = difference_NGA
        differences[body][arc] = {
                                "diff": difference_NGA,
                                "times": times,     
                                "T_peri": T_perihelium_mjd,
                                }
        
    Util.plot_difference(arcs, NGA_diff, x_scales, f"{body}", title=r'NGA fitted Pure Gravity vs NGA fitted with $F_{NGA}$',suptitle=f"Relativity + NBP, Integrator={integrator}", saver='NGAGR_NGA')      # Pure gravity NGA fitted vs NGA fitted with NGA accelerations
    Util.plot_NGA_acc_MJD(arcs, F_NGA_RTN_all, times, f"{body}", data, suptitle=f"Relativity + NBP, Integrator={integrator}")
    Util.plot_gr_AU(arcs, g_r, x_scales, f"{body}", data, suptitle=f"Relativity + NBP, Integrator={integrator}")
    Util.plot_gr_MJD(arcs, g_r, times, f"{body}", data, suptitle=f"Relativity + NBP, Integrator={integrator}")

with open("Coding/pkl_files/NGA_perturbed_states.pkl", "wb") as f:
    pickle.dump(differences, f)

print('Complete')



