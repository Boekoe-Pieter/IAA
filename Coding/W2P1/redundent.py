"""
Mainfile of week 1 PHASE 2, goal of coding:
    - limited data for q<=1.2
    - Compare NGA fitted w/ NGA on to PG fitted, and compare NGA fitted w/o NGA on to NGA fitted w/ NGA on
    - Compare 
"""

# loading packages
import numpy as np
import matplotlib.pyplot as plt
import rebound

# load python files
import Utilities as Util

TBP_on = False

#####################################################################
# DEFINE BODY SETTINGS ##############################################
#####################################################################
data = 'Coding/W2P1/unfiltered.txt'

# manually give the NGA data, CORRECT WAY: 'Name', 'Model', {'A1':, 'A2':, 'A3':, 'dT':} spacebar sensitive! 
# Example: 'C/1885 X1': { "MODEL": {'A1':, 'A2':, 'A3': 'dT':}}
# NOTE: THE FIRST MODEL ALWAYS HAS TO BE THE PURE GRAVITY FITTED DATA

NGA_data = {
    'C/2001 Q4': {
        "a6": {},
        "n5": {"A1": 1.6506, "A2": 0.062406, "A3": 0.001412,'m':2.15,'n':5.093,'k':4.6142,'r0':2.808,'alph':	0.1113, "tau": 0},
        "ng": {"A1": 1.6575, "A2": 0.087078, "A3": -0.010842,'m':2.15,'n':5.093,'k':4.6142,'r0':4.000,'alph':	0.0510, "tau": 0},
        "Data-arc": "Full"
    },

    'C/2001 Q4': {
        "pa": {},
        "p5": {"A1": 7.5133, "A2": -3.4627, "A3": -1.383,'m':2.15,'n':5.093,'k':4.6142,'r0':2.808,'alph':	0.1113, "tau": 0},
        "pg": {"A1": 4.8347, "A2": -1.5452, "A3": -0.75232,'m':1.90,'n':5.093,'k':4.6142,'r0':4.000,'alph':	0.0510, "tau": 0},
        "Data-arc": "Pre"
    },
    
    # 'C/2002 O7': {
    #     "a5": {},
    #     "n5": {"A1": 34.312, "A2": -0.27275, "A3": 0,'m':2.15,'n':5.093,'k':4.6142,'r0':2.808,'alph':0.1113, "tau": 0},
    #     "c5": {"A1": 6.7831, "A2": 0.36982, "A3":-0.097038,'m':2.00,'n':3.000,'k':2.6000,'r0':10.000,'alph':0.0100, "tau": 0},    
    # },

    # 'C/2002 T7': {
    #     "a5": {},
    #     "n5": {"A1": 0.37536, "A2": 0.30565, "A3": -0.15262,'m':2.15,'n':5.093,'k':4.6142,'r0':2.808,'alph':	0.1113, "tau": 0},
    #     "n6": {"A1": 0.047697, "A2": 0.31592, "A3":-0.17793,'m':2.15,'n':5.093,'k':4.6142,'r0':2.808,'alph':	0.0510, "tau": -13.1470},
    #     "ng": {"A1": 0.55776, "A2": 0.13539, "A3":-0.097038,'m':2.15,'n':5.093,'k':4.6142,'r0':1.500,'alph':	0.7255, "tau": 0},    
    # },
}

# in AU/Day**2, note it is multiplied by 10**-8 later on

#####################################################################
# CREATE SIM SETTINGS ###############################################
#####################################################################

integrator = "ias15"    # addative integrator
Start_delta = 0         # days

end_time_int = 0        # 0 for perihelium, 1 for arc2
End_delta = 0           # days

N = 100000

# Initialize sim
sim = rebound.Simulation()

# add primary body
primary = rebound.Particle(m=1.)
sim.add(primary)

# add integrator
sim.integrator = integrator 

p = sim.particles

#####################################################################s
# SIMULATING ########################################################
#####################################################################

comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)

bodies = data_dict.keys()
for body in bodies:
    print(f"analyzing {body}")
    current_comet_data = data_dict[body]
    models = list(data_dict[body].keys())

    for model in models:
        print(f'adding model {model}')
        current_comet_data = data_dict[body][model]
        arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)

        start_time = arc1_mjd + Start_delta
        sim.t = start_time

        sim.add(primary=p[0],
                a=float(q)/(1-float(ecc)), 
                e=float(ecc), 
                inc=np.deg2rad(float(i)), 
                Omega=np.deg2rad(float(RAAN)), 
                omega=np.deg2rad(float(aop)),
                T= T_perihelium_mjd,
                hash = str(model)
                )

    if end_time_int == 0:
        end_time = T_perihelium_mjd + End_delta
    else:
        end_time = arc2_mjd + End_delta
    
    print("integrating orbit W/O NGA and with NGA")
    times = np.linspace(start_time, end_time, N)

    kepler_all = {model: np.zeros((len(times), 6)) for model in models}
    cartesian_all = {model: np.zeros((len(times), 6)) for model in models}
    F_NGA_all = {model: np.zeros((len(times), 3)) for model in models}
    F_NGA_RTN_all = {model: np.zeros((len(times), 3)) for model in models}

    for i, time in enumerate(times):
        def NGA(reb_sim):
            for model in models[1:]:    
                r_vec = np.array([p[model].x, p[model].y, p[model].z])
                r_norm = np.linalg.norm(r_vec)
                v_vec = np.array([p[model].vx, p[model].vy, p[model].vz])

                tau = data_dict[body][model]['tau']

                NGA_x = data_dict[body][model]['A1'] * 1e-8
                NGA_y = data_dict[body][model]['A2'] * 1e-8
                NGA_z = data_dict[body][model]['A3'] * 1e-8
                A_vec = np.array([NGA_x, NGA_y, NGA_z])

                m = data_dict[body][model]['m']
                n = data_dict[body][model]['n']
                k = data_dict[body][model]['k']
                r0 = data_dict[body][model]['r0']
                alpha = data_dict[body][model]['alph']

                g = alpha * (r_norm/r0)**(-m) * (1 + (r_norm/r0)**n)**(-k)

                F_vec_rtn = g * A_vec
                F_NGA_RTN_all[model][i] = F_vec_rtn
                C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

                F_vec_inertial = C_rtn2eci @ F_vec_rtn
                F_NGA_all[model][i] = F_vec_inertial

                p[model].ax += F_vec_inertial[0]
                p[model].ay += F_vec_inertial[1]
                p[model].az += F_vec_inertial[2]

        sim.additional_forces = NGA

        for model in models:
            cartesian_all[model][i] = [p[model].x, p[model].y, p[model].z, 
                                    p[model].vx, p[model].vy, p[model].vz]
            kepler_all[model][i] = [p[model].a, p[model].e, np.rad2deg(p[model].inc), 
                                    np.rad2deg(p[model].omega), np.rad2deg(p[model].Omega), 
                                    np.rad2deg(p[model].theta)]

        sim.integrate(time)

    # compute differences
    cartesian_diff = Util.compute_difference(models,cartesian_all)
    r_norm = np.linalg.norm(cartesian_all[models[0]][:,:3],axis = 1)

    # plot differences
    Util.plot_difference(models, cartesian_diff, r_norm, body)
    Util.plot_NGA_acc_MJD(models, F_NGA_RTN_all, times, body, data_dict[body])