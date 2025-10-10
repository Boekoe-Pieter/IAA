"""
Mainfile of week 1 PHASE 2, goal of coding:
    - limited data for q<=1.5
    - Compare to data pure gravity
    - 3BP + relativity / acc analysis
"""

# loading packages
import numpy as np
import matplotlib.pyplot as plt
import rebound
import reboundx

# load python files
import Utilities as Util

TBP_on = False

#####################################################################
# DEFINE BODY SETTINGS ##############################################
#####################################################################
data_file_NGA = 'Coding/W1P2/pre_arc_standard.txt'
data_file_GR =  'Coding/W1P2/pre_arc_pg.txt'

bodies_of_interest = [
            'C/1990 K1',
            'C/1999 S4',
            'C/2001 Q4',
            'C/2002 T7',
            'C/2003 T3',
            ]

Main_body = "Sun"

NGA_data = {
    'C/1990 K1':  {"A1": 0.67588, "A2": -3.8056, "A3": 2.608},
    'C/1999 S4':  {"A1": 9.3198, "A2": -0.95107, "A3": 0.84772},
    'C/2001 Q4':  {"A1": 7.5133, "A2": -3.4627, "A3": -1.383},
    'C/2002 T7':  {"A1": 2449.2, "A2": 1338.1, "A3": -175.36},
    'C/2003 T3':  {"A1": 4.2858, "A2": 4.0988, "A3": 3.4581},
   
} # in AU/Day**2, note it is multiplied by 10**-8 later on

#####################################################################
# CREATE SIM SETTINGS ##############################################
#####################################################################

integrator = "whfast"    # addative integrator
Start_delta = 0         # days
end_time_int = 0        # 0 for perihelium, 1 for arc2
End_delta = 3          # days

N = 100000

#####################################################################
# SIMULATING ########################################################
#####################################################################

for body in bodies_of_interest:
    print(f"analyzing {body}")    
    # Initialize data for pure gravity orbit   
    data_gr       =   Util.get_data(data_file_GR)
    comet_gr      =   data_gr.get(body, None)

    # Initialize data for NGA orbit and add NGA values
    NGA_data_body =   NGA_data[body]
    data_NGA      =   Util.add_NGA(Util.get_data(data_file_NGA),NGA_data_body)
    comet_NGA     =   data_NGA.get(body, None)

    # Retrieve data for both fitted parameters
    arc1_mjd_gr, arc2_mjd_gr, epoch_mjd_gr, T_perihelium_mjd_gr, q_gr, ecc_gr, aop_gr, RAAN_gr, i_gr, a_recip_gr = Util.get_orbital_elements(comet_gr)
    arc1_mjd_NGA, arc2_mjd_NGA, epoch_mjd_NGA,T_perihelium_mjd_NGA, q_NGA, ecc_NGA, aop_NGA, RAAN_NGA, i_NGA, a_recip_NGA = Util.get_orbital_elements(comet_NGA)

    # Initialize sim & create environment
    start_time = arc1_mjd_gr+Start_delta
    sim = Util.create_sim(integrator,start_time,TBP_on)
    p = sim.particles

    sim.add(primary=p[0],
            a=float(q_gr)/(1-float(ecc_gr)), 
            e=float(ecc_gr), 
            inc=np.deg2rad(float(i_gr)), 
            Omega=np.deg2rad(float(RAAN_gr)), 
            omega=np.deg2rad(float(aop_gr)),
            T= T_perihelium_mjd_gr,
            )
    
    sim.add(primary=p[0],
            a=float(q_NGA)/(1-float(ecc_NGA)), 
            e=float(ecc_NGA), 
            inc=np.deg2rad(float(i_NGA)), 
            Omega=np.deg2rad(float(RAAN_NGA)), 
            omega=np.deg2rad(float(aop_NGA)),
            T= T_perihelium_mjd_NGA,
            )
    
    # Integrate
    if end_time_int == 0:
        end_time = T_perihelium_mjd_gr + End_delta
    else:
        end_time = arc2_mjd_gr + End_delta

    times = np.linspace(start_time,end_time,N)
    kepler_gr = np.zeros((len(times), 6))
    cartesian_gr = np.zeros((len(times), 6))
    kepler_NGA = np.zeros((len(times), 6))
    cartesian_NGA = np.zeros((len(times), 6))
    F_NGA = np.zeros((len(times),3))
    F_NGA_RTN = np.zeros((len(times),3))

    print("integrating orbit W/O NGA and with NGA")
    for i, time in enumerate(times):
        def NGA(reb_sim):
            r_vec = np.array([p[-1].x, p[-1].y, p[-1].z])
            r_norm = np.linalg.norm(r_vec)

            v_vec = np.array([p[-1].vx, p[-1].vy, p[-1].vz])

            NGA_x = NGA_data[body]['A1'] * 1e-8
            NGA_y = NGA_data[body]['A2'] * 1e-8
            NGA_z = NGA_data[body]['A3'] * 1e-8
            A_vec = np.array([NGA_x, NGA_y, NGA_z])

            m = 2.15
            n = 5.093
            k = 4.6142
            r0 = 2.808
            alpha = 0.1113

            g = alpha * (r_norm/r0)**(-m) * (1 + (r_norm/r0)**n)**(-k)

            F_vec_rtn = g * A_vec
            F_NGA_RTN[i] = F_vec_rtn
            C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

            F_vec_inertial = C_rtn2eci @ F_vec_rtn
            F_NGA[i] = F_vec_inertial

            p[-1].ax += F_vec_inertial[0]
            p[-1].ay += F_vec_inertial[1]
            p[-1].az += F_vec_inertial[2]

        sim.additional_forces = NGA
        sim.integrate(time)
        # sim.status()
        cartesian_gr[i] = [p[-2].x, p[-2].y, p[-2].z, p[-2].vx, p[-2].vy, p[-2].vz]
        kepler_gr[i] = [p[-2].a, p[-2].e, np.rad2deg(p[-2].inc), np.rad2deg(p[-2].omega), np.rad2deg(p[-2].Omega), np.rad2deg(p[-2].theta)]
        
        cartesian_NGA[i] = [p[-1].x, p[-1].y, p[-1].z, p[-1].vx, p[-1].vy, p[-1].vz]
        kepler_NGA[i] = [p[-1].a, p[-1].e, np.rad2deg(p[-1].inc), np.rad2deg(p[-1].omega), np.rad2deg(p[-1].Omega), np.rad2deg(p[-1].theta)]

    # compute differences
    cartesian_diff = Util.compute_difference(cartesian_gr,cartesian_NGA)
    kepler_diff = Util.compute_difference(kepler_gr,kepler_NGA)
    r_norm = np.linalg.norm(cartesian_gr[:,:3],axis =1)

    # plot differences
    Util.plot_difference(cartesian_diff, r_norm,body, c=True)
    Util.plot_difference(kepler_diff, r_norm,body, c=False)
    Util.plot_NGA_acc(F_NGA,r_norm,body,RTN=False)
    Util.plot_NGA_acc(F_NGA_RTN,r_norm,body,RTN=True)
    Util.plot_NGA_acc_MJD(F_NGA,times,body,T_perihelium_mjd_gr,RTN=False)
    Util.plot_NGA_acc_MJD(F_NGA_RTN,times,body,T_perihelium_mjd_gr,RTN=True)


    