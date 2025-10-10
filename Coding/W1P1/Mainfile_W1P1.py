"""
Mainfile of week 1 PHASE 1, goal of coding:
    - understand cometary data
    - understand standard NGA model
    - understand Rebound
    - generate orbit trajectories
    - understand Reboundx
    - generate nominal vs NGA kepler/cowell data

This file will show the trajectory of the selected comet of the standard g(r) fitted data. It also includes an comparison to
these orbit parameters with pure gravity to the NGA turned on.
    NOTE: Since the orbit parameters are fitted with standard g(r) this is not a 'fair' comparison, thus see Mainfile_W1P2.py 
    where we compare the same data but fitted for pure gravity to the same data but fitted for g(r)
"""
# loading packages
import numpy as np
import matplotlib.pyplot as plt
import rebound
import reboundx

# load python files
import Utilities as Util

plot_unperturbed = True

#####################################################################
# DEFINE BODY SETTINGS ##############################################
#####################################################################
data_file = 'Coding/W1P1/pre_arc_standard.txt'

bodies_of_interest = [
            'C/2001 Q4',
            ]

Main_body = "Sun"

NGA_data = {
    'C/2001 Q4':  {"A1": 7.5133, "A2": -3.4627, "A3": -1.383},
   
} # in AU/Day**2, note it is multiplied by 10**-8 later on

#####################################################################
# CREATE SIM SETTINGS ##############################################
#####################################################################

integrator = "ias15"    # addative integrator
Start_delta = 0         # days
end_time_int = 0        # 0 for perihelium, 1 for arc2
End_delta = 0          # days

N = 100000

#####################################################################
# SIMULATING ########################################################
#####################################################################

for body in bodies_of_interest:
    print(f"analyzing {body}")
    # initialize data & add NGA to NGA fitted data    
    NGA_data_body =   NGA_data[body]
    data_NGA      =   Util.add_NGA(Util.get_data(data_file),NGA_data_body)
    comet_NGA     =   data_NGA.get(body, None)

    # Retrieve data for both fitted parameters
    arc1_mjd, arc2_mjd, epoch_mjd,T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(comet_NGA)
    print(arc1_mjd, arc2_mjd, epoch_mjd,T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip)
    # initialize sim & create environment
    start_time = arc1_mjd+Start_delta
    sim = Util.create_sim(integrator,start_time)
    sim_nga = Util.create_sim(integrator,start_time)
    

    comet_orbit = rebound.Particle(simulation=sim, 
                                   a=float(q)/(1-float(ecc)),
                                   e=float(ecc), 
                                   inc=np.deg2rad(float(i)), 
                                   Omega=np.deg2rad(float(RAAN)), 
                                   omega=np.deg2rad(float(aop)),
                                   T= T_perihelium_mjd,
                                   hash=body)
    sim.add(comet_orbit)
    sim_nga.add(comet_orbit)

    # Integrate
    if end_time_int ==0:
        end_time = T_perihelium_mjd + End_delta
    else:
        end_time = arc2_mjd + End_delta

    times = np.linspace(start_time,end_time,N)
    kepler = np.zeros((len(times), 6))
    cartesian = np.zeros((len(times), 6))
    print("integrating 2BP orbit")
    for i, time in enumerate(times):
        sim.integrate(time)
        p = sim.particles[1] 
        cartesian[i] = [p.x, p.y, p.z, p.vx, p.vy, p.vz]
        kepler[i] = [p.a, p.e, np.rad2deg(p.inc), np.rad2deg(p.omega), np.rad2deg(p.Omega), np.rad2deg(p.theta)]

    if plot_unperturbed:
        Util.plot_elements(sim,times,kepler,cartesian,body)

    # Adding NGA
    kepler_NGA = np.zeros((len(times), 6))
    cartesian_NGA = np.zeros((len(times), 6))
    p_nga = sim_nga.particles[1] 
    F_NGA = np.zeros((len(times),3))
    
    print("integrating orbit with NGA")
    for i, time in enumerate(times):
        def NGA(reb_sim):
            """
            definition to calculate the NGA
            """
            r_vec = np.array([p_nga.x, p_nga.y, p_nga.z])
            r_norm = np.linalg.norm(r_vec)

            v_vec = np.array([p_nga.vx, p_nga.vy, p_nga.vz])


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
            C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

            F_vec_inertial = C_rtn2eci @ F_vec_rtn
            F_NGA[i] = F_vec_inertial

            p_nga.ax += F_vec_inertial[0]
            p_nga.ay += F_vec_inertial[1]
            p_nga.az += F_vec_inertial[2]

        sim_nga.additional_forces = NGA
        sim_nga.integrate(time)
        cartesian_NGA[i] = [p_nga.x, p_nga.y, p_nga.z, p_nga.vx, p_nga.vy, p_nga.vz]
        kepler_NGA[i] = [p_nga.a, p_nga.e, np.rad2deg(p_nga.inc), np.rad2deg(p_nga.omega), np.rad2deg(p_nga.Omega), np.rad2deg(p_nga.theta)]

    # compute differences
    cartesian_diff = Util.compute_difference(cartesian,cartesian_NGA)
    kepler_diff = Util.compute_difference(kepler,kepler_NGA)
    r_norm = np.linalg.norm(cartesian[:,:3],axis =1)

    Util.plot_difference(cartesian_diff, r_norm,body, c=True)
    Util.plot_difference(kepler_diff, r_norm,body, c=False)
    Util.plot_NGA_acc(F_NGA,r_norm,body)
    Util.plot_NGA_acc_MJD(F_NGA,times,body,T_perihelium_mjd)

    