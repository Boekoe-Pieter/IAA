# loading packages
import numpy as np
import rebound
import reboundx 
import pickle
import re
from scipy import constants as const
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# load python files
import Utilities as Util

np.set_printoptions(linewidth=160)

NGA_data = {
    'C/2001 Q4': {
        "Full": {   
            "n5": {"A1": 1.6506, "A2": 0.062406, "A3": 0.001412,
                   'm': 2.15, 'n': 5.093, 'k': 4.6142,
                   'r0': 2.808, 'alph': 0.1113, "tau": 0,

                   'e_sig':0.00000079, 'q_sig':0.00000034, 'q_T_sig': 0.00002320,
                   'om_sig': 0.000050, 'Om_sig': 0.000008, 'i_sig':0.000011,
                   "A1_sig":0.014, "A2_sig":0.006102, "A3_sig":0.00511
                   },
        },
    }
}

#Om = RAAN
#om = AoP

data = 'Coding/W2P1/unfiltered.txt'

body = 'C/2001 Q4'
primary = "sun"
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)

current_comet_data = data_dict[body]["Full"]["n5"]  

arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)

sigmas = np.array([
    current_comet_data['e_sig'],
    current_comet_data['q_sig'],
    current_comet_data['q_T_sig'],
    np.deg2rad(current_comet_data['Om_sig']),
    np.deg2rad(current_comet_data['om_sig']),
    np.deg2rad(current_comet_data['i_sig']),
    current_comet_data['A1_sig'],
    current_comet_data['A2_sig'],
    current_comet_data['A3_sig'],
])

covariance = np.diag(sigmas**2)

N_time = 10000

mean_conditions = np.array([T_perihelium_mjd,q, ecc, np.deg2rad(RAAN), np.deg2rad(aop), np.deg2rad(i), 1/a_recip])
mean_to_sample = np.array([ecc,q, T_perihelium_mjd, np.deg2rad(RAAN), np.deg2rad(aop), np.deg2rad(i),
                           current_comet_data['A1'], current_comet_data['A2'], current_comet_data['A3']])

mean_orbit = rebound.Simulation()
mean_orbit.units = ('Days', 'AU', 'Msun')

start_time = arc1_mjd
end_time = T_perihelium_mjd 

mean_orbit.add(primary)
JD = start_time + 2400000.5
JD_str = 'JD'+str(JD)
mean_orbit.add(planets, date = JD_str)
mean_orbit.move_to_com()

p1 = mean_orbit.particles

mean_orbit.t = start_time
mean_orbit.integrator = "trace"
mean_orbit.dt = 1
times = np.linspace(start_time, end_time, N_time)

sun_positions     = np.zeros((len(times), 3))
planet_positions  = np.zeros((len(times), len(planets), 3))
comet_positions   = np.zeros((len(times), 3))

print('Integrating mean sim')
mean_orbit.add(
    primary=p1[0],
    m=0.0,
    a=(q) / (1 - ecc),
    e=ecc,
    inc=i,
    Omega=RAAN,
    omega=aop,
    T=T_perihelium_mjd,
    hash='mean'
)
mean_orbit.status()

for i, time in enumerate(times):
    def NGA(reb_sim):
        r_vec = np.array([p1['mean'].x, p1['mean'].y, p1['mean'].z])
        r_norm = np.linalg.norm(r_vec)
        v_vec = np.array([p1['mean'].vx, p1['mean'].vy, p1['mean'].vz])

        NGA_x = current_comet_data['A1'] * 1e-8
        NGA_y = current_comet_data['A2'] * 1e-8
        NGA_z = current_comet_data['A3'] * 1e-8
        A_vec = np.array([NGA_x, NGA_y, NGA_z])

        m = current_comet_data['m']
        n = current_comet_data['n']
        k = current_comet_data['k']
        r0 = current_comet_data['r0']
        alpha = current_comet_data['alph']

        g = alpha * (r_norm/r0)**(-m) * (1 + (r_norm/r0)**n)**(-k)

        F_vec_rtn = g * A_vec
        C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

        F_vec_inertial = C_rtn2eci @ F_vec_rtn

        p1['mean'].ax += F_vec_inertial[0]
        p1['mean'].ay += F_vec_inertial[1]
        p1['mean'].az += F_vec_inertial[2]

    mean_orbit.additional_forces = NGA
    mean_orbit.integrate(time)

    sun_positions[i]   = [p1[0].x, p1[0].y, p1[0].z]
    comet_positions[i] = [p1['mean'].x, p1['mean'].y, p1['mean'].z]

    for j in range(len(planets)):
        planet_positions[i, j] = [p1[j+1].x, p1[j+1].y, p1[j+1].z]

N_samples = 500
samples = np.random.multivariate_normal(mean_to_sample, covariance, size=N_samples)

print("creating perturbed sim")
positions_perturbed = {}
perturbed_orbit = rebound.Simulation()
perturbed_orbit.units = ('Days', 'AU', 'Msun')
perturbed_orbit.t = start_time
perturbed_orbit.integrator = "trace"

perturbed_orbit.add(primary)
perturbed_orbit.add(planets, date=JD_str)
perturbed_orbit.move_to_com()
perturbed_orbit.dt = 1

for idx, s in enumerate(samples):

    
    ecc_s, q_s, q_date, RAAN_s, aop_s, i_s, A1_s, A2_s, A3_s = s
    perturbed_orbit.add(
        primary=perturbed_orbit.particles[0],
        m=0.0,
        a=q_s / (1 - ecc_s),
        e=ecc_s,
        inc=i_s,
        Omega=RAAN_s,
        omega=aop_s,
        T=q_date,
        hash=f"clone_{idx}"
    )
    positions_perturbed[idx] = np.zeros((len(times), 3)) 
perturbed_orbit.status()

def NGA(reb_sim):
    for idx, s in enumerate(samples):
        _, _, _, _, _, _, A1_s, A2_s, A3_s = s
        p_clone = perturbed_orbit.particles[f"clone_{idx}"]

        r_vec = np.array([p_clone.x, p_clone.y, p_clone.z])
        r_norm = np.linalg.norm(r_vec)
        v_vec = np.array([p_clone.vx, p_clone.vy, p_clone.vz])

        NGA_x = A1_s * 1e-8
        NGA_y = A2_s * 1e-8
        NGA_z = A3_s * 1e-8
        A_vec = np.array([NGA_x, NGA_y, NGA_z])

        m = current_comet_data['m']
        n = current_comet_data['n']
        k = current_comet_data['k']
        r0 = current_comet_data['r0']
        alpha = current_comet_data['alph']

        g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)

        F_vec_rtn = g * A_vec
        C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)
        F_vec_inertial = C_rtn2eci @ F_vec_rtn

        p_clone.ax += F_vec_inertial[0]
        p_clone.ay += F_vec_inertial[1]
        p_clone.az += F_vec_inertial[2]
perturbed_orbit.additional_forces = NGA

print("Integrating perturbed sim")
for j, time in enumerate(times):
    for idx in range(len(samples)):
        p_clone = perturbed_orbit.particles[f"clone_{idx}"]
        positions_perturbed[idx][j] = [p_clone.x, p_clone.y, p_clone.z]
    perturbed_orbit.integrate(time)

print("done")

safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
with open(f"Coding/W3P1/Mean_orbit_{safe_name}_NGA.pkl", "wb") as f:
    pickle.dump(comet_positions, f)
with open(f"Coding/W3P1/Monte_perturbed_orbits_{safe_name}_NGA.pkl", "wb") as f:
    pickle.dump(positions_perturbed, f)
with open(f"Coding/W3P1/time_{safe_name}_NGA.pkl", "wb") as f:
    pickle.dump(times, f)
with open(f"Coding/W3P1/Planets_{safe_name}_NGA.pkl", "wb") as f:
    pickle.dump(planet_positions, f)




