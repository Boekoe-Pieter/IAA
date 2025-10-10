# load python libraries
import time
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
                   },
        },
    }
}

# ------------------------------------
# Info
data = 'Coding/W2P1/unfiltered.txt'
body = 'C/2001 Q4'
primary = "sun"
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

correctors = [0,3,5,7,11]
method = "reference"
# ------------------------------------
# Retrieving data
comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)
current_comet_data = data_dict[body]["Full"]["n5"]  
arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)

start_time = arc1_mjd
end_time = T_perihelium_mjd 
JD = start_time + 2400000.5
JD_str = 'JD'+str(JD)
integrator = 'trace'
N_steps = 10000
times_to_test = np.linspace(start_time, end_time, N_steps)
print((end_time-start_time)/10000,(end_time-start_time)/1000)
# ------------------------------------
# Create sim
def make_sim():
    sim = rebound.Simulation()
    sim.units = ('Days', 'AU', 'Msun')
    sim.t = start_time
    sim.integrator = integrator
    sim.dt = 0.001

    # sim.add(m=1)
    sim.add(primary)
    sim.add(planets, date=JD_str)

    sim.add(
            primary=sim.particles[0],
            m=0.0,
            a=(q) / (1 - ecc),
            e=ecc,
            inc=np.deg2rad(i),
            Omega=np.deg2rad(RAAN),
            omega=np.deg2rad(aop),
            T=T_perihelium_mjd,
            hash='comet'
        )
    
    sim.move_to_com()
    
    return sim

# ------------------------------------
# run integration
def run_integration(sim_test, times):
    traj = np.zeros((len(times), 3))
    def NGA(reb_sim):
        r_vec = np.array([sim_test.particles['comet'].x, sim_test.particles['comet'].y, sim_test.particles['comet'].z])
        r_norm = np.linalg.norm(r_vec)
        v_vec = np.array([sim_test.particles['comet'].vx, sim_test.particles['comet'].vy, sim_test.particles['comet'].vz])

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

        sim_test.particles['comet'].ax += F_vec_inertial[0]
        sim_test.particles['comet'].ay += F_vec_inertial[1]
        sim_test.particles['comet'].az += F_vec_inertial[2]
    
    sim_test.additional_forces = NGA
    for j, t in enumerate(times):
        sim_test.integrate(t,exact_finish_time=True)
        p = sim_test.particles['comet'] 
        traj[j] = [p.x*const.au, p.y*const.au, p.z*const.au]
    return traj

# ------------------------------------
# test different timesteps 
sim = make_sim()

timesteps = np.logspace(-3, 1, 24)
results = []

ref_sim = sim.copy()
traj_ref = run_integration(ref_sim, times_to_test)
traj_ref_norm = np.linalg.norm(traj_ref, axis = 1)
for dt in timesteps:
    sim_cor = sim.copy()
    # sim_cor.ri_whfast.corrector = 3
    sim_cor.dt = dt
    start = time.perf_counter()
    traj_dt = run_integration(sim_cor, times_to_test)
    end = time.perf_counter()

    runtime = end-start

    # idx_1AU = np.argmin(np.abs(traj_ref_norm - 1.0))
    # diffs = traj_dt[idx_1AU] - traj_ref[idx_1AU]
    # err_1au = np.linalg.norm(diffs)
    err = max(np.linalg.norm(traj_dt-traj_ref, axis=1))
    results.append({
        "dt": dt,
        "error": err,
        "Runtime": runtime
    })

    print(f"cor={dt:.3e}")

# ------------------------------------
# Plot error vs timestep
dts = [r["dt"] for r in results]
errs = [r["error"] for r in results]
runtimes = [r["Runtime"] for r in results]

fig, ax1 = plt.subplots(figsize=(15,9))

color = 'red'
ax1.set_xlabel("Timestep [days]")
ax1.set_ylabel("Maximum position error [m]", color=color)
ax1.plot(dts, errs, 'o-', color=color, label="Error")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'blue'
ax2.set_ylabel("Runtime [s]", color=color)
ax2.plot(dts, runtimes, 'o-', color=color, label="Runtime")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.tick_params(axis='y', labelcolor=color)

if method == "reference":
    title = rf"Integrator timstep analysis {integrator}, {N_steps} ||$\epsilon(t;\Delta t)-\epsilon(t;\Delta 0.001)||$"
# else:
#     title = rf"benchmark analysis for {int}, method {method} ||$\epsilon(t;\Delta t)-\epsilon(t;\Delta t/2)||$"
fig.suptitle(title)
ax1.grid(True, which="both")
plt.savefig(f"Coding/W3/Integrator_analysis/beIntegrator timstep analysis {integrator}, {N_steps}.pdf",dpi=300)

plt.show()



# # ------------------------------------
# # Defining time
# start_time = arc1_mjd
# end_time = T_perihelium_mjd 
# N_time = 10000
# times = np.linspace(start_time, end_time, N_time)

# # ------------------------------------
# # Creating benchmark simulator 
# Benchmark_sim = rebound.Simulation()
# Benchmark_sim.units = ('Days', 'AU', 'Msun')

# Benchmark_sim.add(primary)
# JD = start_time + 2400000.5
# JD_str = 'JD'+str(JD)
# Benchmark_sim.add(planets, date = JD_str)
# Benchmark_sim.move_to_com()

# p1 = Benchmark_sim.particles

# Benchmark_sim.t = start_time
# Benchmark_sim.integrator = "whfast"

# Benchmark_sim.add(
#     primary=p1[0],
#     a=(q) / (1 - ecc),
#     e=ecc,
#     inc=i,
#     Omega=RAAN,
#     omega=aop,
#     T=T_perihelium_mjd,
#     hash='mean'
# )


# # ------------------------------------
# # Analyzing benchmark timestep
# seconds_in_day = const.day
# timesteps = []


# for i, time in enumerate(times):
#     def NGA(reb_sim):
#         r_vec = np.array([p1['mean'].x, p1['mean'].y, p1['mean'].z])
#         r_norm = np.linalg.norm(r_vec)
#         v_vec = np.array([p1['mean'].vx, p1['mean'].vy, p1['mean'].vz])

#         NGA_x = current_comet_data['A1'] * 1e-8
#         NGA_y = current_comet_data['A2'] * 1e-8
#         NGA_z = current_comet_data['A3'] * 1e-8
#         A_vec = np.array([NGA_x, NGA_y, NGA_z])

#         m = current_comet_data['m']
#         n = current_comet_data['n']
#         k = current_comet_data['k']
#         r0 = current_comet_data['r0']
#         alpha = current_comet_data['alph']

#         g = alpha * (r_norm/r0)**(-m) * (1 + (r_norm/r0)**n)**(-k)

#         F_vec_rtn = g * A_vec
#         C_rtn2eci = Util.rtn_to_eci(r_vec, v_vec)

#         F_vec_inertial = C_rtn2eci @ F_vec_rtn

#         p1['mean'].ax += F_vec_inertial[0]
#         p1['mean'].ay += F_vec_inertial[1]
#         p1['mean'].az += F_vec_inertial[2]

#     mean_orbit.additional_forces = NGA
#     mean_orbit.integrate(time)

#     sun_positions[i]   = [p1[0].x, p1[0].y, p1[0].z]
#     comet_positions[i] = [p1['mean'].x, p1['mean'].y, p1['mean'].z]

#     for j in range(len(planets)):
#         planet_positions[i, j] = [p1[j+1].x, p1[j+1].y, p1[j+1].z]




# def benchmark_integration(sim, times, label=""):
#     start = time.perf_counter()
#     for t in times:
#         sim.integrate(t)
#     end = time.perf_counter()
#     runtime = end - start
#     print(f"{label} runtime: {runtime:.3f} seconds")
#     return runtime















# safe_name = re.sub(r'[^\w\-_\." "]', '_', body)
# with open(f"Coding/W3P1/Mean_orbit_{safe_name}_NGA.pkl", "wb") as f:
#     pickle.dump(comet_positions, f)



