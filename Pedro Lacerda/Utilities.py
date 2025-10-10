# import python libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# simulation libraries
import rebound
import reboundx
from astropy.time import Time, TimeDelta
from scipy import constants as const

def create_sim(primary,start_time,integrator,timestep):
    sim = rebound.Simulation()
    sim.units = ('Days', 'AU', 'Msun')

    start_time_JD = start_time + 2400000.5
    JD_str = f'JD{start_time_JD}'
    sim.add(primary,date=JD_str)
    
    sim.t = start_time
    sim.integrator = integrator
    sim.dt = timestep

    return sim

def add_NBP(sim, start_time, planets):
    start_time_JD = start_time + 2400000.5
    JD_str = f'JD{start_time_JD}'
    for planet in planets:
        sim.add(planet,date=JD_str)

def add_Rel(sim):
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    rebx.add_force(gr)
    c_au_per_day = const.c * const.day / const.au
    gr.params["c"] = c_au_per_day

def rtn_to_eci(r_vec, v_vec):
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    t_hat /= np.linalg.norm(t_hat)
    return np.column_stack((r_hat, t_hat, n_hat))

def compute_difference_NGA(arc,models,data_PG,data_NGA):
    diff = {}
    for model in models:
        diff[model] = data_NGA[arc][model] - data_PG[arc][model]
    return diff

