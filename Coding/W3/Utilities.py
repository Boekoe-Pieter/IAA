# import python libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# simulation libraries
import rebound
import reboundx
from astropy.time import Time, TimeDelta
from scipy import constants as const

plt.rcParams.update({
        "font.size": 14,              # Base font size
        "axes.titlesize": 14,         # Title font size
        "axes.labelsize": 14,         # X/Y label font size
        "xtick.labelsize": 12,        # X tick label size
        "ytick.labelsize": 12,        # Y tick label size
        "legend.fontsize": 12,        # Legend font size
        "figure.titlesize": 18        # Figure title size (if using suptitle)
    })

def get_data(data_file):
    KEYS = [
        'designation', 'model', 'nobs', 'arc1', 'arc2', 'epoch',
        'T', 'q', 'e', 'omega', 'Omega', 'i', 'recip(a)'
    ]
    comet_dict = {}

    with open(data_file, "r") as f:
        next(f)  
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue 

            designation = f"{parts[0]} {parts[1]}".strip()
            model = parts[2].strip()

            key = (designation, model)
            comet_dict[key] = {k: v for k, v in zip(KEYS, [designation] + parts[2:])}

    return comet_dict

def select_comet(comet_dict, NGA_data):
    result = {}

    for designation, designators in NGA_data.items():    
        for designator, models in designators.items():    
            for model, NGA_vals in models.items():        
                key = (designation, model)

                entry = comet_dict[key].copy()
                entry.update(NGA_vals)
                entry["designator"] = designator 

                if designation not in result:
                    result[designation] = {}
                if designator not in result[designation]:
                    result[designation][designator] = {}

                result[designation][designator][model] = entry

    return result

def create_sim(primary,start_time,integrator,dt):
    sim = rebound.Simulation()
    sim.units = ('Days', 'AU', 'Msun')
    sim.add(primary)
    sim.t = start_time
    sim.integrator = integrator

    sim.dt = dt
    return sim

def add_NBP(sim,start_time,planets):
    JD = start_time + 2400000.5
    JD_str = 'JD'+str(JD)
    for planet in planets:
        sim.add(planet, date = JD_str)

def add_Rel(sim):
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    rebx.add_force(gr)
    c_au_per_day = const.c * const.day / const.au
    gr.params["c"] = c_au_per_day

def get_orbital_elements(data):
    arc1_calender = data['arc1']           # yyyymmdd
    arc1_mjd = calender_to_MDJ(arc1_calender)   # MJD
    arc2_calender = data['arc2']           # yyyymmdd
    arc2_mjd = calender_to_MDJ(arc2_calender)   # MJD
    epoch_calender = data['epoch']         # yyyymmdd
    epoch_mjd = calender_to_MDJ(epoch_calender) # MJD
    T_perihelium_calender = data['T']      # yyyymmddf
    T_perihelium_mjd = calender_to_MDJ(T_perihelium_calender)      

    q = float(data['q'])               # AU
    ecc = float(data['e'])                        # -
    aop = float(data['omega'])                    # deg
    RAAN = float(data['Omega'])                   # deg
    i = float(data['i'])                          # deg
    a_recip = float(data['recip(a)'])*1e-6        # 1/a 10^-6
    return arc1_mjd, arc2_mjd, epoch_mjd, T_perihelium_mjd, q, ecc, aop, RAAN, i, a_recip

def calender_to_MDJ(date):
    """
    definition to convert the YYYYMMDD in the codec file to MJD
    input:
        - EPOCH: YYYYMMDD
        - T_perihelium: YYYYMMDDF
    output
        - EPOCH: MJD
        - T_periheliumL MJD
    """
    T_date_str = date[:8]
    T_frac_day = float('0' + date[8:])
    T_iso = f"{T_date_str[:4]}-{T_date_str[4:6]}-{T_date_str[6:]}"
    T_time = Time(T_iso, format='iso') + TimeDelta(T_frac_day, format='jd')
    T_mjd = T_time.mjd

    return T_mjd

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

