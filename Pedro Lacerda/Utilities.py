# import python libraries
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tudatpy.data.sbdb import SBDBquery
from tudatpy.astro import time_representation
import pandas as pd
import re

# simulation libraries
import rebound
import reboundx
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

def extract_traditional_covariance(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    epoch_jd = None
    for line in lines:
        match = re.search(r"epoch JD\s*([0-9]+\.[0-9]+)", line)
        if match:
            epoch_jd = float(match.group(1))
            break

    start_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\bTp\b.*\be\b.*\bq\b.*\bQ\b.*1/a', line):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Covariance matrix header not found in file.")

    data_lines = []
    for line in lines[start_idx + 2:]:
        if line.strip() == "" or re.match(r"^-{5,}|Covariance", line):
            break
        data_lines.append(line.strip())

    labels = []
    matrix_rows = []
    for line in data_lines:
        parts = re.split(r'\s+', line.strip())
        *nums, label = parts
        try:
            nums = [float(x) for x in nums]
        except ValueError:
            continue
        labels.append(label)
        matrix_rows.append(nums)

    C = np.array(matrix_rows)
    df_cov = pd.DataFrame(C, index=labels, columns=labels)
    cov_df = df_cov.drop(index=["MOID", "H", "Q", "M"], columns=["MOID", "H", "Q", "M"])

    return cov_df, epoch_jd


def SBDB(body,Tp_mjd):
    target_sbdb = SBDBquery(body,full_precision=True)
    e = target_sbdb["orbit"]['elements'].get('e')
    a = target_sbdb["orbit"]['elements'].get('a').value 
    q = target_sbdb["orbit"]['elements'].get('q').value 
    i = np.deg2rad(target_sbdb["orbit"]['elements'].get('i').value)
    om = np.deg2rad(target_sbdb["orbit"]['elements'].get('om').value)
    w = np.deg2rad(target_sbdb["orbit"]['elements'].get('w').value)
    Tp = target_sbdb["orbit"]['elements'].get('tp').value
    SBDB = np.array([e,a,q,i,om,w,Tp_mjd])
    return SBDB

def initial_conditions(total_data,body):
    e = total_data["objects"][body]["elements"].get("e")
    a = total_data["objects"][body]["elements"].get("a")
    q = total_data["objects"][body]["elements"].get("q")
    i = np.deg2rad(total_data["objects"][body]["elements"].get("i"))
    w = np.deg2rad(total_data["objects"][body]["elements"].get("arg_per"))
    om = np.deg2rad(total_data["objects"][body]["elements"].get("asc_node"))
    Tp = total_data["objects"][body]["elements"].get("Tp_iso").replace("Z", "")
    Tp_mjd = time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.iso_string_to_epoch(str(Tp))))
    Lacerda = np.array([e,a,q,i,om,w,Tp_mjd])
    return Lacerda