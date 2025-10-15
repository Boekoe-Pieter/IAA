# loading packages
import numpy as np
import matplotlib.pyplot as plt

# Import systems
import sys
import time as timer
import pickle
import re
import os
import glob
import json
import pprint
import rebound

# load sim packages
from scipy import constants as const
from astropy.time import Time
from tudatpy.data.sbdb import SBDBquery
from tudatpy.astro import time_representation, element_conversion

# Import python files
sys.path.append('/Users/pieter/IAA/Coding')
import Utilities as Util       


np.set_printoptions(linewidth=160)

comets = ['C2001Q4','C2008A1','C2013US10']
base_path = "Pedro Lacerda/orbit_analysis_2033.00-2037.00"


for comet in comets:
    if comet == "C2001Q4":
        state_Vector = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/covar_C2001Q4_2033.00_2035.40.json"
        total_file = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/total_C2001Q4_2033.00_2035.40.json"

    elif comet == "C2008A1":
        state_Vector = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/covar_C2008A1_2033.00_2035.26.json"
        total_file = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/total_C2008A1_2033.00_2035.26.json"
    else:
        state_Vector = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/covar_C2013US10_2033.00_2035.68.json"
        total_file = "Pedro Lacerda/orbit_analysis_2033.00-2037.00/total_C2013US10_2033.00_2035.68.json"

    with open(state_Vector) as f:
        state_Vector_data = json.loads(f.read())
    with open(total_file) as f:
        total_data = json.loads(f.read())
    body = total_data.get("ids")[0]

    # saving directory
    rel_dir = os.path.dirname(os.path.abspath(__file__))
    directory_name = f"{rel_dir}/data_{body}"

    # environment
    primary = "Sun"

    # sim details
    Integrator = "trace"
    timestep = 1

    # saving dictionary 
    data_to_write = {
        "SBDB_reference_orbit":0,
        "Nominal_trajectory_times":0,
        "trajectory_fit":0,

        "Initial_condition": 0,
        "Sampled_data": 0,
    }

    # ---------------------------------
    # Initial conditions
    "Lacerda's"
    state = state_Vector_data.get("state_vect")
    start_time = state_Vector_data.get("epoch")
    Tp = total_data["objects"][body]["elements"].get("Tp_iso").replace("Z", "")
    Tp_mjd = time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.iso_string_to_epoch(str(Tp))))
    Lacerda = state

    "SBDB"
    target_sbdb = SBDBquery(body,full_precision=True)
    e = target_sbdb["orbit"]['elements'].get('e')
    a = target_sbdb["orbit"]['elements'].get('a').value 
    q = target_sbdb["orbit"]['elements'].get('q').value 
    i = np.deg2rad(target_sbdb["orbit"]['elements'].get('i').value)
    om = np.deg2rad(target_sbdb["orbit"]['elements'].get('om').value)
    w = np.deg2rad(target_sbdb["orbit"]['elements'].get('w').value)
    Tp =target_sbdb["orbit"]['elements'].get('tp').value
    SBDB = np.array([e,a,q,i,om,w,Tp_mjd])

    # ---------------------------------
    # Simulator
    def create_sim(primary,start_time,integrator):
        sim = rebound.Simulation()
        sim.units = ('Days', 'AU', 'Msun')

        start_time_JD = time_representation.modified_julian_day_to_julian_day(start_time)
        JD_str = f'JD{start_time_JD}'
        sim.add(primary,date=JD_str)
        
        sim.t = start_time
        sim.integrator = integrator

        return sim
    
    end_time = total_data["objects"][body]["elements"].get("Tp_iso").replace("Z", "")

    start_time_MJD = time_representation.julian_day_to_modified_julian_day(start_time)
    end_time_MJD =  time_representation.julian_day_to_modified_julian_day(time_representation.seconds_since_epoch_to_julian_day(time_representation.iso_string_to_epoch(str(end_time))))

    sim = create_sim(primary,start_time_MJD,Integrator)
    times = np.arange(start_time_MJD, end_time_MJD+timestep, timestep)

    data_to_write['Nominal_trajectory_times'] = times

    # ---------------------------------
    # add mean orbit
    def add_SBDB(ref_sim):
        e,a,q,i,om,w,Tp_mjd = SBDB
        ref_sim.add(
            primary=ref_sim.particles[0],
            m=0.0,
            a=(q) / (1 - e),
            e=e,
            inc=i,
            Omega=om,
            omega=w,
            T=Tp_mjd,
            hash='0'
        )

    add_SBDB(sim)
    data_to_write["SBDB_reference_orbit"] = np.zeros((len(times), 6))

    # ---------------------------------
    # add fitted reference
    def add_fit(ref_sim):
        x,y,z,vx,vy,vz= Lacerda
        ref_sim.move_to_hel()
        ref_sim.add(
            m=0.0,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            hash='fit'
        )

    add_fit(sim)
    
    data_to_write["trajectory_fit"] = np.zeros((len(times), 6))
    print(
        f"Integrating \n"
        f"integrator: {Integrator}\n"
        )
    
    start = timer.perf_counter()
    for j, time in enumerate(times):
        sim.integrate(time)
        p_reference = sim.particles["0"]
        p_reference_fit = sim.particles["fit"]
        data_to_write["SBDB_reference_orbit"][j] = [p_reference.x*const.au, p_reference.y*const.au, p_reference.z*const.au,p_reference.vx*const.au/const.day, p_reference.vy*const.au/const.day, p_reference.vz*const.au/const.day]
        data_to_write["trajectory_fit"][j] = [p_reference_fit.x*const.au, p_reference_fit.y*const.au, p_reference_fit.z*const.au,p_reference_fit.vx*const.au/const.day, p_reference_fit.vy*const.au/const.day, p_reference_fit.vz*const.au/const.day]

    end = timer.perf_counter()
    runtime = end-start
    
    def diff_orbit_fits(data):
        JPL_elements = data["SBDB_reference_orbit"]
        Fit_elements = data["trajectory_fit"]
        diff_elements = JPL_elements - Fit_elements

        diff = np.linalg.norm(diff_elements[:,:3],axis=1)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        labels = ['x (km)', 'y (km)', 'z (km)']
        for i in range(3):
            axs[i].plot(data["Nominal_trajectory_times"]-data["Nominal_trajectory_times"][-1], diff_elements[:, i] / 1e3, color='tab:blue')
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)

        axs[2].set_xlabel('Time [MJD]')

        distance_sorted = sorted(np.linalg.norm(JPL_elements,axis=1) / const.au, reverse=True)
        axs[3].plot(distance_sorted, diff / 1e3, color='tab:orange')
        axs[3].set_ylabel(r'$||r_{diff}||$ (km)')
        axs[3].grid(True)
        axs[3].set_xlabel('Distance [AU]')
        axs[3].invert_xaxis()

        fig.suptitle(f'{comet} - Difference between cartesian state fit and JPL fitted')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"Pedro Lacerda/{comet}/Cartesian_Fit_vs_SBDB.pdf",dpi=300)
        plt.show()

    diff_orbit_fits(data_to_write)

    def D_traject(cartesian, ax=None, label=None, color=None, alpha=1.0,style=None):
        if ax is None:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal', adjustable='box')

        ax.plot(cartesian[:, 0]/const.au, cartesian[:, 1]/const.au, cartesian[:, 2]/const.au, 
                label=label, color=color, alpha=alpha,linestyle=style)
        ax.scatter(cartesian[0, 0]/const.au, cartesian[0, 1]/const.au, cartesian[0, 2]/const.au, marker="o", s=10, color=color)
        ax.scatter(cartesian[-1, 0]/const.au, cartesian[-1, 1]/const.au, cartesian[-1, 2]/const.au, marker="x", s=10, color=color)

        return ax

    def plot_ensemble(data):
        def draw_sun(ax):
            radius_sun = 696340 * 1000/const.au
            _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
            _x = radius_sun * np.cos(_u) * np.sin(_v)
            _y = radius_sun * np.sin(_u) * np.sin(_v)
            _z = radius_sun * np.cos(_v)
            ax.plot_wireframe(_x, _y, _z, color="orange", alpha=0.5, lw=0.5, zorder=0)


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal", adjustable="box")
        
        D_traject(np.array(data["SBDB_reference_orbit"]), ax=ax, label='NASA JPL API', color="black", alpha=0.9,style="-")
        D_traject(np.array(data["trajectory_fit"]), ax=ax, label='Fitted', color="red", alpha=0.9,style=':')

        draw_sun(ax)

        max_value = max(np.max(np.abs(np.array(traj))) for traj in data["SBDB_reference_orbit"])
        ax.set_xlim([-max_value/const.au, max_value/const.au])
        ax.set_ylim([-max_value/const.au, max_value/const.au])
        ax.set_zlim([-max_value/const.au, max_value/const.au])
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_zlabel("z [AU]")

        ax.legend()
        plt.show()

    plot_ensemble(data_to_write)