# import python libraries
import numpy as np
from astropy.time import Time, TimeDelta
import rebound
import reboundx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
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
    """
    This function gives the data for each called comet to simulate.
    It creates an dictionary of the given data_file for the keys, the rate of change in orbital elements have been left out.
    inputs:
        data_file: string to the file of the Catalogue of Cometary Orbits and their Dynamical Evolution
        body: current comet that is being simulated
        NGA_data_body: current comet NGA values
    output:
        dict: keys for 'designation', 'model', 'nobs', 'arc1', 'arc2', 'epoch', 'T', 'q', 'e', 'omega', 'Omega', 'i', 'recip(a)', A1, A2, A3
    """
    KEYS = [
        'designation', 'model', 'nobs', 'arc1', 'arc2', 'epoch',
        'T', 'q', 'e', 'omega', 'Omega', 'i', 'recip(a)'
    ]
    data_rows = []
    with open(data_file, "r") as f:
        next(f)
        for line in f:
            parts = line.split()[:14]
            data_rows.append(parts)

    comet_dict = {}
    for row in data_rows:
        designation = row[0] +' '+ row[1]
        comet_dict[designation] = {k: v for k, v in zip(KEYS, [designation] + row[2:])}

    return comet_dict

def add_NGA(comet_dict,NGA_data_body):
    KEYS = [
        'designation', 'model', 'nobs', 'arc1', 'arc2', 'epoch',
        'T', 'q', 'e', 'omega', 'Omega', 'i', 'recip(a)'
    ]
    data_rows = []
    for row in data_rows:
        designation = row[0] +' '+ row[1]
        comet_dict[designation] = {k: v for k, v in zip(KEYS, [designation] + row[2:])}
        comet_dict[designation].update(NGA_data_body)
    return comet_dict

def get_orbital_elements(data):
    arc1_calender = data['arc1']           # yyyymmdd
    arc1_mjd = calender_to_MDJ(arc1_calender)   # MJD
    arc2_calender = data['arc2']           # yyyymmdd
    arc2_mjd = calender_to_MDJ(arc2_calender)   # MJD
    epoch_calender = data['epoch']         # yyyymmdd
    epoch_mjd = calender_to_MDJ(epoch_calender) # MJD
    T_perihelium_calender = data['T']      # yyyymmddf
    T_perihelium_mjd = calender_to_MDJ(T_perihelium_calender)      

    q = data['q']               # AU
    ecc = data['e']                        # -
    aop = data['omega']                    # deg
    RAAN = data['Omega']                   # deg
    i = data['i']                          # deg
    a_recip = data['recip(a)']             # 1/a 10^-6
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

def MJD_to_Calendar(mjd):
    t = Time(mjd, format='mjd')
    return t.iso

def create_sim(integrator,start_time):
    """
    definition to create the Sim evironment
    input: 
        - integrator
        - primary

    output:
        - sim
    """
    # initialize sim
    sim = rebound.Simulation()
    sim.units = ('Days', 'AU', 'Msun')

    # add primary body
    primary = rebound.Particle(m=1.)
    sim.add(primary)

    # add integrator
    sim.integrator = integrator 
    sim.t = start_time

    return sim

def eci_to_perif(raan,aop,i):
    row0=[-np.sin(raan)*np.cos(i)*np.sin(aop)+np.cos(raan)*np.cos(aop),np.cos(raan)*np.cos(i)*np.sin(aop)+np.sin(raan)*np.cos(aop),np.sin(i)*np.sin(aop)]
    row1=[-np.sin(raan)*np.cos(i)*np.cos(aop)-np.cos(raan)*np.sin(aop),np.cos(raan)*np.cos(i)*np.cos(aop)-np.sin(raan)*np.sin(aop),np.sin(i)*np.cos(aop)]
    row2=[np.sin(raan)*np.sin(i),-np.cos(raan)*np.sin(i),np.cos(i)]
    return np.array([row0,row1,row2])

def rtn_to_eci(r_vec, v_vec):
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    t_hat /= np.linalg.norm(t_hat)
    return np.column_stack((r_hat, t_hat, n_hat))

def plot_elements(sim,times,kepler,cartesian,body):
    print("plotting elements")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  
    labels = ["x [AU]", "y [AU]", "z [AU]", r"$v_x$ [AU/day]", r"$v_y$ [AU/day]", r"$v_z$ [AU/day]"]
    # time_labels = MJD_to_Calendar(times)

    for j in range(6):
        axes[j].plot(times, cartesian[:,j])
        axes[j].scatter(times[0], cartesian[0, j], label='start')
        axes[j].scatter(times[-1], cartesian[-1, j], label='stop')
        axes[j].set_ylabel(labels[j])
        axes[j].grid(True)
        axes[j].legend()
        if j>=3:
            axes[j].set_xlabel("Time [MJD]")

    fig.suptitle(f"Cartesian state evolution of {body}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_Cartesian_state_evolution_of.pdf",dpi=300)
    plt.show()

    # plot Kepler elements
    labels = [r"$a$ [AU]", 
            r"$e$ [-]", 
            r"$i$ [deg]", 
            r"$\omega$ [deg]", 
            r"$\Omega$ [deg]", 
            r"$\theta$ [deg]"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for j in range(6):
        axes[j].plot(times, kepler[:,j])
        axes[j].scatter(times[0], kepler[0, j], label='start')
        axes[j].scatter(times[-1], kepler[-1, j], label='stop')
        axes[j].legend()
        axes[j].set_ylabel(labels[j])
        axes[j].grid(True)
        if j>=3:
            axes[j].set_xlabel("Time [MJD]")

    fig.suptitle(f"Keplerian elements of {body}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_Keplerian_elements.pdf",dpi=300)
    plt.show()

    # plot 3D orbit
    fig=plt.figure(figsize=(15, 8))
    ax=fig.add_subplot(111,projection='3d')
    ax.set_aspect('equal', adjustable='box')

    ax.plot(cartesian[:,0],cartesian[:,1],cartesian[:,2],label=f'{body}') 
    ax.scatter(cartesian[0,0],cartesian[0,1],cartesian[0,2],label='Start')
    ax.scatter(cartesian[-1,0],cartesian[-1,1],cartesian[-1,2],label='End')


    # radius_sun = consts.sun['radius']*1000/const.au # in AU
    # _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
    # _x = radius_sun*np.cos(_u)*np.sin(_v)
    # _y = radius_sun*np.sin(_u)*np.sin(_v)
    # _z = radius_sun*np.cos(_v)
    # ax.plot_wireframe(_x,_y,_z,color="r",alpha=0.5,lw=0.5,zorder=0)

    max_value=np.max(np.abs(cartesian))
    ax.set_xlim([-max_value,max_value])
    ax.set_ylim([-max_value,max_value])
    ax.set_zlim([-max_value,max_value])

    ax.set_xlabel('x,AU')
    ax.set_ylabel('y,AU')
    ax.set_zlabel('z,AU')

    plt.title(f"3D trajectory of {body}")
    plt.legend()
    plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_3D_trajectory.pdf",dpi=300)
    plt.show()

def compute_difference(nominal,perturbed):
    difference = nominal[:,:] - perturbed[:,:]
    return difference

def plot_difference(states, x_scale, body, c=False):
    if c:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        labels = [r"$\Delta{x}$ [km]", r"$\Delta{y}$ [km]", r"$\Delta{z}$ [km]", r"$\Delta{v_x}$ [km/s]", r"$\Delta{v_y}$ [km/s]", r"$\Delta{v_z}$ [km/s]"]

        states_scaled = np.zeros_like(states)
        states_scaled[:,:3] = states[:,:3]*const.au/1000
        states_scaled[:,3:] = states[:,3:]*const.au/(const.day*1000)

        for j in range(6):
            axes[j].plot(x_scale, states_scaled[:,j])
            axes[j].set_ylabel(labels[j])
            axes[j].scatter(x_scale[0], states_scaled[0, j], label='start')
            axes[j].scatter(x_scale[-1], states_scaled[-1, j], label='stop')
            axes[j].legend()
            axes[j].set_xscale("log")
            # axes[j].set_yscale("log")
            axes[j].grid(True,which='minor')

            if j>=3:
                axes[j].set_xlabel("Heliocentric distance [AU]")

        fig.suptitle(f"Cartesian difference between gravity fitted and NGA fitted of {body}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_Cartesian_state_difference.pdf",dpi=300)
        plt.show()

        diff_norm = np.linalg.norm(states_scaled[:, :3],axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        ax.plot(x_scale, diff_norm, label='trajectory')
        ax.scatter(x_scale[0], diff_norm[0], label='start')
        ax.scatter(x_scale[-1], diff_norm[-1], label='stop')
        ax.set_xlabel('Heliocentric distance [AU]')
        ax.set_ylabel(r'$||\Delta{r}||$ [km]')
        ax.grid(True, which="both")
        ax.set_xscale("log")
        # ax.set_yscale("log")

        x_min = min(x_scale)
        x_zoom_min = 0.98 * x_min
        x_zoom_max = 1.02 * x_min
        mask = (x_scale >= x_zoom_min) & (x_scale <= x_zoom_max)
        y_zoom_min = np.min(diff_norm[mask])
        y_zoom_max = np.max(diff_norm[mask])

        axins = inset_axes(ax, width="25%", height="25%",borderpad=6)  
        axins.plot(x_scale, diff_norm, label='trajectory')
        axins.scatter(x_scale[0], diff_norm[0])
        axins.scatter(x_scale[-1], diff_norm[-1])

        axins.set_xlim(x_zoom_min, x_zoom_max)
        axins.set_ylim(y_zoom_min, y_zoom_max)
        axins.grid(True, which="minor")
        axins.set_title(f"zoomed in at Perihelium")

        ax.legend()
        fig.suptitle(f"Cartesian difference norm between gravity fitted and NGA fitted of {body}")
        plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_Cartesian_state_difference_norm.pdf",dpi=300)
        plt.show()

    else:
        labels = [
            r"$\Delta{a}$ [AU]", 
            r"$\Delta{e}$ [-]", 
            r"$\Delta{i}$ [deg]", 
            r"$\Delta{\omega}$ [deg]", 
            r"$\Delta{\Omega}$ [deg]", 
            r"$\Delta{\theta}$ [deg]"
            ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for j in range(6):
            axes[j].plot(x_scale, states[:,j])
            axes[j].scatter(x_scale[0], states[0, j], label='start')
            axes[j].scatter(x_scale[-1], states[-1, j], label='stop')
            axes[j].set_ylabel(labels[j])
            axes[j].set_xscale("log")
            # axes[j].set_yscale("log")
            axes[j].grid(True,which='minor')
            axes[j].legend()
            if j>=3:
                axes[j].set_xlabel("Heliocentric distance [AU]")

        fig.suptitle(f"Keplerian difference between gravity fitted and NGA fitted of {body}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_Keplerian_state_difference.pdf",dpi=300)
        plt.show()   

def plot_NGA_acc(F_NGA,x_scale,body):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes = axes.flatten()
    labels = [r'$a_x$ $[m/s^2]$',r'$a_y$ $[m/s^2]$',r'$a_z$ $[m/s^2]$']

    states_scaled = np.zeros_like(F_NGA)
    states_scaled[:,:] = F_NGA[:,:]*const.au/(const.day**2)

    for j in range(3):
        axes[j].plot(x_scale, states_scaled[:,j])
        axes[j].scatter(x_scale[0], states_scaled[0, j], label='start')
        axes[j].scatter(x_scale[-1], states_scaled[-1, j], label='stop')
        axes[j].set_ylabel(labels[j])
        axes[j].set_xscale("log")
        axes[j].grid(True,which='minor')
        axes[j].legend()
        axes[j].set_xlabel("Heliocentric distance [AU]")

    fig.suptitle(f"NGA components of {body}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_NGA_components_AU.pdf",dpi=300)
    plt.show()   

def plot_NGA_acc_MJD(F_NGA,x_scale,body,MJD_perih):
    fig, axes = plt.subplots(1, 4, figsize=(15, 8))
    axes = axes.flatten()
    labels = [r'$a_x$ $[m/s^2]$',r'$a_y$ $[m/s^2]$',r'$a_z$ $[m/s^2]$']

    states_scaled = np.zeros_like(F_NGA)
    states_scaled[:,:] = F_NGA[:,:]*const.au/(const.day**2)
    x_scaled = x_scale-MJD_perih
    for j in range(3):
        axes[j].plot(x_scaled, states_scaled[:,j])
        axes[j].set_ylabel(labels[j])
        axes[j].grid(True,which='both')
        axes[j].legend()
        axes[j].set_xlabel("MJD (centered at perihelium)")
        axes[j].set_xlim([-20,x_scaled[-1]])
    
    axes[3].plot(x_scaled, np.linalg.norm(states_scaled, axis=1))
    axes[3].grid(True, which='both')
    axes[3].legend()
    axes[3].set_xlabel("MJD (centered at perihelium)")
    axes[3].set_xlim([-20, x_scaled[-1]])
    axes[3].set_ylabel(r"$||a||$ $[m/s^2]$")

    fig.suptitle(f"NGA components of {body}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"Coding/W1P1/{re.sub(r"[^\w\-_\. ]", "_", body)}_NGA_components_MJD.pdf",dpi=300)
    plt.show()   