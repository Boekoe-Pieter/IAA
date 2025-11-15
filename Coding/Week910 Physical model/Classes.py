# Simulator libraries
import rebound
import reboundx
from scipy import constants as const
from scipy.stats import norm
from astropy.time import Time

# load in technical libraries
import requests
import json
import base64
import sys
import contextlib
import os
import sys

# Data management libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams.update({
        "font.size": 14,              # Base font size
        "axes.titlesize": 14,         # Title font size
        "axes.labelsize": 14,         # X/Y label font size
        "xtick.labelsize": 12,        # X tick label size
        "ytick.labelsize": 12,        # Y tick label size
        "legend.fontsize": 12,        # Legend font size
        "figure.titlesize": 18        # Figure title size (if using suptitle)
    })

class Simulator:
    def __init__(self, Primary, Integrator, Timestep, Comet_data, datarate):
        # Comet data
        self.Comet_data = Comet_data

        A1,A2,A3,DT = self.Comet_data[7:11]
        self.A_vec = np.array([
            float(A1) if A1 is not None else 0,
            float(A2) if A2 is not None else 0,
            float(A3) if A3 is not None else 0,
            float(DT) if DT is not None else 0,
            ])

        # Sim specific
        self.Primary = Primary
        self.Integrator = Integrator
        self.Timestep = Timestep

        self.datarate = datarate

        self.data_array = np.zeros((self.datarate,3)) #a,om,w
        self.trajectory = np.zeros((self.datarate,6)) #x,y,z,vx,vy,vz
        self.times = np.zeros((self.datarate,1)) 
        self.times_T_tp = np.zeros((self.datarate,1)) 

        self.trajectory_ASYM = np.zeros((self.datarate,6)) 
        self.gr = np.zeros((self.datarate,1)) 

    def time(self):
        # Time frame
        self.first_obs,self.last_obs,self.tp =  self.Comet_data[11:]
        times = [self.first_obs,self.last_obs]
        t = Time(times, format='fits', scale='utc')

        self.Start_time = t.mjd[0]
        self.End_time = t.mjd[1]

        times = [self.tp]
        t = Time(times, format='jd', scale='utc')
        self.Tp_MJD = t.mjd
        self.Time_array = np.linspace(self.Start_time,self.End_time,self.datarate)

        if self.A_vec[-1] != 0.0:
            self.Start_time_ASYM = self.Start_time - self.A_vec[-1]
            self.End_time_ASYM = self.End_time - self.A_vec[-1]
            self.Time_array_ASYM = np.linspace(self.Start_time_ASYM,self.End_time_ASYM,self.datarate)

    def Create_sim(self):
        # Create Sim
        self.sim = rebound.Simulation()
        self.sim.units = ('Days', 'AU', 'Msun')
        self.time()
        # Add Primary body
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            self.sim.add(self.Primary, date=self.first_obs)

        # Define sim conditions
        self.sim.t = self.Start_time
        self.sim.integrator = self.Integrator
        self.sim.dt = self.Timestep

    def Create_ASYM_sim(self):
        # Create Sim
        self.sim_ASYM = rebound.Simulation()
        self.sim_ASYM.units = ('Days', 'AU', 'Msun')

        # Add Primary body
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            date_str = f"JD{self.Start_time_ASYM+2400000.5}"
            self.sim_ASYM.add(self.Primary, date=date_str)

        # Define sim conditions
        self.sim_ASYM.t = self.Start_time_ASYM
        self.sim_ASYM.integrator = self.Integrator
        self.sim_ASYM.dt = self.Timestep
      
    def add_NGAs(self,reb_sim):

        comet = self.sim.particles[self.full_name]
        if self.A_vec[-1] != 0.0:
            r_vec = np.array(self.trajectory_ASYM[self.i,:3])
            v_vec = np.array(self.trajectory_ASYM[self.i,3:])
        else:
            r_vec = np.array([comet.x, comet.y, comet.z])
            v_vec = np.array([comet.vx, comet.vy, comet.vz])

        r_norm = np.linalg.norm(r_vec)

        m = 2.15
        n = 5.093
        k = 4.6142
        r0 = 2.808
        alpha = 0.1113

        g = alpha * (r_norm / r0) ** (-m) * (1 + (r_norm / r0) ** n) ** (-k)

        self.gr[self.i] = [g]

        F_vec_rtn = g * self.A_vec[:3]
        C_rtn2eci = self.rtn_to_eci(r_vec, v_vec)
        F_vec_inertial = C_rtn2eci @ F_vec_rtn

        comet.ax += F_vec_inertial[0]
        comet.ay += F_vec_inertial[1]
        comet.az += F_vec_inertial[2]

    def rtn_to_eci(self,r_vec, v_vec):
        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        n_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(n_hat, r_hat)
        t_hat /= np.linalg.norm(t_hat)
        return np.column_stack((r_hat, t_hat, n_hat))

    def simulate_2BP(self):
        self.Create_sim()

        full_name,e,a,q,i,om,w,A1,A2,A3,DT = self.Comet_data[:11]
        self.full_name = full_name

        # Add comet
        self.sim.move_to_hel()
        self.sim.add(
            primary=self.sim.particles[0],
            m=0.0,
            a=float(q)/(1.-float(e)),
            e=float(e),
            inc=np.deg2rad(float(i)),
            Omega=np.deg2rad(float(om)),
            omega=np.deg2rad(float(w)),
            T=self.Tp_MJD,
            hash=full_name
        )
        self.sim.move_to_com()

        comet = self.sim.particles[full_name]

        for i,time in enumerate(self.Time_array):
            self.sim.integrate(time)
            self.data_array[i] = [comet.a, comet.Omega, comet.omega]
            self.trajectory[i] = [comet.x, comet.y, comet.z, comet.vx, comet.vy, comet.vz]
            self.times[i] = [time]

    def simulate_marsden(self):
        self.Create_sim()
        
        full_name,e,a,q,i,om,w,A1,A2,A3,DT = self.Comet_data[:11]
        self.full_name = full_name

        # Add comet
        self.sim.move_to_hel()
        self.sim.add(
            primary=self.sim.particles[0],
            m=0.0,
            a=float(q)/(1.-float(e)),
            e=float(e),
            inc=np.deg2rad(float(i)),
            Omega=np.deg2rad(float(om)),
            omega=np.deg2rad(float(w)),
            T=self.Tp_MJD,
            hash=full_name
        )
        self.sim.move_to_com()

        if self.A_vec[-1] != 0.0:
            self.Create_ASYM_sim()
            self.sim.move_to_hel()

            self.sim_ASYM.add(
                primary=self.sim.particles[0],
                m=0.0,
                a=float(q)/(1.-float(e)),
                e=float(e),
                inc=np.deg2rad(float(i)),
                Omega=np.deg2rad(float(om)),
                omega=np.deg2rad(float(w)),
                T=self.Tp_MJD,
                hash=full_name
            )

            comet = self.sim_ASYM.particles[full_name]

            for i,time in enumerate(self.Time_array_ASYM):
                self.sim_ASYM.integrate(time)
                self.data_array[i] = [comet.a, np.rad2deg(comet.Omega), np.rad2deg(comet.omega)]
                self.trajectory_ASYM[i] = [comet.x, comet.y, comet.z, comet.vx, comet.vy, comet.vz]

        comet = self.sim.particles[full_name]
        self.sim.additional_forces = self.add_NGAs
        for i,time in enumerate(self.Time_array):
            self.i = i
            self.sim.integrate(time)
            self.data_array[i] = [comet.a, np.rad2deg(comet.Omega), np.rad2deg(comet.omega)]
            self.trajectory[i] = [comet.x, comet.y, comet.z, comet.vx, comet.vy, comet.vz]
            self.times[i] = [time]
            time_t_tp = time-self.Tp_MJD
            self.times_T_tp[i] = [time_t_tp[0]]

    def interpolate_struct(self,R,T,N,original_time):
        self.original_time = original_time + self.Tp_MJD
        self.rebx = reboundx.Extras(self.sim)
        self.interp_R_struct = reboundx.Interpolator(self.rebx, self.original_time, R, "spline") 
        self.interp_T_struct = reboundx.Interpolator(self.rebx, self.original_time, T, "spline")
        self.interp_N_struct = reboundx.Interpolator(self.rebx, self.original_time, N, "spline")

    def interpolate_RTN(self,reb_sim):
        comet = self.sim.particles[self.full_name]
        time = self.sim.t
        time_check = time 
    
        if(self.original_time[0]<=time_check<=self.original_time[-1]):
            R_interp = self.interp_R_struct.interpolate(self.rebx,time)
            T_interp = self.interp_T_struct.interpolate(self.rebx,time)
            N_interp = self.interp_N_struct.interpolate(self.rebx,time)

            a_RTN = np.array([R_interp,T_interp,N_interp])
            r_vec = np.array([comet.x, comet.y, comet.z])
            v_vec = np.array([comet.vx, comet.vy, comet.vz])
            
            C_rtn2eci = self.rtn_to_eci(r_vec, v_vec)
            F_vec_inertial = C_rtn2eci @ a_RTN

            comet.ax += F_vec_inertial[0]
            comet.ay += F_vec_inertial[1]
            comet.az += F_vec_inertial[2]
                    
    def simulate_Physical(self,R,T,N,AU_gas,original_time,Interpolate):
        self.Create_sim()
        self.interpolate_struct(R,T,N,original_time)

        full_name,e,a,q,i,om,w,_,_,_,_ = self.Comet_data[:11]
        self.full_name = full_name

        # Add comet
        self.sim.move_to_hel()
        
        self.sim.add(
            primary=self.sim.particles[0],
            m=0.0,
            a=float(q)/(1.-float(e)),
            e=float(e),
            inc=np.deg2rad(float(i)),
            Omega=np.deg2rad(float(om)),
            omega=np.deg2rad(float(w)),
            T=self.Tp_MJD,
            hash=full_name
        )
        
        self.sim.move_to_com()

        comet = self.sim.particles[full_name]
        
        for i, time in enumerate(self.Time_array):
            self.sim.integrate(time)
            
            self.sim.additional_forces = self.interpolate_RTN

            self.data_array[i] = [comet.a, np.rad2deg(comet.Omega), np.rad2deg(comet.omega)]
            self.trajectory[i] = [comet.x, comet.y, comet.z, comet.vx, comet.vy, comet.vz]
            self.times[i] = [time]
        
class Physical_model:
    def __init__(self, datafile):
        self.T_to_peri = datafile[0]
        self.AU = datafile[1]
        self.Q = datafile[2]
        self.NGA = datafile[3]

        # time array for interpolation

    def create_pandas(self):
        num_rows, num_cols = self.NGA[0].shape
        regions = [i for i in range(num_cols)]

        def make_df(data, T):
            df = pd.DataFrame(data, columns=regions)
            df["Time_to_peri"] = T
            return df

        self.R_pandas = make_df(self.NGA[0],  self.T_to_peri)
        self.T_pandas = make_df(self.NGA[1],  self.T_to_peri)
        self.N_pandas = make_df(self.NGA[2],  self.T_to_peri)

        Q_data = self.Q.T
        self.Q_pandas = make_df(Q_data, self.T_to_peri)
   
    def sample_regions(self, sampled_regions):
        self.sampled_Q = self.Q_pandas[sampled_regions]

        sampled_R = self.R_pandas[sampled_regions]
        sampled_T = self.T_pandas[sampled_regions]
        sampled_N = self.N_pandas[sampled_regions]
        
        self.sum_Q = self.sampled_Q.sum(axis = 1)
        self.sum_R = sampled_R.sum(axis = 1)
        self.sum_T = sampled_T.sum(axis = 1)
        self.sum_N = sampled_N.sum(axis = 1)

    def Q_validation(self,sampled_regions):
        """Vallidation is when the water production falls within the 3 AU Inbound and Outbound light curve"""

        self.sample_regions(sampled_regions)

        helio = self.AU

        k_near = 6.7285 # Near-Sun slope

        k_near_u = 10.0 # Estimated from Fig. 7

        k_near_l = 3.0 # Estimated from Fig. 7

        k_far = 12.847 # Far-Sun slope

        k_far_u = 16.25 # Estimated from Fig. 7

        k_far_l = 8.75 # Estimated from Fig. 7

        m1 = 8.28 # Pre-perihelion magnitude at 1 au

        m1_l = 10.48 # 75% upper quartile from table 8

        m1_u = 6.9 # 25% lower quartile from table 8


        AJorda = 30.74

        BJorda = 0.24 # Jorda


        ASosa = 30.53 # Sosa & Fernandez 2011. Valid below 3 AU

        BSosa = 0.234


        ABiver = 30. # Biver 2001 (https://ui.adsabs.harvard.edu/abs/2001ICQ....23...85B/abstract). Valid above 3 AU

        BBiver = 0.29


        TRANSITION_R = 3.16 # au

        transition_log_r = np.log10(TRANSITION_R)

        log_r = np.log10(np.asarray(helio))

        mask = log_r < transition_log_r

        m_far = m1 + transition_log_r * (k_near - k_far)

        m_far_u = m1_u + transition_log_r * (k_near_l - k_far_l)

        m_far_l = m1_l + transition_log_r * (k_near_u - k_far_u)

        total_mag = np.where(mask, m1 + k_near * log_r, m_far + k_far * log_r)

        u_mag = np.where(mask, m1_u + k_near_l * log_r, m_far_u + k_far_l * log_r)

        l_mag = np.where(mask, m1_l + k_near_u * log_r, m_far_l + k_far_u * log_r)



        Oproduct = 10**(ASosa - BSosa * total_mag)

        Oproduct_u = 10**(ASosa - BSosa * u_mag)

        Oproduct_l = 10**(ASosa - BSosa * l_mag)

        # validation
        q_peri = min(helio)
        mask = helio == q_peri
        idx = np.argmax(mask)+1
        helio_trunc=helio[:idx]

        mask2 = helio_trunc<=3                      # look for q<=au<=3 AU indices
        Q_masked = self.sum_Q[:idx][mask2]
        Oproduct_u_masked = Oproduct_u[:idx][mask2]  # upper limit light curve
        Oproduct_l_masked = Oproduct_l[:idx][mask2]  # lower limit light curve

        # check if out of bounds between the 3 AU
        out_of_bounds = (Q_masked > Oproduct_u_masked)


        if not np.any(out_of_bounds):
            # q_peri = min(helio)
            # mask = helio == q_peri
            # idx = np.argmax(mask)+1

            # plt.figure(figsize=(15,8))
            # plt.title(f"Comet 67P Water production curve for combined regions {sampled_regions} on Q4's Orbit")
            # plt.semilogy(helio,self.sum_Q,label='Q',color = 'black',linestyle = "-")
            # plt.semilogy(helio, Oproduct, label = 'Brightening trend', color = 'grey',linestyle = '--')
            # plt.semilogy(helio, Oproduct_u, label = 'Upper Brightening trend', color = 'grey',linestyle = "-")
            # plt.semilogy(helio, Oproduct_l, label = 'Lower Brightening trend', color = 'grey',linestyle = "-")
            # plt.fill_between(helio[:idx], Oproduct_l[:idx], Oproduct_u[:idx], color = 'lightgrey')


            # plt.xlabel('AU', fontsize = 'xx-large')
            # plt.ylabel(r'Outgassing Rate (s$^{-1}$)', fontsize = 'xx-large')

            # plt.grid(True)
            # plt.legend()
            # plt.show()
            return self.sum_R, self.sum_T, self.sum_N, sampled_regions, 1
        else:
            # print(f"Invalid sampling: Q out of bounds for regions {np.sort(sampled_regions)}")
            return 0,0,0,sampled_regions, 0 

    def random_valid_Q(self,sampled_regions,saving):
        # For the plot filling
        helio = self.AU
        sampled_Q = self.Q_pandas[sampled_regions]
        sum_Q = self.sampled_Q.sum(axis = 1)

        k_near = 6.7285 # Near-Sun slope

        k_near_u = 10.0 # Estimated from Fig. 7

        k_near_l = 3.0 # Estimated from Fig. 7

        k_far = 12.847 # Far-Sun slope

        k_far_u = 16.25 # Estimated from Fig. 7

        k_far_l = 8.75 # Estimated from Fig. 7

        m1 = 8.28 # Pre-perihelion magnitude at 1 au

        m1_l = 10.48 # 75% upper quartile from table 8

        m1_u = 6.9 # 25% lower quartile from table 8


        AJorda = 30.74

        BJorda = 0.24 # Jorda


        ASosa = 30.53 # Sosa & Fernandez 2011. Valid below 3 AU

        BSosa = 0.234


        ABiver = 30. # Biver 2001 (https://ui.adsabs.harvard.edu/abs/2001ICQ....23...85B/abstract). Valid above 3 AU

        BBiver = 0.29


        TRANSITION_R = 3.16 # au

        transition_log_r = np.log10(TRANSITION_R)

        log_r = np.log10(np.asarray(helio))

        mask = log_r < transition_log_r

        m_far = m1 + transition_log_r * (k_near - k_far)

        m_far_u = m1_u + transition_log_r * (k_near_l - k_far_l)

        m_far_l = m1_l + transition_log_r * (k_near_u - k_far_u)

        total_mag = np.where(mask, m1 + k_near * log_r, m_far + k_far * log_r)

        u_mag = np.where(mask, m1_u + k_near_l * log_r, m_far_u + k_far_l * log_r)

        l_mag = np.where(mask, m1_l + k_near_u * log_r, m_far_l + k_far_u * log_r)



        Oproduct = 10**(ASosa - BSosa * total_mag)

        Oproduct_u = 10**(ASosa - BSosa * u_mag)

        Oproduct_l = 10**(ASosa - BSosa * l_mag)

        q_peri = min(helio)
        mask = helio == q_peri
        idx = np.argmax(mask)+1

        plt.figure(figsize=(15,8))
        plt.title(f"Comet 67P Water production curve for combined regions {sampled_regions} on Q4's Orbit")
        plt.semilogy(helio,sum_Q,label='Q',color = 'black',linestyle = "-")
        plt.semilogy(helio, Oproduct, label = 'Brightening trend', color = 'grey',linestyle = '--')
        plt.semilogy(helio, Oproduct_u, label = 'Upper Brightening trend', color = 'grey',linestyle = "-")
        plt.semilogy(helio, Oproduct_l, label = 'Lower Brightening trend', color = 'grey',linestyle = "-")
        plt.fill_between(helio[:idx], Oproduct_l[:idx], Oproduct_u[:idx], color = 'lightgrey')


        plt.xlabel('AU', fontsize = 'xx-large')
        plt.ylabel(r'Outgassing Rate (s$^{-1}$)', fontsize = 'xx-large')

        plt.grid(True)
        plt.legend()
        plt.savefig(saving)
        plt.close()

class plotting:
    def __init__(self,elements):
        self.elements = elements

    def plot_STD(self,saving):
        #------------------------------------
        # Determine Median & STD
        labels = [r"$\Delta a$", r"$\Delta \Omega$", r"$\Delta \omega$"]
        xlabels = ["AU","deg",'deg']
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))

        self.statistics = []
        for i in range(3):
            data = self.elements[:, i]
        
            # Fit a normal distribution
            median = np.median(data)
            std = np.std(data)
            self.statistics.append([median,std])
            # Histogram
            n, bins, patches = axes[i].hist(
                data, bins=50, density=True, alpha=0.6, color='skyblue', label='Data'
            )
            
            # Line fit
            x = np.linspace(bins[0], bins[-1], 200)
            y = norm.pdf(x, median, std)
            axes[i].plot(x, y, 'r-', linewidth=2)
            
            axes[i].set_title(f"{labels[i]}, Median {'{:e}'.format(median)}, std {'{:e}'.format(std)}")
            axes[i].set_xlabel(xlabels[i])
            axes[i].legend()
            axes[i].grid(alpha=0.3)
            axes[i].tick_params(axis='x', labelrotation=70)
        axes[0].set_ylabel('Probability Density')
        plt.tight_layout()
        plt.savefig(f"{saving}")
        plt.close()
    
    def deviation_time(self,times,trajectory,Marsden_trajectory,TwoBP_rajectory,saving):
        self.times = times
        self.Marsden_trajectory = Marsden_trajectory
        self.TwoBP_rajectory = TwoBP_rajectory
        self.Physical_trajctories = trajectory

        plt.figure(figsize=(8, 6))
        difference = self.Physical_trajctories - self.TwoBP_rajectory
        for i, traj in enumerate(difference):
            pos_error = np.linalg.norm(traj[:,:3], axis=1)*const.au/1000
            plt.plot(self.times, pos_error)

        plt.yscale("log")
        plt.grid(True)
        plt.xlabel("Time [Julian Days]")
        plt.ylabel("Position difference [km]")
        plt.title("Clone dispersion from the Thermophysical model")
        plt.tight_layout()
        plt.savefig(f"{saving}")
        plt.close()

    def deviation_AU(self,times,trajectory,Marsden_trajectory,TwoBP_rajectory,saving):
        self.times = times
        self.Marsden_trajectory = Marsden_trajectory
        self.TwoBP_rajectory = TwoBP_rajectory
        self.Physical_trajctories = trajectory

        plt.figure(figsize=(8, 6))
        difference = self.Physical_trajctories - self.TwoBP_rajectory
        for i, traj in enumerate(difference):
            pos_error = np.linalg.norm(traj[:,:3], axis=1)*const.au/1000
            plt.plot(np.linalg.norm(self.TwoBP_rajectory[:,:3],axis=1), pos_error)

        plt.yscale("log")
        plt.xlabel("Heliocentric distance [AU]")
        plt.ylabel("Position difference [km]")
        plt.grid(True)
        plt.title("Clone dispersion from the Thermophysical model")
        plt.tight_layout()
        plt.savefig(f"{saving}")
        plt.close()

    def plot_gr(self,gr,times,names,saving):
        names = np.array(names)
        plt.figure(figsize=(8,6))
        for i, name in enumerate(names):
            plt.plot(times[i], gr[i], label=name)

        plt.yscale("log")
        plt.grid(True)
        # plt.legend()
        plt.title("gr function of the filtered JPL comets")
        plt.savefig(saving)
        plt.close()
    
    def plot_line_kms(self, trajectory, Marsden_trajectory, title,saving,saving2):
        Truth = np.linalg.norm(Marsden_trajectory[:, :3], axis=1)

        idx_peri = np.argmin(Truth)
        masked_Truth = Marsden_trajectory[:idx_peri, 3:]

        all_pos_errors = []
        for traj in trajectory:
            masked_trajectory = traj[:idx_peri, 3:]
            pos_error = np.linalg.norm(masked_trajectory - masked_Truth, axis=1)
            all_pos_errors.append(pos_error)

        all_pos_errors = np.array(all_pos_errors)*const.au/const.day

        median_error = np.median(all_pos_errors, axis=0)
        std_error = np.std(all_pos_errors, axis=0)

        plt.figure(figsize=(8, 6))
        plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
        plt.fill_between(
            Truth[:idx_peri],
            median_error - std_error,
            median_error + std_error,
            color='tab:blue',
            alpha=0.3,
            label=r'±1$\sigma$'
        )

        plt.yscale("log")
        plt.xlabel("Distance from Sun [AU]")
        plt.ylabel("Absolute velocity difference [km/s]")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(saving)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
        plt.fill_between(
            Truth[:idx_peri],
            median_error - std_error,
            median_error + std_error,
            color='tab:blue',
            alpha=0.3,
            label=r'±1$\sigma$'
        )

        plt.yscale("log")
        plt.xlabel("Distance from Sun [AU]")
        plt.ylabel("Absolute velocity difference [km/s]")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xlim((np.min(Truth),1.5))
        plt.ylim((0.001,10))
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(saving2)
        plt.close()

    def plot_line_km(self, trajectory, Marsden_trajectory,title, saving, saving2):
        # Truth orbit
        Truth = np.linalg.norm(Marsden_trajectory[:, :3], axis=1)

        # Find perihelion
        idx_peri = np.argmin(Truth)
        masked_Truth = Marsden_trajectory[:idx_peri, :3]

        all_pos_errors = []
        for traj in trajectory:
            masked_trajectory = traj[:idx_peri, :3]
            pos_error = np.linalg.norm(masked_trajectory - masked_Truth, axis=1)
            all_pos_errors.append(pos_error)

        all_pos_errors = np.array(all_pos_errors)*const.au/1000

        median_error = np.median(all_pos_errors, axis=0)
        std_error = np.std(all_pos_errors, axis=0)

        plt.figure(figsize=(8, 6))
        plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
        plt.fill_between(
            Truth[:idx_peri],
            median_error - std_error,
            median_error + std_error,
            color='tab:blue',
            alpha=0.3,
            label=r'±1$\sigma$'
        )

        plt.yscale("log")
        plt.xlabel("Distance from Sun [AU]")
        plt.ylabel("Absolute position difference [km]")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(saving)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
        plt.fill_between(
            Truth[:idx_peri],
            median_error - std_error,
            median_error + std_error,
            color='tab:blue',
            alpha=0.3,
            label=r'±1$\sigma$'
        )

        plt.yscale("log")
        plt.xlabel("Distance from Sun [AU]")
        plt.ylabel("Absolute position difference [km]")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xlim((np.min(Truth),1.5))
        plt.ylim((1,10000))
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(saving2)
        plt.close()

    # def plot_line_kms_COMB(self, trajectory, twoBP, Marsden_trajectory, title,saving,saving2):
    #     TwoBP = np.linalg.norm(twoBP[:, :3], axis=1)
    #     Mars = np.linalg.norm(Marsden_trajectory[:, :3], axis=1)

    #     idx_peri = np.argmin(TwoBP)
    #     masked_twoBP = twoBP[:idx_peri, 3:]
    #     masked_Mars = Mars[:idx_peri, 3:]

    #     all_vel_errors_Term = []
    #     for traj in trajectory:
    #         masked_trajectory = traj[:idx_peri, 3:]
    #         Vel_diff_Term = np.linalg.norm(masked_trajectory - masked_twoBP, axis=1)
    #         all_vel_errors_Term.append(Vel_diff_Term)


    #     Vel_diff_Mars = np.linalg.norm(masked_Mars - masked_twoBP, axis=1)
    #     all_vel_errors_Mars = np.array(Vel_diff_Mars)*const.au/const.day
    #     median_error_Mars = np.median(all_vel_errors_Mars, axis=0)

    #     all_vel_errors_Term = np.array(all_vel_errors_Term)*const.au/const.day
    #     median_error_term = np.median(all_vel_errors_Term, axis=0)
    #     STD_error_term = np.std(all_vel_errors_Term, axis=0)

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(TwoBP[:idx_peri], median_error_term, color='tab:blue', label='Median difference Marsden')
    #     plt.fill_between(
    #         TwoBP[:idx_peri],
    #         all_vel_errors_Term - STD_error_term,
    #         all_vel_errors_Term + STD_error_term,
    #         color='tab:blue',
    #         alpha=0.3,
    #         label=r'±1$\sigma$'
    #     )

    #     plt.plot(TwoBP[:idx_peri], median_error_Mars, color='tab:blue', label='Median difference')
    #     # plt.fill_between(
    #     #     TwoBP[:idx_peri],
    #     #     median_error - std_error,
    #     #     median_error + std_error,
    #     #     color='tab:blue',
    #     #     alpha=0.3,
    #     #     label=r'±1$\sigma$'
    #     # )

    #     plt.yscale("log")
    #     plt.xlabel("Distance from Sun [AU]")
    #     plt.ylabel("Absolute velocity difference [km/s]")
    #     plt.grid(True, which="both", linestyle="--", alpha=0.5)
    #     plt.legend()
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(saving)
    #     plt.close()

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(TwoBP[:idx_peri], median_error_term, color='tab:blue', label='Median difference Marsden')
    #     plt.fill_between(
    #         TwoBP[:idx_peri],
    #         all_vel_errors_Term - STD_error_term,
    #         all_vel_errors_Term + STD_error_term,
    #         color='tab:blue',
    #         alpha=0.3,
    #         label=r'±1$\sigma$'
    #     )

    #     plt.yscale("log")
    #     plt.xlabel("Distance from Sun [AU]")
    #     plt.ylabel("Absolute velocity difference [km/s]")
    #     plt.grid(True, which="both", linestyle="--", alpha=0.5)
    #     plt.xlim((np.min(Truth),1.5))
    #     plt.ylim((0.001,10))
    #     plt.legend()
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(saving2)
    #     plt.close()

    # def plot_line_km_COMB(self, trajectory, twoBP, Marsden_trajectory,title, saving, saving2):
    #     # Truth orbit
    #     Truth = np.linalg.norm(Marsden_trajectory[:, :3], axis=1)

    #     # Find perihelion
    #     idx_peri = np.argmin(Truth)
    #     masked_Truth = Marsden_trajectory[:idx_peri, :3]

    #     all_pos_errors = []
    #     for traj in trajectory:
    #         masked_trajectory = traj[:idx_peri, :3]
    #         pos_error = np.linalg.norm(masked_trajectory - masked_Truth, axis=1)
    #         all_pos_errors.append(pos_error)

    #     all_pos_errors = np.array(all_pos_errors)*const.au/1000

    #     median_error = np.median(all_pos_errors, axis=0)
    #     std_error = np.std(all_pos_errors, axis=0)

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
    #     plt.fill_between(
    #         Truth[:idx_peri],
    #         median_error - std_error,
    #         median_error + std_error,
    #         color='tab:blue',
    #         alpha=0.3,
    #         label=r'±1$\sigma$'
    #     )

    #     plt.yscale("log")
    #     plt.xlabel("Distance from Sun [AU]")
    #     plt.ylabel("Absolute position difference [km]")
    #     plt.grid(True, which="both", linestyle="--", alpha=0.5)
    #     plt.legend()
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(saving)
    #     plt.close()

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(Truth[:idx_peri], median_error, color='tab:blue', label='Median difference')
    #     plt.fill_between(
    #         Truth[:idx_peri],
    #         median_error - std_error,
    #         median_error + std_error,
    #         color='tab:blue',
    #         alpha=0.3,
    #         label=r'±1$\sigma$'
    #     )

    #     plt.yscale("log")
    #     plt.xlabel("Distance from Sun [AU]")
    #     plt.ylabel("Absolute position difference [km]")
    #     plt.grid(True, which="both", linestyle="--", alpha=0.5)
    #     plt.xlim((np.min(Truth),1.5))
    #     plt.ylim((1,10000))
    #     plt.legend()
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(saving2)
    #     plt.close()

    def sample_stats(self, regions, valid_regions, invalid_Q, invalid_sim, title, saving):
        # Flatten all lists of lists into single lists
        flat_valid = [r for sub in valid_regions for r in sub]
        flat_invalid_Q = [r for sub in invalid_Q for r in sub]
        flat_invalid_sim = [r for sub in invalid_sim for r in sub]

        # Count occurrences of each region
        valid_counts = Counter(flat_valid)
        invalidQ_counts = Counter(flat_invalid_Q)
        invalidSim_counts = Counter(flat_invalid_sim)

        # Ensure consistent ordering across all regions
        region_names = np.arange(0, len(regions)) if isinstance(regions, int) else np.array(regions)
        valid_vals = [valid_counts.get(r, 0) for r in region_names]
        invalidQ_vals = [invalidQ_counts.get(r, 0) for r in region_names]
        invalidSim_vals = [invalidSim_counts.get(r, 0) for r in region_names]

        # Plot 
        plt.figure(figsize=(15, 6))
        x = np.arange(len(region_names))
        width = 0.2

        plt.bar(x - width, valid_vals, width, color="green", label="Valid")
        plt.bar(x, invalidSim_vals, width, color="darkred", label="Invalid (Simulation)")
        plt.bar(x + width, invalidQ_vals, width, color="red", label="Invalid (Q)")


        plt.xticks(x, region_names, rotation=70)
        plt.ylabel("Number of samples")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(saving)
        plt.close()


