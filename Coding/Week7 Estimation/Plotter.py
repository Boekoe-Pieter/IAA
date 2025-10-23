import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import numpy as np
import random
import matplotlib.patches as patches

from tudatpy import constants

AU = constants.ASTRONOMICAL_UNIT
day = constants.JULIAN_DAY

class Observations_Plotter:
    def __init__(self,sim,name,simulated_observations,directory_name,addition):
        self.sim = sim

        self.name = name
        self.saving_dir = f"{directory_name}{addition}"

        self.simulated_observations = simulated_observations

        self.observation_times = simulated_observations.get_observation_times
        self.simulated_observations = simulated_observations.get_observations

        self.epochs = []
        self.ra_deg = []
        self.dec_deg = []
        for epochs, obs in zip(self.observation_times(),
                                self.simulated_observations()):
            self.epochs.append(epochs)
            self.ra_deg.append(np.rad2deg(obs[0::2]))
            self.dec_deg.append(np.rad2deg(obs[1::2]))

        self.time_days = (np.array(self.epochs)[0] - self.epochs[0][0]) / (3600.0*24)

    def RADEC_overtime(self):
        # ----------------------------------------------------------------------
        # RA/DEC over time
        plt.figure(figsize=(10,4))
        plt.suptitle(f"RA and DEC ober time for {len(self.ra_deg[0])} observations")
        plt.subplot(1,2,1)

        plt.scatter(self.time_days, self.ra_deg, s=2, label="Batch")

        plt.xlabel("Time [days]")
        plt.ylabel("Right Ascension [deg]")
        plt.grid(True)
        plt.title("Simulated RA")

        plt.subplot(1,2,2)
        plt.scatter(self.time_days, self.dec_deg, s=2, label="Batch")
        plt.xlabel("Time [days]")
        plt.ylabel("Declination [deg]")
        plt.grid(True)
        plt.title("Simulated Dec")

        plt.tight_layout()
        plt.savefig(f"{self.saving_dir}/Observation/{self.sim}_RADEC_time.pdf", dpi=300)
        plt.close()
    
    def skyplot(self):
        # ----------------------------------------------------------------------
        # RA/DEC Skyplot
        plt.figure(figsize=(7,7))
        sc = plt.scatter(self.ra_deg, self.dec_deg, c=self.time_days, cmap='viridis', s=2)
        plt.xlabel("Right Ascension [deg]")
        plt.ylabel("Declination [deg]")
        plt.title(f"Sky Track, comet:{self.name},Nobs: {len(self.ra_deg[0])}")
        plt.gca().invert_xaxis()  
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label("Time [days]")
        plt.savefig(f"{self.saving_dir}/Observation/{self.sim}_RADEC_skyplot.pdf", dpi=300)
        plt.close()

    def aitoff(self):
        # ----------------------------------------------------------------------
        # Aitoff Projection
        ra_rad = np.deg2rad(self.ra_deg)
        dec_rad = np.deg2rad(self.dec_deg)

        plt.figure(figsize=(10,5))
        plt.title(f"Aitoff projection of {self.name} and {len(self.ra_deg[0])} observations")
        ax = plt.subplot(111, projection='aitoff')

        sc = ax.scatter(ra_rad, dec_rad, c=self.time_days, cmap='viridis', s=5)

        ax.set_title("Comet Sky Track", va='bottom')
        plt.grid(True)

        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label("Time [days]")

        plt.tight_layout()
        plt.savefig(f"{self.saving_dir}/Observation/{self.sim}_RADEC_Aitoff.pdf", dpi=300)
        plt.close()

class estimation_plotter:
    def __init__(self,sim,name,pod_it,pod_output,simulated_observations,covariance_output,parameters_to_estimate,directory_name,addition):
        self.sim = sim
        self.name = name
        self.saving_dir = f"{directory_name}{addition}"

        self.simulated_observations = simulated_observations
        self.pod_output = pod_output
        self.pod_it = pod_it

        self.covariance_output = covariance_output
        self.parameters_to_estimate = parameters_to_estimate

        self.observation_times = simulated_observations.get_observation_times
        self.simulated_observations = simulated_observations.get_observations

        self.epochs = []
        self.ra_deg = []
        self.dec_deg = []
        for epochs, obs in zip(self.observation_times(),
                                self.simulated_observations()):
            self.epochs.append(epochs)
            self.ra_deg.append(np.rad2deg(obs[0::2]))
            self.dec_deg.append(np.rad2deg(obs[1::2]))
        self.time_days = (np.array(self.epochs) - self.epochs[0])/ (3600.0*24)

    def residuals(self,simulated_observations):
        # ----------------------------------------------------------------------
        # Plotting of the residuals
        residual_history = self.pod_output.residual_history
        # Number of columns and rows for our plot
        number_of_columns = 2

        number_of_rows = (
            int( self.pod_it / number_of_columns)
            if  self.pod_it % number_of_columns == 0
            else int(( self.pod_it + 1) / number_of_columns)
        )

        fig, axs = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(9, 3.5 * number_of_rows),
            sharex=True,
            sharey=False,
        )

        # We cheat a little to get an approximate year out of our times (which are in seconds since J2000)
        residual_times = (
            np.array(simulated_observations.concatenated_times) / (86400 * 365.25) + 2000
        )

        # plot the residuals, split between RA and DEC types
        for idx, ax in enumerate(fig.get_axes()):
            ax.grid()
            # we take every second
            ax.scatter(
                residual_times[::2],
                residual_history[
                    ::2,
                    idx,
                ],
                marker="+",
                s=60,
                label="Right Ascension",
            )
            ax.scatter(
                residual_times[1::2],
                residual_history[
                    1::2,
                    idx,
                ],
                marker="+",
                s=60,
                label="Declination",
            )
            ax.set_ylabel("Observation Residual [rad]")
            ax.set_title("Iteration " + str(idx + 1))
        
        plt.tight_layout()

        # add the year label for the x-axis
        for col in range(number_of_columns):
            axs[int(number_of_rows - 1), col].set_xlabel("Year")

        axs[0, 0].legend()
        plt.savefig(f"{self.saving_dir}/Estimation/{self.sim}_Residuals.pdf", dpi=300)
        plt.close()

    def correlation(self):
        # ----------------------------------------------------------------------
        # correlation plot
        corr_matrix = self.covariance_output.correlations

        covar_names = ["x", 'y', 'z', 'vx', 'vy', 'vz']

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr_matrix, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(covar_names)), labels=covar_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(covar_names)), labels=covar_names)

        for i in range(len(covar_names)):
            for j in range(len(covar_names)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center", color="w", fontsize=8)

        cb = plt.colorbar(im)
        cb.set_label("Correlation coefficient")

        ax.set_xlabel("Estimated Parameter")
        ax.set_ylabel("Estimated Parameter")
        fig.suptitle(f"Correlation matrix for estimated parameters of {self.name}")

        fig.tight_layout()
        plt.savefig(f"{self.saving_dir}/Estimation/{self.sim}_Corr_matrix.pdf", dpi=300)
        plt.close()

    def formal_erros(self):
        # ----------------------------------------------------------------------
        # Formal Errors and Covariance Matrix
        x_star = self.parameters_to_estimate.parameter_vector # Nominal solution (center of the ellipsoid)

        initial_covariance = self.covariance_output.covariance
        formal_errors = self.covariance_output.formal_errors

        # # Set methodological options
        # state_transition_interface = self.state_transition
        # output_times = self.observation_times

        diagonal_covariance = np.diag(formal_errors**2)
        # print(f'Formal Error Matrix:\n\n{diagonal_covariance}\n')

        sigma = 3  # Confidence level
        original_eigenvalues, original_eigenvectors = np.linalg.eig(diagonal_covariance)
        original_diagonal_eigenvalues, original_diagonal_eigenvectors = np.linalg.eig(diagonal_covariance)
        # print(f'Estimated state and parameters:\n\n {parameters_to_estimate.parameter_vector}\n')
        # print(f'Eigenvalues of Covariance Matrix:\n\n {original_eigenvalues}\n')
        # print(f'Eigenvalues of Formal Errors Matrix:\n\n {original_diagonal_eigenvalues}\n')

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(original_eigenvalues)[::-1]
        diagonal_sorted_indices = np.argsort(original_diagonal_eigenvalues)[::-1]

        eigenvalues = original_eigenvalues[sorted_indices]
        eigenvectors = original_eigenvectors[:, sorted_indices]

        diagonal_eigenvalues = original_diagonal_eigenvalues[diagonal_sorted_indices]
        diagonal_eigenvectors = original_diagonal_eigenvectors[:, diagonal_sorted_indices]

        # Output results
        # print(f"Sorted Eigenvalues (variances along principal axes):\n\n{eigenvalues}\n")
        # print(f"Sorted Formal Error Matrix Eigenvalues (variances along principal axes):\n\n{diagonal_eigenvalues}\n")
        # print(f"Sorted Eigenvectors (directions of principal axes):\n\n{eigenvectors}\n")
        # print(f"Sorted Formal Error Matrix Eigenvectors (directions of principal axes):\n\n{diagonal_eigenvectors}\n")

        COV_sub = initial_covariance[np.ix_(np.sort(sorted_indices)[:3], np.sort(sorted_indices)[:3])]  #Covariance restriction to first 3 (spatial) eigenvectors
        diagonal_COV_sub = diagonal_covariance[np.ix_(np.sort(diagonal_sorted_indices)[:3], np.sort(diagonal_sorted_indices)[:3])]  #Covariance restriction to first 3 (spatial) eigenvectors

        x_star_sub = x_star[sorted_indices[:3]] #Nominal solution subset
        diagonal_x_star_sub = x_star[diagonal_sorted_indices[:3]] #Nominal solution subset

        # Eigenvalue decomposition of the submatrix
        eigenvalues, eigenvectors = np.linalg.eig(COV_sub)
        diagonal_eigenvalues, diagonal_eigenvectors = np.linalg.eig(diagonal_COV_sub)

        # Ensure eigenvalues are positive
        if np.any(eigenvalues <= 0):
            raise ValueError(f"$Covariance$ submatrix is not positive definite. Eigenvalues must be positive.\n")
        if np.any(diagonal_eigenvalues <= 0):
            raise ValueError(f"$Formal Errors$ submatrix is not positive definite. Eigenvalues must be positive.\n")

        phi = np.linspace(0, np.pi, 50)
        theta = np.linspace(0, 2 * np.pi,50)
        phi, theta = np.meshgrid(phi, theta)

        # Generate points on the unit sphere and multiply each direction by the corresponding eigenvalue
        x_ell= np.sqrt(eigenvalues[0])*  np.sin(phi) * np.cos(theta)
        y_ell = np.sqrt(eigenvalues[1])* np.sin(phi) * np.sin(theta)
        z_ell = np.sqrt(eigenvalues[2])* np.cos(phi)

        # Generate points on the unit sphere and multiply each direction by the corresponding diagonal_eigenvalue
        diagonal_x_ell = np.sqrt(diagonal_eigenvalues[0])*np.sin(phi) * np.cos(theta)
        diagonal_y_ell = np.sqrt(diagonal_eigenvalues[1])*np.sin(phi) * np.sin(theta)
        diagonal_z_ell = np.sqrt(diagonal_eigenvalues[2])*np.cos(phi)

        ell = np.stack([x_ell, y_ell, z_ell], axis=0)
        diagonal_ell = np.stack([diagonal_x_ell, diagonal_y_ell, diagonal_z_ell], axis=0)

        #Rotate the Ellipsoid(s). This is done by multiplying ell and diagonal_ell by the corresponding eigenvector matrices
        ellipsoid_boundary_3_sigma = 3 * np.tensordot(eigenvectors, ell, axes=1)
        ellipsoid_boundary_1_sigma = 1 * np.tensordot(eigenvectors, ell, axes=1)
        diagonal_ellipsoid_boundary_3_sigma = 3 * np.tensordot(diagonal_eigenvectors, diagonal_ell, axes=1)
        diagonal_ellipsoid_boundary_1_sigma = 1 * np.tensordot(diagonal_eigenvectors, diagonal_ell, axes=1)

        # Plot the ellipsoid in 3D
        fig = plt.figure(figsize=(15, 8))
        fig.tight_layout()

        ax = fig.add_subplot(121, projection='3d')
        diagonal_ax =fig.add_subplot(122, projection='3d')

        ax.plot_surface(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], color='cyan', alpha=0.4, label = '3-sigma (covariance)')
        ax.plot_surface(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], color='blue', alpha=0.4, label = '1-sigma (covariance)')

        diagonal_ax.plot_surface(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[1], diagonal_ellipsoid_boundary_3_sigma[2], color='red', alpha=0.2, label = '3-sigma (formal errors)')
        diagonal_ax.plot_surface(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[1], diagonal_ellipsoid_boundary_1_sigma[2], color='black', alpha=0.2, label = '1-sigma (formal errors)')

        ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[2], 'r+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
        ax.plot(ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], 'r+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
        ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], 'r+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))

        ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[2], 'b+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
        ax.plot(ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], 'b+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
        ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], 'b+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))

        diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[2], 'r+', alpha=0.1, zdir='y', zs=2*np.max(diagonal_ellipsoid_boundary_3_sigma[1]))
        diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[1], diagonal_ellipsoid_boundary_1_sigma[2], 'r+',alpha=0.1, zdir='x', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[0]))
        diagonal_ax.plot(diagonal_ellipsoid_boundary_1_sigma[0], diagonal_ellipsoid_boundary_1_sigma[1], 'r+',alpha=0.1, zdir='z', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[2]))

        diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[2], 'b+', alpha=0.1, zdir='y', zs=2*np.max(diagonal_ellipsoid_boundary_3_sigma[1]))
        diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[1], diagonal_ellipsoid_boundary_3_sigma[2], 'b+',alpha=0.1, zdir='x', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[0]))
        diagonal_ax.plot(diagonal_ellipsoid_boundary_3_sigma[0], diagonal_ellipsoid_boundary_3_sigma[1], 'b+',alpha=0.1, zdir='z', zs=-2*np.max(diagonal_ellipsoid_boundary_3_sigma[2]))

        ax.set_xlabel(r'$(x-x^*)$')
        ax.set_ylabel(r'$(y-y^*)$')
        ax.set_zlabel(r'$(z-z^*)$')
        ax.set_title('3D Confidence Ellipsoid and Projections')
        ax.legend(loc = 'upper right')

        diagonal_ax.set_xlabel(r'$(x-x^*)$')
        diagonal_ax.set_ylabel(r'$(y-y^*)$')
        diagonal_ax.set_zlabel(r'$(z-z^*)$')
        diagonal_ax.set_title('Formal Errors and Projections')
        diagonal_ax.legend(loc = 'upper right')

        plt.legend()
        plt.savefig(f"{self.saving_dir}/Estimation/{self.sim}_Confidence.pdf", dpi=300)
        plt.close()

class statistics_plotter:
    def __init__(self,data,info_dict_synobs,directory_name,addition):
        self.data = data
        self.info = info_dict_synobs

        self.saving_dir = f"{directory_name}{addition}"

        self.Family = {
            "Truth_norm":{},

            "Fitted_norm":{},

            "Clone_norm": {},
            "Clone_Divergence": {},
            "Clone_Divergence_Norm": {},
            "Clone_Divergence_Norm_peri": {},                  

            "Clone_vel_norm":{},
            "Clone_divergence_vel":{},
            "Clone_divergence_vel_norm":{},
            "Clone_Divergence_Nor_vel_peri": {},                  

            "Pos_div_1AU": {},
            "Vel_div_1AU": {},
            }

        self.general_statistics = {
            "Clone_Divergence_Norm_peri_DPERI": {},  
            "Clone_Divergence_Vel_Norm_peri_DPERI": {},
            "Clone_Divergence_Norm_peri_NOBS": {},  
            "Clone_Divergence_Vel_Norm_peri_NOBS": {},
        }

        self.obs_scatter = {
            "N_obs": {},
            "Date": {},
            "Helio": {},
        }

        def compute_family(family_dict, data):
            comet = self.info["Name"]

            if comet not in self.general_statistics["Clone_Divergence_Norm_peri_DPERI"]:
                self.general_statistics["Clone_Divergence_Norm_peri_DPERI"][comet] = {}
            if comet not in self.general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"]:
                self.general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"][comet] = {}

            if comet not in self.general_statistics["Clone_Divergence_Norm_peri_NOBS"]:
                self.general_statistics["Clone_Divergence_Norm_peri_NOBS"][comet] = {}
            if comet not in self.general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"]:
                self.general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"][comet] = {}

            monte_sample = data['Montecarlo_trajectory']
            Truth_trajectory = data['Truth_Reference_trajectory']
            Truth_pos_norm = np.linalg.norm(Truth_trajectory[:, 0:3], axis=1)
            family_dict["Truth_norm"] = Truth_pos_norm

            for sim, sim_data in monte_sample.items():
                for key, traj in sim_data.items():
                    pos = traj[:, 0:3]
                    vel = traj[:, 3:]

                    # ----------------------------------------------------------------------
                    # Compute divergence
                    family_dict["Clone_norm"].setdefault(sim, {})[key] = np.linalg.norm(pos, axis=1)
                    family_dict["Clone_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(vel, axis=1)

                    family_dict["Clone_Divergence"].setdefault(sim, {})[key] = pos - Truth_trajectory[:, 0:3]
                    family_dict["Clone_Divergence_Norm"].setdefault(sim, {})[key] = np.linalg.norm(
                        family_dict["Clone_Divergence"][sim][key], axis=1
                    )

                    family_dict["Clone_divergence_vel"].setdefault(sim, {})[key] = vel - Truth_trajectory[:, 3:]
                    family_dict["Clone_divergence_vel_norm"].setdefault(sim, {})[key] = np.linalg.norm(
                        family_dict["Clone_divergence_vel"][sim][key], axis=1
                    )

                    # ----------------------------------------------------------------------
                    # Find perihelion divergence
                    idx_peri = np.argmin(Truth_pos_norm)

                    family_dict["Clone_Divergence_Norm_peri"].setdefault(sim, {})[key] = \
                        family_dict["Clone_Divergence_Norm"][sim][key][idx_peri]
                    family_dict["Clone_Divergence_Nor_vel_peri"].setdefault(sim, {})[key] = \
                        family_dict["Clone_divergence_vel_norm"][sim][key][idx_peri]
                    
                    # ----------------------------------------------------------------------
                    # timeline
                    perihelion = self.data["Truth_Reference_trajectory_times"][-1]
                    last_obs = self.data["observation_times"][sim][-1]

                    N_obs = self.data["Sim_info"][sim].get("N_obs")
                    dt_days_peri = (perihelion - last_obs) / constants.JULIAN_DAY

                    # ----------------------------------------------------------------------
                    # Saving to dictionary             
                    self.general_statistics["Clone_Divergence_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                        family_dict["Clone_Divergence_Norm_peri"][sim][key]
                    )

                    self.general_statistics["Clone_Divergence_Vel_Norm_peri_DPERI"][comet].setdefault(dt_days_peri[0], []).append(
                        family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])



                    self.general_statistics["Clone_Divergence_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                        family_dict["Clone_Divergence_Norm_peri"][sim][key]
                    )

                    self.general_statistics["Clone_Divergence_Vel_Norm_peri_NOBS"][comet].setdefault(N_obs, []).append(
                        family_dict["Clone_Divergence_Nor_vel_peri"][sim][key])
                        
        compute_family(self.Family, self.data)

    def plot_3D(self):
        def D_traject(cartesian, ax=None, label=None, color=None, linestyle="-", alpha=1.0, lw=1.2):
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_aspect('equal', adjustable='box')

            ax.plot(cartesian[:, 0]/AU, cartesian[:, 1]/AU, cartesian[:, 2]/AU,
                    label=label, color=color, linestyle=linestyle, alpha=alpha, lw=lw)
            ax.scatter(cartesian[0, 0]/AU, cartesian[0, 1]/AU, cartesian[0, 2]/AU, marker="o", s=10, color=color)
            ax.scatter(cartesian[-1, 0]/AU, cartesian[-1, 1]/AU, cartesian[-1, 2]/AU, marker="x", s=10, color=color)
            return ax

        def plot_ensemble(data,montecarlo, fit, info, n_clones=20):   
            perturbing_bodies = data["environment"][1:]

            planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
            moons = [b for b in perturbing_bodies if b not in planets]

            def draw_sun(ax):
                radius_sun = 696340e3 /AU
                _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
                _x = radius_sun * np.cos(_u) * np.sin(_v)
                _y = radius_sun * np.sin(_u) * np.sin(_v)
                _z = radius_sun * np.cos(_v)
                ax.plot_wireframe(_x, _y, _z, color="orange", alpha=0.5, lw=0.5, zorder=0)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_aspect("equal", adjustable="box")

            clone_keys = list(montecarlo.keys())
            selected_keys = random.sample(clone_keys, min(n_clones, len(clone_keys)))
            for key in selected_keys:
                cartesian = montecarlo[key]
                ax.plot(cartesian[:, 0]/AU, cartesian[:, 1]/AU, cartesian[:, 2]/AU,
                        alpha=0.35, color="gray", lw=0.8)

            D_traject(np.array(data["Truth_Reference_trajectory"]), ax=ax,
                    label='JPL SBDB fitted orbit', color="black", linestyle="-", alpha=0.9, lw=1.8)
            D_traject(np.array(fit), ax=ax,
                    label='TUDAT fitted orbit', color="red", linestyle=":", alpha=0.9, lw=1.8)

            colors = plt.cm.tab20(np.linspace(0, 1, len(perturbing_bodies)))
            planet_handles, moon_handles = [], []

            for i, body in enumerate(perturbing_bodies):
                color = colors[i]
                linestyle = "-" if body in planets else ":"
                D_traject(np.array(data["N_body_trajectories"][body]), ax=ax,
                        color=color, linestyle=linestyle, alpha=0.9, lw=1.2)
                handle = Line2D([0], [0], color=color, lw=1.2, linestyle=linestyle)
                if body in planets:
                    planet_handles.append((handle, body))
                else:
                    moon_handles.append((handle, body))

            draw_sun(ax)

            max_value = max(np.max(np.abs(np.array(traj))) for traj in data["Truth_Reference_trajectory"])
            ax.set_xlim([-max_value/AU, max_value/AU])
            ax.set_ylim([-max_value/AU, max_value/AU])
            ax.set_zlim([-max_value/AU, max_value/AU])
            ax.set_xlabel("x [AU]")
            ax.set_ylabel("y [AU]")
            ax.set_zlabel("z [AU]")

            comet_legend = [
                Line2D([0], [0], color='black', lw=1.2, label='Nasa JPL SBDB Fit'),
                Line2D([0], [0], color='red', lw=1.2, linestyle=":", label='Synthetic Observations Fit'),
                Line2D([0], [0], color='gray', lw=1.2, linestyle=":", label='Monte carlo samples')

            ]
            planet_legend = [h for h, lbl in planet_handles]
            moon_legend = [h for h, lbl in moon_handles]

            leg1 = ax.legend(handles=comet_legend, loc='upper left', title="Comet Orbits", fontsize=8)
            leg2 = ax.legend(planet_legend, [lbl for _, lbl in planet_handles],
                            loc='upper right', title="Planets", fontsize=8)
            leg3 = ax.legend(moon_legend, [lbl for _, lbl in moon_handles],
                            loc='lower left', title="Moons", fontsize=7, ncol=2)

            ax.add_artist(leg1)
            ax.add_artist(leg2)
            ax.add_artist(leg3)

            plt.title(f"{n_clones} Monte Carlo samples of comet {info['Name']} — {info['Observations']} observations")
            plt.savefig(f"{self.saving_dir}/3D_trajectory_worstfit.pdf", dpi=300)
            plt.close() 

        keys = list(self.data["Estimated_Reference_trajectory"].keys())
        best_key = keys[0]
        worst_key = keys[-1]

        best_fit = self.data["Estimated_Reference_trajectory"][best_key]
        worst_fit = self.data["Estimated_Reference_trajectory"][worst_key]

        plot_ensemble(self.data,self.data["Montecarlo_trajectory"][worst_key],worst_fit,self.info, n_clones=self.info['Orbit_clones'])

    def boxplot(self, extra_time=15):
        comet = self.info["Name"]
        info = self.info
        stats = self.general_statistics

        def make_boxplot(divergence_dict, height, ylabel, title, scale=1.0, save_name=None, xlabel=r"$\Delta{Days}$"):
            dt = sorted(divergence_dict.keys(), reverse=True)
            data = [np.array(divergence_dict[n]) / scale for n in dt]

            fig, ax = plt.subplots(figsize=(15, 8))
            box = plt.boxplot(
                data,
                positions=dt,
                widths=5,
                patch_artist=True,
                manage_ticks=False
            )

            for patch in box["boxes"]:
                patch.set_facecolor("tab:blue")
                patch.set_alpha(0.6)

            if isinstance(dt[0], (int, float)) and xlabel == r"$\Delta{Days}$":
                time = np.arange(max(dt), min(dt) - extra_time, -extra_time)
                plt.xticks(time, rotation=70)
                plt.gca().invert_xaxis()

                # width = -60
                # x_start = 60
                # y_start = 0
                # square = patches.Rectangle(
                #     (x_start, y_start),
                #     width,
                #     height,
                #     linewidth=1,
                #     edgecolor='black',
                #     facecolor='green',
                #     alpha=0.3
                # )
                # ax.add_patch(square)

            plt.yscale("log")
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.title(
                f"{title} {comet}\n"
                # f"{data['Sim_info'][sim]} clones, Integrator: {info['Integrator']}, timestep: {info['timestep']} sec"
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            if save_name:
                plt.savefig(f"{save_name}.pdf", dpi=300)
            plt.close() 

        # ------------------------------------------------------
        # Position boxplot
        make_boxplot(
            stats["Clone_Divergence_Norm_peri_DPERI"][comet],
            height=1e3,
            ylabel="Clone Position Divergence Norm at perihelion [km]",
            title="Clone Position Divergence vs. Days to Perihelion",
            scale=1e3,
            save_name=f"{self.saving_dir}/Stat_Position_boxplot_dtPeri"
        )

        # ------------------------------------------------------
        # Velocity boxplot
        make_boxplot(
            stats["Clone_Divergence_Vel_Norm_peri_DPERI"][comet],
            height=1,
            ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
            title="Clone Velocity Divergence vs. Days to Perihelion",
            scale=1.0,
            save_name=f"{self.saving_dir}/Stat_Velocity_boxplot_dtPeri"
        )

        # ------------------------------------------------------
        # N_obs boxplot
        make_boxplot(
            stats["Clone_Divergence_Norm_peri_NOBS"][comet],
            height=1e3,
            ylabel="Clone Position Divergence Norm at perihelion [km]",
            title="Clone Position Divergence Norm vs. Number of Observations",
            scale=1e3,
            xlabel="Number of Observations",
            save_name=f"{self.saving_dir}/Stat_Position_boxplot_NOBS"
        )

        make_boxplot(
            stats["Clone_Divergence_Vel_Norm_peri_NOBS"][comet],
            height=1,
            ylabel="Clone Velocity Divergence Norm at perihelion [m/s]",
            title="Clone Velocity Divergence Norm vs. Number of Observations",
            scale=1.0,
            xlabel="Number of Observations",
            save_name=f"{self.saving_dir}/Stat_Velocity_boxplot_NOBS"
        )
    
    def fit(self):        
        Truth_trajectory = self.data["Truth_Reference_trajectory"]
        Fit_trajectory = self.data["Estimated_Reference_trajectory"]
        Times = (
            np.array(self.data["Truth_Reference_trajectory_times"]) / (86400 * 365.25) + 2000
        )
        for sim, traj in Fit_trajectory.items():
            Fit_pos = np.array(traj)[:, 0:3]
            Truth_pos = np.array(Truth_trajectory)[:, 0:3]

            diff_elements = Fit_pos - Truth_pos
        
            diff = np.linalg.norm(diff_elements[:,:3],axis=1)
            fig, axs = plt.subplots(4, 1, figsize=(10, 8))

            labels = ['x (km)', 'y (km)', 'z (km)']
            for i in range(3):
                axs[i].plot(Times, diff_elements[:, i] / 1e3, color='tab:blue')
                axs[i].set_ylabel(labels[i])
                axs[i].grid(True)

            axs[2].set_xlabel('Time [Calender]')

            distance_sorted = sorted(np.linalg.norm(Truth_trajectory,axis=1) / AU, reverse=True)
            axs[3].plot(distance_sorted, diff / 1e3, color='tab:orange')
            axs[3].set_ylabel(r'$||r_{diff}||$ (km)')
            axs[3].grid(True)
            axs[3].set_xlabel('Distance [AU]')
            axs[3].invert_xaxis()

            fig.suptitle(f'Difference between fitted orbit and truth orbit, {self.data["Sim_info"][sim].get("N_obs")} observations')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"{self.saving_dir}/Fit_to_Truth/{sim}_Fit_to_Truth.pdf", dpi=300)
            plt.close() 

    def ref_to_spice(self):
        Truth_trajectory = self.data["Truth_Reference_trajectory"]
        Spice = self.data["Spice_Reference_trajectory"]
        Times = (
            np.array(self.data["Truth_Reference_trajectory_times"]) / (86400 * 365.25) + 2000
        )

        Spice = np.array(Spice)[:, 0:3]
        Truth_pos = np.array(Truth_trajectory)[:, 0:3]

        diff_elements = Spice - Truth_pos
    
        diff = np.linalg.norm(diff_elements[:,:3],axis=1)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        labels = ['x (km)', 'y (km)', 'z (km)']
        for i in range(3):
            axs[i].plot(Times, diff_elements[:, i] / 1e3, color='tab:blue')
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)

        axs[2].set_xlabel('Time [Calender]')

        distance_sorted = sorted(np.linalg.norm(Truth_trajectory,axis=1) / AU, reverse=True)
        axs[3].plot(distance_sorted, diff / 1e3, color='tab:orange')
        axs[3].set_ylabel(r'$||r_{diff}||$ (km)')
        axs[3].grid(True)
        axs[3].set_xlabel('Distance [AU]')
        axs[3].invert_xaxis()

        fig.suptitle(f'Difference between Truth and Spice')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"{self.saving_dir}/Truth_To_Spice.pdf", dpi=300)
        plt.close() 
    
    def clone_divergence(self):
        Truth_trajectory = self.data["Truth_Reference_trajectory"]
        Fit_trajectory = self.data["Estimated_Reference_trajectory"]
        Clone_trajectory = self.data["Montecarlo_trajectory"]

        Truth_pos = np.array(Truth_trajectory)[:, 0:3]


        Truth_norm = np.linalg.norm(Truth_pos,axis=1)
        for sim, sim_data in Clone_trajectory.items():

            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            fig.suptitle(f'Difference of the Fit and clones to the Truth orbit')
            for key, traj in sim_data.items():

                diff_norm = np.linalg.norm(Truth_pos-Clone_trajectory[sim][key][:, 0:3],axis=1)

                axs.plot(Truth_norm/AU,diff_norm/1000)

            Fit = np.array(Fit_trajectory[sim])[:, 0:3]
            diff_truth_to_fit = Truth_pos - Fit
            FIT_TRUTH_Dif = np.linalg.norm(diff_truth_to_fit, axis=1) 
            axs.plot(Truth_norm/AU,FIT_TRUTH_Dif/1000,label='Fitted orbit',color="black",linewidth=3.0)
            axs.invert_xaxis()
            plt.legend()
            plt.xlabel(f"Heliocentric distance [AU]")
            plt.ylabel(f"Divergence [km]")
            plt.grid()
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"{self.saving_dir}/Clone_divergence/{sim}_Clone_difference.pdf", dpi=300)
            plt.close() 

