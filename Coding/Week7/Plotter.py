import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

from tudatpy import constants

AU = constants.ASTRONOMICAL_UNIT
day = constants.JULIAN_DAY

class observations_Plotter:
    def __init__(self,name,simulated_observations):
        self.name = name

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

    def RADEC_overtime(self,directory_name,addition):
        # ----------------------------------------------------------------------
        # RA/DEC over time
        plt.figure(figsize=(10,4))
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
        plt.savefig(f"{directory_name}{addition}/Obs_RADEC_time.pdf", dpi=300)
        plt.close()
    
    def skyplot(self,directory_name,addition):
        # ----------------------------------------------------------------------
        # RA/DEC Skyplot
        plt.figure(figsize=(7,7))
        sc = plt.scatter(self.ra_deg, self.dec_deg, c=self.time_days, cmap='viridis', s=2)
        plt.xlabel("Right Ascension [deg]")
        plt.ylabel("Declination [deg]")
        plt.title(f"Sky Track, comet:{self.name},Nobs: {len(self.ra_deg)}")
        plt.gca().invert_xaxis()  
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label("Time [days]")
        plt.savefig(f"{directory_name}{addition}/Obs_RADEC_skyplot.pdf", dpi=300)
        plt.close()

    def aitoff(self,directory_name,addition):
        # ----------------------------------------------------------------------
        # Aitoff Projection
        ra_rad = np.deg2rad(self.ra_deg)
        dec_rad = np.deg2rad(self.dec_deg)

        plt.figure(figsize=(10,5))
        ax = plt.subplot(111, projection='aitoff')

        sc = ax.scatter(ra_rad, dec_rad, c=self.time_days, cmap='viridis', s=5)

        ax.set_title("Comet Sky Track", va='bottom')
        plt.grid(True)

        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label("Time [days]")

        plt.tight_layout()
        plt.savefig(f"{directory_name}{addition}/Obs_RADEC_Aitoff.pdf", dpi=300)
        plt.close()
class estimation_plotter:
    def __init__(self,name,pod_it,pod_output,simulated_observations,covariance_output,parameters_to_estimate,state_transition):
        self.name = name

        self.simulated_observations = simulated_observations
        self.pod_output = pod_output
        self.pod_it = pod_it

        self.covariance_output = covariance_output
        self.parameters_to_estimate = parameters_to_estimate
        self.state_transition = state_transition

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

    def residuals(self,directory_name,addition,simulated_observations):
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
        plt.savefig(f"{directory_name}{addition}/Est_Residuals.pdf", dpi=300)
        plt.close()

    def correlation(self,directory_name,addition):
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
        plt.savefig(f"{directory_name}{addition}/Est_Corr_matrix.pdf", dpi=300)
        plt.close()

    def formal_erros(self,directory_name,addition):
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
        plt.savefig(f"{directory_name}{addition}/Est_Confidence.pdf", dpi=300)
        plt.close()

class statistics_plotter:
    def __init__(self):
        x=1
