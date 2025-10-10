import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


bodies = ['Q4', 'K4']
                    # "e",     "q",      "tp",    "node",    "peri",  "i",    "A1",      "A2",     "A3"
cov_JPL = np.array([[9.94e-13,4.3e-14,-4.15e-12,-3.89e-12,-6.51e-12,1.15e-12,-1.54e-16,-1.61e-17,9.5e-18],
                    [4.3e-14,9.08e-15,-1.44e-13,-1.85e-13,-1.99e-13,6.92e-14,-7.84e-18,-3.2e-19,4.51e-19],
                    [-4.15e-12,-1.44e-13,8.86e-11,1.82e-11,9.45e-11,-1.3e-11,6.91e-16,2.73e-16,-5.06e-17],
                    [-3.89e-12,-1.85e-13,1.82e-11,1.18e-10,4.46e-11,-5.77e-12,5.9e-16,1.6e-17,-3.05e-16],
                    [-6.51e-12,-1.99e-13,9.45e-11,4.46e-11,1.34e-10,-1.14e-11,1.07e-15,3.31e-16,-1.26e-16],
                    [1.15e-12,6.92e-14,-1.3e-11,-5.77e-12,-1.14e-11,7.69e-11,-1.72e-16,-1.91e-17,-1.35e-16],
                    [-1.54e-16,-7.84e-18,6.91e-16,5.9e-16,1.07e-15,-1.72e-16,2.58e-20,3.5e-21,-1.5e-21],
                    [-1.61e-17,-3.2e-19,2.73e-16,1.6e-17,3.31e-16,-1.91e-17,3.5e-21,5.93e-21,-2.4e-22],
                    [9.5e-18,4.51e-19,-5.06e-17,-3.05e-16,-1.26e-16,-1.35e-16,-1.5e-21,-2.4e-22,4.12e-21]])

cov_JPL_K4 = np.array([[1.24e-12,8.5e-13,-4.94e-11,2.92e-13,-7.28e-11,-4.5e-12,-2.64e-16,-7.07e-17,-6.64e-18],
                      [8.5e-13,8.44e-13,-2.5e-11,-2.64e-12,-6.13e-11,-3.79e-12,-2.93e-16,-1.69e-17,-7.39e-18],
                      [-4.94e-11,-2.5e-11,2.71e-09,-1.99e-10,2.55e-09,1.28e-10,5.99e-15,4.41e-15,-1.09e-16],
                      [2.92e-13,-2.64e-12,-1.99e-10,3.85e-10,3.29e-10,1.11e-11,1.22e-15,-4.15e-16,9.16e-16],
                      [-7.28e-11,-6.13e-11,2.55e-09,3.29e-10,4.96e-09,2.92e-10,2.03e-14,2.84e-15,1.06e-15],
                      [-4.5e-12,-3.79e-12,1.28e-10,1.11e-11,2.92e-10,5.07e-11,1.31e-15,1.37e-16,1.57e-16],
                      [-2.64e-16,-2.93e-16,5.99e-15,1.22e-15,2.03e-14,1.31e-15,1.13e-19,-3.58e-21,1.5e-21],
                      [-7.07e-17,-1.69e-17,4.41e-15,-4.15e-16,2.84e-15,1.37e-16,-3.58e-21,1.11e-20,6.94e-22],
                      [-6.64e-18,-7.39e-18,-1.09e-16,9.16e-16,1.06e-15,1.57e-16,1.5e-21,6.94e-22,5.72e-21]])

cov_JPL_VZ13 = np.array([[3.3e-14,-1.13e-13,-3.44e-11,1.6e-13,-7.23e-12,-5.45e-14],[-1.13e-13,3.87e-13,1.18e-10,-5.39e-13,2.46e-11,1.88e-13],[-3.44e-11,1.18e-10,6.46e-08,-5.16e-10,1.49e-08,5.08e-11],[1.6e-13,-5.39e-13,-5.16e-10,4.77e-08,-4.78e-08,5.14e-10],[-7.23e-12,2.46e-11,1.49e-08,-4.78e-08,5.12e-08,-5.03e-10],[-5.45e-14,1.88e-13,5.08e-11,5.14e-10,-5.03e-10,4.32e-11]])
correlations = np.corrcoef(cov_JPL)

estimated_param_names = [ "e",  "q",  "tp",  "RAAN",    "AoP",  "i",    "A1",      "A2",     "A3"]

fig, ax = plt.subplots(1, 1, figsize=(9, 7))

im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

# add numbers to each of the boxes
for i in range(len(estimated_param_names)):
    for j in range(len(estimated_param_names)):
        text = ax.text(
            j, i, round(correlations[i, j], 2), ha="center", va="center", color="w"
        )

cb = plt.colorbar(im)

ax.set_xlabel("Estimated Parameter")
ax.set_ylabel("Estimated Parameter")

fig.suptitle(f"Correlations for estimated parameters for {'K4'} (JPL)")

fig.set_tight_layout(True)

plt.show()