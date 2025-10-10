# import python libraries
import numpy as np
from astropy.time import Time, TimeDelta
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

# def plot_difference(arcs, difference, x_scale, body, title,suptitle,saver):
#     fig, ax = plt.subplots(1, 1, figsize=(15, 8))

#     linestyles = ['-', '--', ':', '-.']

#     for j, arc in enumerate(arcs):
#         linestyle = linestyles[j % len(linestyles)]
#         models = list(difference[arc].keys())
#         for model in models: 
#             x_AU = x_scale[arc][model]
#             states_scaled = np.zeros_like(difference[arc][model])
#             states_scaled[:, :3] = difference[arc][model][:, :3] * const.au/1000

#             diff_norm = np.linalg.norm(states_scaled[:, :3], axis=1)

#             ax.plot(x_AU, diff_norm, linestyle=linestyle, label=f"{model} ({arc} data)")

#     ax.set_xlabel('Heliocentric distance [AU]')
#     ax.set_ylabel(r'$||\Delta{r}||$ [km]')
#     ax.grid(True, which="both")
#     ax.legend()

#     all_x = np.concatenate([
#         x_scale[arc][model] 
#         for arc in arcs 
#         for model in x_scale[arc].keys()
#     ])
#     x_min = np.min(all_x)
#     x_zoom_min = 0.99*x_min
#     x_zoom_max = x_min*1.5 

#     # ax.set_xlim(x_zoom_min, x_zoom_max)
#     ax.set_yscale('log')
#     fig.suptitle(f"{title} {body}",fontweight="bold")
#     fig.text(0.5, 0.90, suptitle, ha="center")
#     plt.savefig(f"Coding/W2P2/{re.sub(r'[^\w\-_\. ]', '_', body)}_Cartesian_{saver}.pdf", dpi=300)
#     # plt.show()
#     plt.close(fig)

# def plot_NGA_acc_MJD(arcs, F_NGA, time, body, MJD_perih,suptitle):
#     fig, axes = plt.subplots(1, 4, figsize=(18, 8))
#     axes = axes.flatten()

#     labels = [r'$a_R$ [m/s²]', r'$a_T$ [m/s²]', r'$a_N$ [m/s²]']
#     fig.suptitle(f"NGA components of {body} in RTN/RWS")

#     linestyles = ['-', '--', ':', '-.']  

#     for j_arc, arc in enumerate(arcs):
#         linestyle = linestyles[j_arc % len(linestyles)]
#         models = list(F_NGA[arc].keys())

#         for k_model, model in enumerate(models[:]): 
#             states_scaled = F_NGA[arc][model] * const.au/(const.day**2)

#             norm = np.linalg.norm(states_scaled, axis=1)
#             time_scaled = time - float(calender_to_MDJ(MJD_perih[arc][model]['T']))

#             for i in range(3):
#                 axes[i].plot(time_scaled, states_scaled[:, i],
#                              label=f"{model} ({arc})",
#                              linestyle=linestyle,
#                              )
#                 axes[i].set_ylabel(labels[i])
#                 axes[i].grid(True, which='both')
#                 axes[i].set_xlabel("MJD (centered at perihelion)")

#             axes[3].plot(time_scaled, norm,
#                          label=f"{model} ({arc})",
#                          linestyle=linestyle,
#                          )
            
#             axes[3].grid(True, which='both')
#             axes[3].set_xlabel("MJD (centered at perihelion)")
#             axes[3].set_ylabel(r"$||a||$ [m/s²]")

#     for ax in axes:
#         ax.legend()
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     fig.text(0.5, 0.90, suptitle, ha="center")

#     safe_name = re.sub(r'[^\w\-_\. ]', '_', body)

#     plt.savefig(f"Coding/W2P2/{safe_name}_NGA_components_MJD_RTN.pdf", dpi=300)
#     # plt.show()
#     plt.close(fig)

# def plot_gr_AU(arcs, g_r, x_scale, body, MJD_perih,suptitle):
#     fig, axes = plt.subplots(1, 1, figsize=(18, 8))
#     fig.suptitle(f"g(r) function of {body}")

#     linestyles = ['-', '--', ':', '-.']  

#     for j_arc, arc in enumerate(arcs):
#         linestyle = linestyles[j_arc % len(linestyles)]
#         models = list(g_r[arc].keys())
#         for model in models: 
#             x_AU = x_scale[arc][model]

#             axes.plot(x_AU, g_r[arc][model],
#                       label = rf"{model} ({arc}), $\tau$:{MJD_perih[arc][model]['tau']}",
#                       linestyle=linestyle,
#                             )
#             axes.set_ylabel('g(r)')
#             axes.grid(True, which='both')
#             axes.set_xlabel("AU from perihelion")
#             axes.legend()
#     all_x = np.concatenate([
#         x_scale[arc][model] 
#         for arc in arcs 
#         for model in x_scale[arc].keys()
#     ])
#     x_min = np.min(all_x)
#     x_zoom_min = 0.99*x_min
#     x_zoom_max = 50 

#     # axes.set_xlim(x_zoom_min, x_zoom_max)
#     axes.set_yscale('log')
#     fig.tight_layout(rect=[0, 0, 1, 0.96])

#     safe_name = re.sub(r'[^\w\-_\. ]', '_', body)
#     fig.text(0.5, 0.90, suptitle, ha="center")

#     plt.savefig(f"Coding/W2P2/{safe_name}_gr_AU.pdf", dpi=300)
#     # plt.show()
#     plt.close(fig)

# def plot_gr_MJD(arcs, g_r, time, body, MJD_perih,suptitle):
#     fig, axes = plt.subplots(1, 1, figsize=(18, 8))
#     fig.suptitle(f"g(r) function of {body}")

#     linestyles = ['-', '--', ':', '-.']  

#     taus = [] 

#     for j_arc, arc in enumerate(arcs):
#         linestyle = linestyles[j_arc % len(linestyles)]
#         models = list(g_r[arc].keys())
#         for model in models: 
#             tau = float(MJD_perih[arc][model]['tau'])
#             taus.append(tau)

#             time_scaled = time - float(calender_to_MDJ(MJD_perih[arc][model]['T']))
#             axes.plot(time_scaled, g_r[arc][model],
#                       label = rf"{model} ({arc}), $\tau$:{tau}",
#                       linestyle=linestyle,
#                       )

#     axes.set_ylabel('g(r)')
#     axes.set_xlabel("MJD from perihelion")
#     axes.grid(True, which='both')
#     axes.legend()
#     axes.set_yscale('log')

#     if taus:
#         tau_max = np.max(np.abs(taus))
#         if tau_max == 0:
#             margin = 20
#         else:
#             margin = 5
#         # axes.set_xlim(-tau_max - margin, tau_max + margin)

#     fig.text(0.5, 0.90, suptitle, ha="center")

#     fig.tight_layout(rect=[0, 0, 1, 0.96])

#     safe_name = re.sub(r'[^\w\-_\. ]', '_', body)
#     plt.savefig(f"Coding/W2P2/{safe_name}_gr_MJD.pdf", dpi=300)
#     plt.close(fig)
#     # plt.show()

# def scatter(NGA_data):
#     for comet, phases in NGA_data.items():
#         fig = plt.figure(figsize=(8,6))
#         ax = fig.add_subplot(111, projection='3d')
        
#         for phase_name, models in phases.items():
#             for model_name, data in models.items():
#                 ax.scatter(data['A1'], data['A2'], data['A3'], 
#                         label=f"{phase_name}:{model_name}", s=80)
        
#         ax.set_xlabel('A1')
#         ax.set_ylabel('A2')
#         ax.set_zlabel('A3')
#         ax.set_title(f'NGA Accelerations for {comet}')
#         ax.legend(fontsize=8, loc='upper left')
        
#         plt.tight_layout()
#         safe_name = re.sub(r'[^\w\-_\. ]', '_', comet)

#         plt.savefig(f"Coding/A_scatter/scatter of NGA accelerations {safe_name}")
#         plt.close(fig)

#     comet_names = []
#     A1_vals, A2_vals, A3_vals = [], [], []
#     labels = []

#     for comet, phases in NGA_data.items():
#         for phase_name, models in phases.items():
#             for model_name, data in models.items():
#                 comet_names.append(comet)
#                 A1_vals.append(data['A1'])
#                 A2_vals.append(data['A2'])
#                 A3_vals.append(data['A3'])
#                 labels.append(f"{comet},{model_name}")

#     fig = plt.figure(figsize=(10,6))
#     ax = fig.add_subplot(111, projection='3d')

#     sc = ax.scatter(A1_vals, A2_vals, A3_vals, c=range(len(A1_vals)),
#                     cmap='tab20', s=60)

#     ax.set_xlabel('A1 (10^-8 AU/day^2)')
#     ax.set_ylabel('A2 (10^-8 AU/day^2)')
#     ax.set_zlabel('A3 (10^-8 AU/day^2)')
#     ax.set_title('NGA Accelerations for Comets and Models')

#     for i, txt in enumerate(labels):
#         ax.text(A1_vals[i], A2_vals[i], A3_vals[i], txt, fontsize=8)

#     plt.colorbar(sc, label='Model index')
#     # plt.show()