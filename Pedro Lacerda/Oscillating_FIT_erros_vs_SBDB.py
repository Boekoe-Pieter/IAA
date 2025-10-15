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

files_dict = {}
for comet in comets:
    files_dict[comet] = {"covar": [], "total": []}
    
    covar_files = sorted(glob.glob(os.path.join(base_path, f"covar_{comet}_*.json")))
    total_files = sorted(glob.glob(os.path.join(base_path, f"total_{comet}_*.json")))
    
    paired_total_files = []
    for covar_file in covar_files:
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".json", "")
        
        matching_total = [tf for tf in total_files if covar_date in tf]
        if matching_total:
            paired_total_files.append(matching_total[0])
    
    files_dict[comet]["covar"] = covar_files
    files_dict[comet]["total"] = paired_total_files



JPL_SBDB_dict = {}
Lacerdas_dict = {}

for comet, data in files_dict.items():
    JPL_SBDB_dict[comet] = {} 
    Lacerdas_dict[comet] = {"models": []}

    for covar_file, total_file in zip(data["covar"], data["total"]):
        with open(covar_file) as f:
            covar_states = json.loads(f.read())
        with open(total_file) as f:
            total_data = json.loads(f.read())

        body = total_data.get("ids")[0]

        # ------------------------------------------------------------------------
        # NASA JPL SBDB 
        target_sbdb = SBDBquery(body, full_precision=True)

        e      = target_sbdb["orbit"]["elements"].get("e") 
        e_sig  = target_sbdb["orbit"]["elements"].get("e_sig")
        a      = target_sbdb["orbit"]["elements"].get("a").value 
        a_sig  = target_sbdb["orbit"]["elements"].get("a_sig").value 
        q      = target_sbdb["orbit"]["elements"].get("q").value 
        q_sig  = target_sbdb["orbit"]["elements"].get("q_sig").value 
        i      = target_sbdb["orbit"]["elements"].get("i").value
        i_sig  = target_sbdb["orbit"]["elements"].get("i_sig").value
        om     = target_sbdb["orbit"]["elements"].get("om").value
        om_sig = target_sbdb["orbit"]["elements"].get("om_sig").value
        w      = target_sbdb["orbit"]["elements"].get("w").value
        w_sig  = target_sbdb["orbit"]["elements"].get("w_sig").value

        SBDB = np.array([e if e is not None else 0, 
                        a if a is not None else 0, 
                        q if q is not None else 0, 
                        i if i is not None else 0, 
                        om if om is not None else 0, 
                        w if w is not None else 0,])
        
        SBDB_sig = np.array([e_sig if e_sig is not None else 0,
                            a_sig if a_sig is not None else 0,
                            q_sig if q_sig is not None else 0,
                            i_sig if i_sig is not None else 0,
                            om_sig if om_sig is not None else 0,
                            w_sig if w_sig is not None else 0,
                            ])
        
        JPL_SBDB_dict[comet] = {
            "array": SBDB,
            "array_sig": SBDB_sig
        }

        # ------------------------------------------------------------------------
        #  LACERDA 
        e = total_data["objects"][body]["elements"].get("e")
        e_sig = total_data["objects"][body]["elements"].get("e sigma")
        a = total_data["objects"][body]["elements"].get("a")
        a_sig = total_data["objects"][body]["elements"].get("a sigma")
        q = total_data["objects"][body]["elements"].get("q")
        q_sig = total_data["objects"][body]["elements"].get("q sigma")
        i = total_data["objects"][body]["elements"].get("i")
        i_sig = total_data["objects"][body]["elements"].get("i sigma")
        w = total_data["objects"][body]["elements"].get("arg_per")
        w_sig = total_data["objects"][body]["elements"].get("arg_per sigma")
        om = total_data["objects"][body]["elements"].get("asc_node")
        om_sig =total_data["objects"][body]["elements"].get("asc_node sigma")
        Tp = total_data["objects"][body]["elements"].get("Tp")

        N_obs = total_data["objects"][body]["observations"].get("used")
        Last_obs = total_data["objects"][body]["observations"].get("latest_used")

        dt_to_pericenter = Tp-Last_obs
        
        Lacerda = np.array([e if e is not None else 0, 
                            a if a is not None else 0, 
                            q if q is not None else 0, 
                            i if i is not None else 0, 
                            om if om is not None else 0, 
                            w if w is not None else 0,])
        
        Lacerda_sig = np.array([e_sig if e_sig is not None else 0,
                                a_sig if a_sig is not None else 0,
                                q_sig if q_sig is not None else 0,
                                i_sig if i_sig is not None else 0,
                                om_sig if om_sig is not None else 0,
                                w_sig if w_sig is not None else 0,
                                ])
        
        Lacerda_obs = np.array([Last_obs, Tp, dt_to_pericenter])

        # Save all model runs in a list
        Lacerdas_dict[comet]["models"].append({
            "array": Lacerda,
            "array_sig": Lacerda_sig,
            "obs": Lacerda_obs
        })


# -----------------------------------------------------------------
# Plotting
elements = ["e", "a", "q", "i", "om", "w"]

for comet, data in Lacerdas_dict.items():
    models_nominal = np.array([m["array"] for m in data["models"]])      
    models_sigma   = np.array([m["array_sig"] for m in data["models"]])
    dt_from_peri   = np.array([m["obs"][2] for m in data["models"]])     

    jpl_nominal = JPL_SBDB_dict[comet]["array"]
    jpl_sigma   = JPL_SBDB_dict[comet]["array_sig"]

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"{comet} — Oscillating elements from observation fits and JPL SBDB", fontsize=14)

    for idx, el in enumerate(elements):
        ax = axs[idx]

        ax.errorbar(dt_from_peri, models_nominal[:, idx],
                    yerr=models_sigma[:, idx],
                    fmt='o', ms=4, mfc='royalblue', mec='navy',
                    ecolor='lightblue', elinewidth=1.5, capsize=3,
                    label='Lacerda models')

        jpl_val = jpl_nominal[idx]
        jpl_sig = jpl_sigma[idx]
        ax.set_xlim(dt_from_peri[-1]-10,dt_from_peri[0]+10)
        ax.axhline(jpl_val, color='black', linestyle='--', label='JPL nominal')
        ax.fill_between(ax.get_xlim(), jpl_val - jpl_sig, jpl_val + jpl_sig,
                        color='red', alpha=0.2, label=r'JPL $\pm\epsilon$')

        ax.set_ylabel(el)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    axs[-1].set_xlabel(r"$\Delta{t}$ from perihelion [days]")
    axs[0].legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Pedro Lacerda/{comet}/oscillating_elements.pdf",dpi=300)
    plt.show()
    plt.close()

    models_nominal = np.array([m["array"] for m in data["models"]])
    models_sigma   = np.array([m["array_sig"] for m in data["models"]])
    dt_from_peri   = np.array([m["obs"][2] for m in data["models"]])

    jpl_nominal = JPL_SBDB_dict[comet]["array"]
    jpl_sigma   = JPL_SBDB_dict[comet]["array_sig"]

    for idx, el in enumerate(elements):
        mask = ~np.isnan(models_nominal[:, idx]) & ~np.isnan(models_sigma[:, idx])
        x = dt_from_peri[mask]
        y = models_nominal[mask, idx]
        yerr = models_sigma[mask, idx]

        plt.figure(figsize=(10, 5))
        plt.errorbar(x, y, yerr=yerr,
                     fmt='o', ms=4, mfc='royalblue', mec='navy',
                     ecolor='lightblue', elinewidth=1.5, capsize=3,
                     label='Lacerda models')

        jpl_val = jpl_nominal[idx]
        jpl_sig = jpl_sigma[idx]
        plt.axhline(jpl_val, color='black', linestyle='--', label='JPL nominal')
        plt.fill_between([x.min()-1, x.max()+1], jpl_val - jpl_sig, jpl_val + jpl_sig,
                         color='red', alpha=0.2, label=r'JPL $\pm\epsilon$')

        plt.xlabel(r"$\Delta{t}$ from perihelion [days]")
        plt.ylabel(el)
        plt.title(f"{comet} — Oscillating element $\\mathit{{{el}}}$ evolution")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.savefig(f"Pedro Lacerda/{comet}/oscillating_{el}.pdf",dpi=300)
        plt.show()
        plt.close()
