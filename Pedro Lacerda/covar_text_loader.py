# loading packages
import numpy as np

# Import systems
import sys
import time as timer
import pickle
import re
import os
import glob
import json
import pprint
import pandas as pd

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
    
    covar_files = sorted(glob.glob(os.path.join(base_path, f"covar_{comet}_*.txt")))
    total_files = sorted(glob.glob(os.path.join(base_path, f"total_{comet}_*.json")))
    
    paired_total_files = []
    for covar_file in covar_files:
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".json", "")
        covar_date = os.path.basename(covar_file).split(f"{comet}_")[1].replace(".txt", "")

        matching_total = [tf for tf in total_files if covar_date in tf]
        if matching_total:
            paired_total_files.append(matching_total[0])
    
    files_dict[comet]["covar"] = covar_files
    files_dict[comet]["total"] = paired_total_files

for comet, data in files_dict.items():
    for covar_file, total_file in zip(data['covar'], data['total']):
        def extract_traditional_covariance(filepath):
            with open(filepath, "r") as f:
                lines = f.readlines()

            start_idx = None
            for i, line in enumerate(lines):
                if re.search(r'\bTp\b.*\be\b.*\bq\b.*\bQ\b.*1/a', line):
                    start_idx = i
                    break

            data_lines = []
            for line in lines[start_idx+2:]:
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
            return df_cov
        cov_df = extract_traditional_covariance(covar_file)
        cov_df = cov_df.drop(index=["MOID", "H","Q","M"], columns=["MOID", "H"])

        print(cov_df)
