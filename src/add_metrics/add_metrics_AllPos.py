import os, subprocess, sys, re
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from natsort import natsorted
from skimage import io
from skimage.measure import label, regionprops
from skimage.filters import gaussian
from tqdm import tqdm
from scipy.stats import entropy, power_divergence, ks_2samp, anderson_ksamp

script_dirpath = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_dirpath))
sys.path.insert(0, src_path)

import prompts

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)

src_listdir = os.listdir(src_path)
main_idx = [i for i, f in enumerate(src_listdir) if f.find('main_') !=- 1][0]
main_filename = src_listdir[main_idx]
NUM = re.findall('v(\d+).py', main_filename)[0]
vNUM = f'v{NUM}'

# Select experiment path
selected_path = prompts.folder_dialog(title=
    'Select the folder containing the Position_n folders')

if not selected_path:
    exit('Execution aborted.')

pos_paths = natsorted(
    [os.path.join(selected_path, name) for name in os.listdir(selected_path)
     if name.find('Position_') != -1
     and os.path.isdir(os.path.join(selected_path, name))]
)

if not pos_paths:
    tk.messagebox.showerror(
        'Not a valid path',
        f'The folder "{selected_path}" does not contain any Position folder.')
    raise FileNotFoundError('Position folder not found.')


main_paths = [selected_path]
scan_run_num = prompts.scan_run_nums(vNUM)
run_nums = scan_run_num.scan(pos_paths)
tot_pos = len(pos_paths)
if len(run_nums) > 1:
    run_num = scan_run_num.prompt(
        run_nums, msg='Select run numer to add metrics to: '
    )
else:
    run_num = 1

ellips_test_csvname = f'{run_num}_1_ellip_test_data_Summary_{vNUM}.csv'
p_test_csvname = f'{run_num}_2_p-_test_data_Summary_{vNUM}.csv'
p_ellips_test_csvname = f'{run_num}_3_p-_ellip_test_data_Summary_{vNUM}.csv'
spotFIT_csvname = f'{run_num}_4_spotfit_data_Summary_{vNUM}.csv'

ellips_test_h5name = f'{run_num}_1_ellip_test_data_{vNUM}.h5'
p_test_h5name = f'{run_num}_2_p-_test_data_{vNUM}.h5'
p_ellips_test_h5name = f'{run_num}_3_p-_ellip_test_data_{vNUM}.h5'
spotFIT_h5name = f'{run_num}_4_spotFIT_data_{vNUM}.h5'

all_paths = [
    {
    ellips_test_csvname: None,
    p_test_csvname: None,
    p_ellips_test_csvname: None,
    spotFIT_csvname: None,

    ellips_test_h5name: None,
    p_test_h5name: None,
    p_ellips_test_h5name: None,
    spotFIT_h5name: None
    } for _ in range(len(pos_paths))
]
# Get path to all summary df and h5 files
for p, pos_path in enumerate(pos_paths):
    spotmax_path = os.path.join(pos_path, 'spotMAX_output')
    ls = os.listdir(spotmax_path)
    for filename in ls:
        file_path = os.path.join(spotmax_path, filename)
        if filename.find(ellips_test_csvname) != -1:
            all_paths[p][ellips_test_csvname] = file_path

        elif filename.find(p_test_csvname) != -1:
            all_paths[p][p_test_csvname] = file_path

        elif filename.find(p_ellips_test_csvname) != -1:
            all_paths[p][p_ellips_test_csvname] = file_path

        elif filename.find(spotFIT_csvname) != -1:
            all_paths[p][spotFIT_csvname] = file_path


        elif filename.find(ellips_test_h5name) != -1:
            all_paths[p][ellips_test_h5name] = file_path

        elif filename.find(p_test_h5name) != -1:
            all_paths[p][p_test_h5name] = file_path

        elif filename.find(p_ellips_test_h5name) != -1:
            all_paths[p][p_ellips_test_h5name] = file_path

        elif filename.find(spotFIT_h5name) != -1:
            all_paths[p][spotFIT_h5name] = file_path


def addMetric(csv_path, h5_path):
    if csv_path is None:
        return
    if h5_path is None:
        return

    df_summary = pd.read_csv(csv_path, index_col=['Cell_ID'])
    df_h5 = pd.read_hdf(h5_path, '/frame_0')

    df_summary['num_spots_inside_ref_ch'] = 0

    for ID, df in df_h5.groupby(level=0):
        n_spots_inside_ref_ch = df['is_spot_inside_ref_ch'].astype(bool).sum()
        df_summary.at[ID, 'num_spots_inside_ref_ch'] = n_spots_inside_ref_ch

    df_summary.to_csv(csv_path)


"""Iterate each postion and extract additional metrics from the h5 file"""
print('')
print('Adding metrics...')
for pos_data_paths in tqdm(all_paths, ncols=100, unit=' Pos'):
    ellpis_test_csvpath = pos_data_paths[ellips_test_csvname]
    ellips_test_h5path = pos_data_paths[ellips_test_h5name]

    addMetric(ellpis_test_csvpath, ellips_test_h5path)

    p_test_csvpath = pos_data_paths[p_test_csvname]
    p_test_h5path = pos_data_paths[p_test_h5name]

    addMetric(p_test_csvpath, p_test_h5path)

    p_ellips_test_csvpath = pos_data_paths[p_ellips_test_csvname]
    p_ellips_test_h5path = pos_data_paths[p_ellips_test_h5name]

    addMetric(p_ellips_test_csvpath, p_ellips_test_h5path)

    spotFIT_csvpath = pos_data_paths[spotFIT_csvname]
    spotFIT_h5path = pos_data_paths[spotFIT_h5name]

    addMetric(spotFIT_csvpath, spotFIT_h5path)
