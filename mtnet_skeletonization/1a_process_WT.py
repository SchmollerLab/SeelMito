import os
import json
import re

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.morphology

import matplotlib.pyplot as plt

from utils import printl
import utils

from tqdm import tqdm

from acdctools.plot import imshow

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

data_path = r'G:\My Drive\1_MIA_Data\Anika\WTs'

json_paths_filenames = [
    'SCD_Haploid_paths.json', 
    'SCD_Diploid_paths.json', 
    'SCGE_Diploid_paths.json',
    'SCGE_Haploid_paths.json'
]

all_dfs = []
keys = []

for json_paths_filename in tqdm(json_paths_filenames, desc='Total', ncols=100):
    with open(os.path.join(pwd_path, 'json_paths', json_paths_filename)) as json_file:
        exp_paths = json.load(json_file)
    medium = exp_paths['medium']
    ploidy = exp_paths['ploidy']
    replicates = exp_paths['replicates']
    run_number = exp_paths['run_number']
    spotmax_df_filename = f'spotMAX_Anika_WT_{medium}_{ploidy}_TOT_data.csv'
    spotmax_df_path = os.path.join(pwd_path, 'spotmax_final_tables', spotmax_df_filename)
    df = (
        pd.read_csv(spotmax_df_path)
        .set_index(['replicate_num', 'horm_conc', 'Position_n', 'Moth_ID'])
        .sort_index()
    )
    df['mtnet_skeleton_length_voxels'] = 0
    for replicate, exp_folders in tqdm(replicates.items(), desc='Replicates', ncols=100, position=1, leave=False):
        for exp_folder in tqdm(exp_folders, desc='Experiment', ncols=100, position=2, leave=False):
            exp_path = os.path.join(
                data_path, medium, f'{ploidy}s', replicate, exp_folder, 'TIFFs'
            )
            pos_foldernames = utils.get_pos_foldernames(exp_path)
            for pos in tqdm(pos_foldernames, desc='Positions', ncols=100, leave=False, position=3):
                images_path = os.path.join(exp_path, pos, 'Images')
                spotmax_path = os.path.join(exp_path, pos, 'spotMAX_output')
                segm_data = None
                cca_df_filepath = None
                acdc_df_filepath = None
                for file in utils.listdir(images_path):
                    file_path = os.path.join(images_path, file)
                    if file.endswith('_mKate.tif'):
                        mKate_data = skimage.io.imread(file_path).astype(np.uint8)
                    elif file.endswith('_mKate_mask.npz'):
                        mtnet_mask = np.load(file_path)['arr_0']
                    elif file.endswith('_segm.npz'):
                        segm_data = np.load(file_path)['arr_0']
                    elif file.endswith('_segm.npy'):
                        segm_file_path = file_path
                        segm_filename = file
                        segm_data = np.load(segm_file_path)
                    elif file.endswith('_cc_stage.csv'):
                        cca_df_filepath = file_path
                    elif file.endswith('_acdc_output.csv'):
                        acdc_df_filepath = file_path

                if cca_df_filepath is not None:
                    cca_df = pd.read_csv(cca_df_filepath).set_index('Cell_ID')
                    cca_df = cca_df.rename(columns={
                        'Cell cycle stage': 'cell_cycle_stage',
                        'Relative\'s ID': 'relative_ID',
                        'Relationship': 'relationship'
                    })
                else:
                    acdc_df = pd.read_csv(acdc_df_filepath).set_index('Cell_ID')
                    cca_df = acdc_df[
                        ['cell_cycle_stage', 'relative_ID', 'relationship']
                    ]

                segm_data_3D = np.tile(segm_data, (len(mKate_data), 1, 1)).astype(np.uint16)

                # gaussian filter like in spotmax data
                mKate_data = (skimage.filters.gaussian(mKate_data, 0.75)*255).astype(np.uint8)

                # Get mother ID
                segm_rp = skimage.measure.regionprops(segm_data_3D)
                IDs = [obj.label for obj in segm_rp]
                
                skel_lengths = []
                for obj in segm_rp:
                    relationship = cca_df.at[obj.label, 'relationship']
                    if relationship == 'bud':
                        continue
                    
                    skel_len, mtnet_skel_obj = utils.get_skel_len(mtnet_mask, obj)

                    ccs = cca_df.at[obj.label, 'cell_cycle_stage']
                    if ccs == 'S':
                        budID = cca_df.at[obj.label, 'relative_ID']
                        bud_idx = IDs.index(budID)
                        obj_bud = segm_rp[bud_idx]
                        bud_skel_len = utils.get_skel_len(mtnet_mask, obj_bud)[0]
                        skel_len += bud_skel_len
                    
                    idx = (replicate, exp_folder, pos, obj.label)
                    df.at[idx, 'mtnet_skeleton_length_voxels'] = skel_len
                    
    df.to_csv(spotmax_df_path)