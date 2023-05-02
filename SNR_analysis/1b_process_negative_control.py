import os
import json
import re

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import skimage.io

import matplotlib.pyplot as plt

from utils import printl
import utils
import core

from tqdm import tqdm

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

data_path = r'G:\My Drive\1_MIA_Data\Anika\NSMB_2022_revision_data\NegativeControl\ASY13_yCO381_control'

json_paths_filenames = [
    'Negative_control.json'
]

all_dfs = []
keys = []

for json_paths_filename in tqdm(json_paths_filenames, ncols=100):
    with open(os.path.join(pwd_path, 'json_paths', json_paths_filename)) as json_file:
        exp_paths = json.load(json_file)
    strain = exp_paths['strain']
    replicates = exp_paths['replicates']
    run_number = exp_paths['run_number']
    for exp_folder in tqdm(replicates, ncols=100, position=1, leave=False):
        exp_path = os.path.join(
            data_path, strain, exp_folder, 'TIFFs'
        )
        pos_foldernames = utils.get_pos_foldernames(exp_path)
        for pos in tqdm(pos_foldernames, ncols=100, leave=False, position=3):
            images_path = os.path.join(exp_path, pos, 'Images')
            spotmax_path = os.path.join(exp_path, pos, 'spotMAX_output')
            for file in utils.listdir(images_path):
                file_path = os.path.join(images_path, file)
                if file.endswith('_mKate_mask.npz'):
                    mtnet_mask = np.load(file_path)['arr_0']
                elif file.endswith('_mKate.tif'):
                    mKate_data = skimage.io.imread(file_path)
                elif file.endswith('_EGFP.tif'):
                    mNeon_data = skimage.io.imread(file_path)
                elif file.endswith('_segm.npz'):
                    segm_data = np.load(file_path)['arr_0']
                    
            segm_data_3D = np.tile(segm_data, (len(mKate_data), 1, 1))

            df_spots = None
            for file in utils.listdir(spotmax_path):
                if file == f'{run_number}_3_p-_ellip_test_data_v1.h5':
                    file_path = os.path.join(spotmax_path, file)
                    try:
                        df_spots = pd.read_hdf(file_path, key='frame_0')
                    except Exception as e:
                        # Zero spots --> skip position
                        continue
                elif file == f'{run_number}_v1_analysis_inputs.csv':
                    file_path = os.path.join(spotmax_path, file)
                    df_inputs = pd.read_csv(file_path).set_index('Description')
            
            if df_spots is None:
                continue

            spots_mask = np.zeros_like(mtnet_mask)
            try:
                spots_mask = core.global_spot_mask(df_inputs, df_spots, spots_mask)
            except Warning:
                import pdb; pdb.set_trace()

            try:
                mKate_stain_indexes = core.compute_stain_index(
                    mKate_data, mtnet_mask, segm_data_3D
                )
                mNeon_stain_indexes = core.compute_stain_index(
                    mNeon_data, spots_mask, segm_data_3D
                )
                mNeon_inMtNet_stain_indexes = core.compute_stain_index(
                    mNeon_data, spots_mask, segm_data_3D, 
                    ref_ch_mask=mtnet_mask
                )
            except Warning:
                import pdb; pdb.set_trace()

            data = np.zeros((len(mNeon_stain_indexes), 3))
            df = pd.DataFrame(
                index=mNeon_stain_indexes.keys(), data=data, 
                columns=[
                    'mKate_stain_index', 'mNeon_inside_cell_stain_index',
                    'mNeon_inside_mtNet_stain_index'
                ]
            )
            mNeon_IDs = mNeon_stain_indexes.keys()
            mNeon_values = list(mNeon_stain_indexes.values())
            df.loc[mNeon_IDs, 'mNeon_inside_cell_stain_index'] = mNeon_values

            mNeon_in_mtnet_IDs = mNeon_inMtNet_stain_indexes.keys()
            mNeon_in_mtnet_values = list(mNeon_inMtNet_stain_indexes.values())
            df.loc[mNeon_IDs, 'mNeon_inside_mtNet_stain_index'] = mNeon_in_mtnet_values

            mKate_IDs = mKate_stain_indexes.keys()
            mKate_values = list(mKate_stain_indexes.values())
            df.loc[mKate_IDs, 'mKate_stain_index'] = mKate_values

            all_dfs.append(df)
            keys.append((strain, exp_folder, pos, run_number))

            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(spots_mask.max(axis=0))
            # ax[1].imshow(mtnet_mask.max(axis=0))
            # ax[2].imshow(segm_data_3D.max(axis=0))
            # plt.show()
            # import pdb; pdb.set_trace()

final_df = pd.concat(
    all_dfs, keys=keys, 
    names=['strain', 'exp_folder', 'Position_n', 'run_number', 'Cell_ID']
)

if SAVE:
    tables_path = os.path.join(pwd_path, 'tables')
    df_filename = f'negative_control_stain_index.csv'
    df_filepath = os.path.join(tables_path, df_filename)
    final_df.to_csv(df_filepath)