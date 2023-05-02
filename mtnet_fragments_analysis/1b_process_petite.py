import os
import json
import re

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import skimage.io
import skimage.measure

import skimage.filters

import matplotlib.pyplot as plt

from utils import printl
import utils

from tqdm import tqdm

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

data_path = r'G:\My Drive\1_MIA_Data\Anika\Mutants\Petite'

json_paths_filenames = ['petite.json']

all_dfs = []
keys = []

for json_paths_filename in tqdm(json_paths_filenames, desc='Total', ncols=100):
    with open(os.path.join(pwd_path, 'json_paths', json_paths_filename)) as json_file:
        exp_paths = json.load(json_file)
    medium = exp_paths['medium']
    ploidy = exp_paths['ploidy']
    replicates = exp_paths['replicates']
    run_number = exp_paths['run_number']
    for replicate, exp_folders in tqdm(replicates.items(), desc='Replicates', ncols=100, position=1, leave=False):
        for exp_folder in tqdm(exp_folders, desc='Experiment', ncols=100, position=2, leave=False):
            exp_path = os.path.join(data_path, replicate, exp_folder, 'TIFFs')
            pos_foldernames = utils.get_pos_foldernames(exp_path)
            for pos in tqdm(pos_foldernames, desc='Positions', ncols=100, leave=False, position=3):
                images_path = os.path.join(exp_path, pos, 'Images')
                spotmax_path = os.path.join(exp_path, pos, 'spotMAX_output')
                segm_data = None
                for file in utils.listdir(images_path):
                    file_path = os.path.join(images_path, file)
                    if file.endswith('_mKate.tif'):
                        mKate_data = skimage.io.imread(file_path).astype(np.uint8)
                    elif file.endswith('_segm.npz'):
                        segm_data = np.load(file_path)['arr_0']
                    elif file.endswith('_segm.npy'):
                        segm_file_path = file_path
                        segm_filename = file
                        segm_data = np.load(segm_file_path)
       
                segm_data_3D = np.tile(segm_data, (len(mKate_data), 1, 1)).astype(np.uint16)

                # gaussian filter like in spotmax data
                mKate_data = (skimage.filters.gaussian(mKate_data, 0.75)*255).astype(np.uint8)

                # Get mother ID
                segm_rp = skimage.measure.regionprops(segm_data_3D)

                if len(segm_rp) > 2:
                    print('')
                    print('-'*40)
                    print(
                        f'The following experiment contains {len(segm_rp)} objects. '
                        'Skipping it.'
                        f'"{images_path}"'
                    )
                    print('*'*40)
                    continue

                IDs = [obj.label for obj in segm_rp]
                areas = [obj.area for obj in segm_rp]
                mothID = IDs[areas.index(max(areas))]

                # Replicate spotmax automatic thresholding
                segm_obj = skimage.measure.regionprops((segm_data_3D>0).astype(np.uint8))[0]
                spotmax_thresh_val = skimage.filters.threshold_li(mKate_data[segm_obj.slice].max(axis=0))

                # print('')
                # print(mothID, spotmax_thresh_val)
                # fig, ax = plt.subplots(1,3)
                # ax[0].imshow(segm_data_3D[segm_obj.slice].max(axis=0))
                # ax[1].imshow(mKate_data[segm_obj.slice].max(axis=0))
                # ax[2].imshow((mKate_data>spotmax_thresh_val)[segm_obj.slice].max(axis=0))
                # plt.show()
                # import pdb; pdb.set_trace()

                thresh_vals = []
                num_fragments = []
                IDs = []
                for thresh_val in tqdm(range(256), desc='Thresholding', leave=False, ncols=100, position=4):
                    thresh = mKate_data > thresh_val
                    segm_obj_thresh = thresh[segm_obj.slice].copy()
                    segm_obj_thresh[~segm_obj.image] = 0

                    segm_obj_thresh_lab = skimage.measure.label(segm_obj_thresh)
                    segm_obj_thresh_rp = skimage.measure.regionprops(segm_obj_thresh_lab)
                    num_fragments.append(len(segm_obj_thresh_rp))
                    thresh_vals.append(thresh_val)

                    # if thresh_val>30:
                    #     print('\n\n')
                    #     print(thresh_val, mKate_data.min(), len(segm_obj_thresh_rp), segm_obj_thresh_lab.max())
                    #     print('')
                    #     fig, ax = plt.subplots(1, 3)
                    #     zc = int(segm_obj.image.shape[0]/2)
                    #     ax[0].imshow(segm_obj_thresh[zc])
                    #     ax[1].imshow(segm_obj_thresh_lab.max(axis=0))
                    #     ax[2].imshow(mKate_data[segm_obj.slice][zc])
                    #     plt.show()
                    #     import pdb; pdb.set_trace()
                
                df = pd.DataFrame({
                    'threshold_value': thresh_vals,
                    'num_fragments': num_fragments, 
                    'spotmax_threshold_value': [spotmax_thresh_val]*len(num_fragments),
                    'Cell_ID': [mothID]*len(num_fragments)
                }).set_index('Cell_ID').sort_index()
                all_dfs.append(df)
                keys.append((
                    medium, ploidy, replicate, exp_folder, pos
                ))

final_df = pd.concat(
    all_dfs, keys=keys, 
    names=[
        'medium', 'ploidy', 'replicate_num', 'exp_folder', 'Position_n',  'Cell_ID'
    ]
).sort_index()

print(final_df)

if SAVE:
    tables_path = os.path.join(pwd_path, 'tables')
    df_filename = f'petite_mtNet_num_fragments.csv'
    df_filepath = os.path.join(tables_path, df_filename)
    final_df.to_csv(df_filepath)