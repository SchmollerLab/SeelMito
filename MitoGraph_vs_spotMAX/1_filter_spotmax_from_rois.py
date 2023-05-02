import os
import json
import re
from roifile import ImagejRoi

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import skimage.draw
import traceback

import matplotlib.pyplot as plt

from utils import printl
import utils
import core

from tqdm import tqdm

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

data_path = r'G:\My Drive\1_MIA_Data\Anika\WTs'
all_rois_path = os.path.join(pwd_path, 'RoiSet_Francesco')

filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')
spotmax_final_df_folderpath = os.path.join(pwd_path, 'spotmax_final_tables')

json_paths_filenames = [
    'SCD_Diploid_paths.json', 'SCD_Haploid_paths.json',
    'SCGE_Diploid_paths.json', 'SCGE_Haploid_paths.json'
]

all_dfs = []
keys = []

for json_paths_filename in tqdm(json_paths_filenames, ncols=100, desc='json_file'):
    with open(os.path.join(pwd_path, 'json_paths', json_paths_filename)) as json_file:
        exp_paths = json.load(json_file)
    medium = exp_paths['medium']
    ploidy = exp_paths['ploidy']
    replicates = exp_paths['replicates']
    run_number = exp_paths['run_number']
    spotmax_final_df_foldername = f'AllExp_mitoQUANT_data_v1_run-num{run_number}'
    spotmax_df_filename = f'spotMAX_Anika_WT_{medium}_{ploidy[:-1]}_TOT_data'
    spotmax_final_df_path = os.path.join(spotmax_final_df_folderpath, f'{spotmax_df_filename}.csv')
    spotmax_final_df = pd.read_csv(
        spotmax_final_df_path
    ).set_index(['replicate_num', 'horm_conc', 'Position_n', 'Moth_ID'])
    # `replicate_num` column == `replicate`; `horm_conc` == `exp_folder`
    # while we have a spotmax table for each medium and ploidy
    filtered_df_filename = f'{spotmax_df_filename}_filtered_rois.csv'
    filtered_df_filepath = os.path.join(
        filtered_tables_path, filtered_df_filename
    )
    all_segm_data = {}
    all_rois_segm = {}
    spotmax_df_idx_to_keep = []
    spotmax_df_additional_cols = {
        'RoiSet_filename': [], 'roi_name': [], 'roi_intersection_over_union': []
    }
    for replicate, exp_folders in tqdm(replicates.items(), ncols=100, position=1, leave=False, desc='Replicates'):
        for exp_folder in tqdm(exp_folders, ncols=100, position=2, leave=False, desc='Experiment'):
            rois_path = os.path.join(
                all_rois_path, medium, ploidy, replicate, f'{exp_folder}.zip'
            )
            try:
                rois = ImagejRoi.fromfile(rois_path)
            except Exception as e:
                traceback.print_exc()
                import pdb; pdb.set_trace()
                exit()
            exp_path = os.path.join(
                data_path, medium, ploidy, replicate, exp_folder, 'TIFFs'
            )
            pos_foldernames = utils.get_pos_foldernames(exp_path)
            for pos in tqdm(pos_foldernames, ncols=100, leave=False, position=3, desc='Positions'):
                pos_num = int(pos[len('Position_'):])
                images_path = os.path.join(exp_path, pos, 'Images')
                spotmax_path = os.path.join(exp_path, pos, 'spotMAX_output')
                cc_stage_csv_path = ''
                acdc_df_csv_path = ''
                for file in utils.listdir(images_path):
                    file_path = os.path.join(images_path, file)
                    if file.endswith('_acdc_output.csv'):
                        acdc_df_csv_path = file_path
                    elif file.endswith('_cc_stage.csv'):
                        cc_stage_csv_path = file_path
                    elif file.endswith('_segm.npz'):
                        segm_file_path = file_path
                        segm_filename = file
                        segm_data = np.load(segm_file_path)['arr_0']
                    elif file.endswith('_segm.npy'):
                        segm_file_path = file_path
                        segm_filename = file
                        segm_data = np.load(segm_file_path)
                    elif file.endswith('_mNeon.tif'):
                        mNeon_filename = file
                
                m = re.findall(r'(\d+)_s(\d+)_', mNeon_filename)
                if m:
                    roi_idx = int(m[0][0])-1
                else:
                    roi_idx = 0
                    
                if cc_stage_csv_path:
                    acdc_df = pd.read_csv(cc_stage_csv_path)
                    relative_ID_col = "Relative's ID"
                    ccs_col = 'Cell cycle stage'
                    relationship_col = 'Relationship'
                else:
                    acdc_df = pd.read_csv(acdc_df_csv_path)
                    relative_ID_col = "relative_ID"
                    ccs_col = 'cell_cycle_stage'
                    relationship_col = 'relationship'
                
                # Merge mother and buds in segmentation data since Anika segmented
                # the ROIs with merged data
                for index, row in acdc_df.iterrows():
                    ID = row.Cell_ID
                    ccs = row[ccs_col]
                    if ccs == 'G1':
                        continue
                    if row[relationship_col]== 'bud':
                        continue
                    budID = row[relative_ID_col]
                    if budID > 0:
                        segm_data[segm_data==budID] = ID
                
                roi_mask = np.zeros(segm_data.shape, dtype=bool)
                try:
                    roi = rois[roi_idx]
                except Exception as e:
                    continue

                contour = roi.coordinates()
                rr, cc = skimage.draw.polygon(contour[:,1], contour[:,0])
                roi_mask[rr, cc] = True
                intersect_IDs, intersections = np.unique(
                    segm_data[roi_mask], return_counts=True
                )
                intersections_argmax = intersections.argmax()
                intersect_ID = intersect_IDs[intersections_argmax]
                intersection = intersections[intersections_argmax]
                
                union = np.count_nonzero(
                    np.logical_or(roi_mask, segm_data==intersect_ID)
                )
                IoU = intersection/union
                # if (
                #     replicate == '2020-02-20_SCGE_Diploid_2'
                #     and exp_folder == '2020-02-20_ASY15-1_0nM_SCGE'
                #     and pos == 'Position_3'
                # ):
                #     printl(roi.name, IoU, intersect_ID)
                #     fig, ax = plt.subplots(1,2)
                #     ax[0].plot(contour[:,0], contour[:,1], color='r')
                #     ax[0].imshow(segm_data)
                #     ax[1].imshow(roi_mask)
                #     plt.show()
                #     import pdb; pdb.set_trace()

                # if (
                #     replicate == '2020-02-18_Diploid_SCD_1'
                #     and exp_folder == '2020-02-18_ASY15-1_150nM'
                #     and pos == 'Position_48'
                # ):
                #     import pdb; pdb.set_trace()

                spotmax_df_idx_to_keep.append(
                    (replicate, exp_folder, pos, intersect_ID)
                )

                spotmax_df_additional_cols['RoiSet_filename'].append(f'{exp_folder}.zip')
                spotmax_df_additional_cols['roi_name'].append(roi.name)
                spotmax_df_additional_cols['roi_intersection_over_union'].append(IoU)
                
                group_name = f'{medium};{ploidy};{replicate};{exp_folder};{pos}'
                all_segm_data[group_name] = segm_data
            
    spotmax_mitograph_final_df = spotmax_final_df.filter(
        spotmax_df_idx_to_keep, axis=0
    )
    spotmax_mitograph_final_df.loc[spotmax_df_idx_to_keep, 'RoiSet_filename'] = (
        spotmax_df_additional_cols['RoiSet_filename']
    )
    spotmax_mitograph_final_df.loc[spotmax_df_idx_to_keep, 'roi_name'] = (
        spotmax_df_additional_cols['roi_name']
    )
    spotmax_mitograph_final_df.loc[spotmax_df_idx_to_keep, 'roi_intersection_over_union'] = (
        spotmax_df_additional_cols['roi_intersection_over_union']
    )
    if SAVE:
        filtered_df_filename = f'{spotmax_df_filename}_filtered_rois.csv'
        filtered_df_filepath = os.path.join(
            filtered_tables_path, filtered_df_filename
        )
        spotmax_mitograph_final_df.to_csv(filtered_df_filepath)

                