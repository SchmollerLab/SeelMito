import os
import re
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import utils
from utils import printl

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

data_path = r'G:\My Drive\1_MIA_Data\Anika\WTs'
mitograph_folder = r'G:\My Drive\1_MIA_Data\Anika\MitoGraph'
spotmax_filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')

json_paths_filenames = [
    'SCD_Diploid_paths.json', 'SCD_Haploid_paths.json',
    'SCGE_Diploid_paths.json', 'SCGE_Haploid_paths.json'
]
for json_paths_filename in tqdm(json_paths_filenames, ncols=100, desc='json_file'):
    with open(os.path.join(pwd_path, 'json_paths', json_paths_filename)) as json_file:
        exp_paths = json.load(json_file)
    medium = exp_paths['medium']
    ploidy = exp_paths['ploidy']
    replicates = exp_paths['replicates']
    run_number = exp_paths['run_number']
    spotmax_df_filename = f'spotMAX_Anika_WT_{medium}_{ploidy[:-1]}_TOT_data_filtered_rois'
    spotmax_final_df_path = os.path.join(spotmax_filtered_tables_path, f'{spotmax_df_filename}.csv')
    # `replicate_num` column == `replicate`; `horm_conc` == `exp_folder`
    spotmax_final_df = pd.read_csv(spotmax_final_df_path).set_index(['replicate_num', 'horm_conc', 'Position_n'])

    # Initialize mitograph columns
    spotmax_final_df['mitograph_volume_from_voxels'] = np.nan
    spotmax_final_df['mitograph_volume_from_length_um3'] = np.nan
    spotmax_final_df['mitograph_total_length_um'] = np.nan
    spotmax_final_df['mitograph_foldername'] = ''
    spotmax_final_df['mitograph_filename'] = ''
    for replicate, exp_folders in tqdm(replicates.items(), ncols=100, position=1, leave=False, desc='Replicates'):
        for exp_folder in tqdm(exp_folders, ncols=100, position=2, leave=False, desc='Experiment'):
            exp_horm_conc = re.findall(r'_(\d+nM)', exp_folder)[0]
            replicate_mitog_folderpath = os.path.join(mitograph_folder, medium, ploidy, replicate)
            mitog_folders = utils.listdir(replicate_mitog_folderpath)
            for mitog_folder in mitog_folders:
                if mitog_folder.find(exp_horm_conc) != -1:
                    break
            else:
                import pdb; pdb.set_trace()
            
            
            exp_path = os.path.join(
                data_path, medium, ploidy, replicate, exp_folder, 'TIFFs'
            )
            pos_foldernames = utils.get_pos_foldernames(exp_path)

            group_name = os.path.join(medium, ploidy, replicate, exp_folder)

            # Determine pos number from .mitograph file
            mitog_exp_folderpath = os.path.join(replicate_mitog_folderpath, mitog_folder)
            for mitog_file in tqdm(utils.listdir(mitog_exp_folderpath), ncols=100, leave=False, position=3, desc='Positions'):
                if not mitog_file.endswith('.mitograph'):
                    continue
                
                try:
                    mitog_pos_num = int(re.findall(r'(\d+).mitograph', mitog_file)[0])+1
                except Exception as e:
                    # Some mitograph files are duplicates, like 
                    # - C2-ASY13-1_30nM-45_044.mitograph
                    # - C2-ASY13-1_30nM-45_044 (1).mitograph
                    # in the folder "MitoGraph\\SCGE\\Haploids\\2020-09-24_SCGE_Haploid_3\\200924_ASY13-1_30nM_SCGE_cells"
                    # By ignoring the error we skip these files
                    continue
                
                # Determine pos number from mNeon.tif file
                spotmax_pos = None
                for pos in pos_foldernames:
                    images_path = os.path.join(exp_path, pos, 'Images')
                    for file in utils.listdir(images_path):
                        if file.endswith('_mNeon.tif'):
                            m = re.findall(r'(\d+)_s(\d+)_', file)
                            if m:
                                czi_pos_num = int(m[0][0])
                            else:
                                czi_pos_num = 1
                            if czi_pos_num == mitog_pos_num:
                                spotmax_pos = pos
                                mNeon_filename = file
                            break
                    else:
                        raise FileNotFoundError(
                            f'This mitograph file does not have a matching mitograph position number {mitog_pos_num}: "{mitog_file}"\n'
                            f'Parent path: "{mitog_exp_folderpath}"'
                        )
                    if spotmax_pos is not None:
                        break
                else:
                    # Mitograph position missing in spotmax data 
                    # (likely because removed by myself, probably it was a 
                    # duplicate position)
                    continue
                
                mitog_filepath = os.path.join(mitog_exp_folderpath, mitog_file)
                mitog_data = pd.read_csv(mitog_filepath, sep=r'\t', engine='python').iloc[0]

                # `replicate_num` column == `replicate`; `horm_conc` == `exp_folder`
                try:
                    index = (replicate, exp_folder, spotmax_pos)
                    spotmax_final_df.at[index, 'mitograph_volume_from_voxels'] = mitog_data['Volume from voxels']
                    spotmax_final_df.at[index, 'mitograph_volume_from_length_um3'] = mitog_data['Volume from length (um3)']
                    spotmax_final_df.at[index, 'mitograph_total_length_um'] = mitog_data['Total length (um)']
                    spotmax_final_df.at[index, 'mitograph_foldername'] = mitog_folder
                    spotmax_final_df.at[index, 'mitograph_filename'] = mitog_file
                except Exception as e:
                    import pdb; pdb.set_trace()

    if SAVE:
        spotmax_final_df.to_csv(spotmax_final_df_path)           
                

            

    


