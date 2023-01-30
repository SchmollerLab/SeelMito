import os
import sys
import shutil
import multiprocessing
import argparse
import re
import tempfile
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.measure
from skimage.draw import disk, ellipsoid
from scipy import stats
from scipy.special import erf
from scipy.optimize import least_squares
from time import time
from natsort import natsorted
from numba import jit, njit, prange
# from tqdm.notebook import tqdm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import prompts, apps, load, core

# Parse arguments for command line
ap = argparse.ArgumentParser(description='spotMAX inputs')
ap.add_argument('-v', '--verbose', type=int, default=0,
                help='Verbosity level. Choose between 0, 1, 2, 3')
ap.add_argument('-i', '--inspect', type=int, default=0,
                help='Visualize intermediate results. '
                     'Available: 0 (no inspection), 1 (inspect global results)'
                     '2 (inspect at single segemnted object level)')

args = vars(ap.parse_args())

verbose = args['verbose']
inspect = args['inspect']

# Initial setup
bp = apps.tk_breakpoint()
num_cpu = multiprocessing.cpu_count()

# Script version (to append for file saving)
script_name = os.path.basename(__file__)
NUM = re.findall('v(\d+).py', script_name)[0]
vNUM = f'v{NUM}'

# Initialize prompt widgets
scan_run_num = prompts.scan_run_nums(vNUM)
spotFIT_inputs = prompts.spotFIT_inputs()
num_frames_prompt = prompts.num_frames_toQuant()

# Folder dialog
selected_path = prompts.folder_dialog(title='Select folder with multiple experiments, '
                                     'the TIFFs folder or '
                                     'a specific Position_n folder')
(main_paths, prompts_pos_to_analyse,
run_num, tot) = load.get_main_paths(selected_path, vNUM)

print('')
print(f'Analysing all data inside {selected_path}...')
pbar = tqdm(total=tot, unit=' exp', leave=True, position=0)

ch_name_selector = prompts.select_channel_name(which_channel='spots')
ref_ch_name_selector = prompts.select_channel_name(which_channel='ref')

t0_tot = time()
"""Iterate experiment folders"""
for exp_idx, main_path in enumerate(main_paths):
    t0_exp = time()

    dirname = os.path.basename(main_path)

    if dirname == 'TIFFs':
        ls_main_path = natsorted(os.listdir(main_path))
        pos_paths = [os.path.join(main_path, p) for p in ls_main_path
                               if os.path.isdir(f'{main_path}/{p}')
                               and p.find('Position_')!=-1]

        ps, pe = 0, len(pos_paths)
        if exp_idx == 0:
            if prompts_pos_to_analyse:
                ps, pe = core.num_pos_toQuant_tk(len(pos_paths)).frange
    elif is_pos_path:
        pos_paths = [main_path]
        ps, pe = 0, len(pos_paths)
    else:
        raise FileNotFoundError(f'The path {main_path} is not a valid path!')


    """Iterate Position folders"""
    num_pos_total = len(pos_paths[ps:pe])
    for pos_idx, pos_path in enumerate(pos_paths[ps:pe]):
        t0_pos = time()

        # Check that the '3_p-_ellip_test_data' exists, if not skip
        skip = load.spotfit_checkpoint(pos_path)
        if skip:
            pbar.update()
            continue


        pos_foldername = os.path.basename(pos_path)
        desc = (f'Analysing experiment {exp_idx+1}/{len(main_paths)}, '
                f'{pos_foldername} ({pos_idx+1}/{num_pos_total})')
        pbar.set_description(desc)

        t0 = time()

        #Load tif file images and metadata
        # print('Loading data...')
        spots_ch_data = load.load_data(pos_path,
                               load_shifts=True,
                               load_segm_npy=True,
                               load_cca_df=True, create_data_folder=False,
                               ch_name_selector=ch_name_selector,
                               load_analysis_inputs=True,
                               load_df_h5=True, run_num=run_num,
                               which_h5='3_p-_ellip_test_data',
                               load_summary_df=True)

        data_path = spots_ch_data.data_path
        channel_name = spots_ch_data.channel_name
        cca_df = spots_ch_data.cca_df
        is_segm_3D = spots_ch_data.is_segm_3D
        df_spots_store = spots_ch_data.store_HDF



        # Check shape of data
        error = prompts.check_img_shape_vs_metadata(
                                            spots_ch_data.frames.shape,
                                            spots_ch_data.num_frames,
                                            spots_ch_data.SizeT,
                                            spots_ch_data.SizeZ)
        if error:
            spots_ch_data.store_HDF.close()
            raise IndexError(f'Shape mismatch. {error}')

        #Align frames according to saved shifts (calculated from the phase contrast)
        if not spots_ch_data.already_aligned:
            # print('Aligning frames...')
            spots_ch_aligned = core.align_frames(spots_ch_data.frames,
                                         path=spots_ch_data.path,
                                         save_aligned=True, register=False,
                                         saved_shifts=spots_ch_data.shifts)

        if spotFIT_inputs.is_first_call:
            spotFIT_inputs.prompt(spots_ch_data.df_inputs, is_segm_3D)
            zyx_vox_size = spotFIT_inputs.zyx_vox_size
            segm_info = spotFIT_inputs.segm_info
            zyx_spot_min_vol_um = spotFIT_inputs.zyx_spot_min_vol_um
            filter_by_ref_ch = spotFIT_inputs.filter_by_ref_ch
            do_save = spotFIT_inputs.do_save

        if filter_by_ref_ch:
            ref_ch_data = load.load_data(
                           pos_path,
                           load_shifts=True,
                           load_segm_npy=True,
                           load_cca_df=True, create_data_folder=False,
                           ch_name_selector=ref_ch_name_selector,
                           load_analysis_inputs=True,
                           load_df_h5=False, run_num=run_num,
                           which_ch='Reference',
                           load_ch_mask=True)

        # print('Data successfully loaded.')
        # print('')

        if do_save:
            temp_dirpath = tempfile.mkdtemp()
            spotFIT_filename = '4_spotFIT_data.h5'
            HDF_temp_path = os.path.join(temp_dirpath, spotFIT_filename)
            spotFIT_store_HDF = pd.HDFStore(HDF_temp_path,  mode='w',
                                            complevel=5, complib = 'zlib')

        spotMAX_data = core.spotMAX()
        spotMAX_data.data_path = data_path
        spotMAX_data.vNUM = vNUM
        spotMAX_data.run_num = run_num
        spotMAX_data.do_save = do_save
        # spotFIT_filename used for saving that gaussian fit was done
        if do_save:
            spotMAX_data.filename = spotFIT_filename

        """Iterate frames"""
        if num_frames_prompt.is_first_call and spots_ch_data.num_segm_frames > 1:
            num_frames = spots_ch_data.num_frames
            num_segm_frames = spots_ch_data.num_segm_frames
            num_frames_prompt.prompt(num_frames, last_segm_i=num_segm_frames-1)
            frange = num_frames_prompt.frange
        else:
            frange = 0, spots_ch_data.num_segm_frames

        summary_dfs = []
        frame_i_li = []
        for frame_i in range(*frange):
            if spots_ch_data.num_segm_frames > 1:
                t0_frame = time()
                desc = (f'Analysing experiment {exp_idx+1}/{len(main_paths)}, '
                        f'{pos_foldername} ({pos_idx+1}/{num_pos_total}), '
                        f'frame {frame_i+1}/{frange[1]}')
                pbar.set_description(desc)
            if spots_ch_data.SizeT > 1:
                V_spots = spots_ch_data.frames[frame_i]
                segm_npy = spots_ch_data.segm_npy[frame_i]
                if filter_by_ref_ch:
                    V_ref_mask = ref_ch_data.ch_mask[frame_i]
                else:
                    V_ref_mask = None
            else:
                V_spots = spots_ch_data.frames
                segm_npy = spots_ch_data.segm_npy
                if filter_by_ref_ch:
                    V_ref_mask = ref_ch_data.ch_mask
                else:
                    V_ref_mask = None
            df_h5 = spots_ch_data.store_HDF[f'frame_{frame_i}']

            if segm_npy is None:
                raise FileNotFoundError('Segmentation file not found. '
                'spotFIT without a segmentation file is not implemented yet.')

            if segm_info == '2D':
                if segm_npy.ndim == 2:
                    segm_npy_3D = np.array([segm_npy]*len(V_spots))
                else:
                    raise IndexError('The loaded segmentation is not a '
                    '2D mask as selected. '
                    f'The shape is {segm_npy_frame.shape}')
            else:
                if segm_npy.ndim == 3:
                    segm_npy_3D = segm_npy.copy()
                else:
                    raise IndexError('The loaded segmentation is not a '
                    '3D mask as selected. '
                    f'The shape is {segm_npy.shape}')

            rp_segm_3D = skimage.measure.regionprops(segm_npy_3D)
            V_spots = skimage.img_as_float(V_spots)
            df_spots_h5 = df_spots_store.select(f'frame_{frame_i}')
            summ_df_frame_i = spots_ch_data.summary_df.loc[frame_i].copy()

            if verbose > 1:
                print(df_spots_h5.index.names)
                print(df_spots_h5.columns)

            """Iterate cells"""
            spotFIT_dfs = []
            keys = []
            IDs = []
            pbar_1 = tqdm(total=len(rp_segm_3D), leave=False, unit=' cell',
                          position=1, desc='Iterating cells')
            for obj_3D in rp_segm_3D:
                ID = obj_3D.label
                IDs.append(ID)
                if ID not in df_spots_h5.index.get_level_values(0):
                    continue
                V_spots_ID = V_spots[obj_3D.slice].copy()
                df_spots_h5_ID = df_spots_h5.loc[ID]
                min_z, min_y, min_x, _, _, _ = obj_3D.bbox
                ID_bbox_lower = (min_z, min_y, min_x)
                mask_ID = obj_3D.image
                if V_ref_mask is not None:
                    V_ref_mask_ID = np.logical_and(mask_ID,
                                                   V_ref_mask[obj_3D.slice])
                else:
                    V_ref_mask_ID = None
                spotFIT_data = core.spotFIT(
                                   V_spots_ID, df_spots_h5_ID, zyx_vox_size,
                                   zyx_spot_min_vol_um, ID_bbox_lower, mask_ID,
                                   V_ref_mask_ID, verbose=verbose,
                                   inspect=inspect)
                spotFIT_data.fit()
                spotFIT_dfs.append(spotFIT_data.df_spotFIT_ID)
                keys.append(ID)
                pbar_1.update()

                """End of for loop iterating cells"""

            pbar_1.close()

            df_spotFIT = pd.concat(spotFIT_dfs, keys=keys,
                                   names=['Cell_ID', 'spot_id'])
            spotMAX_data.agg_spotFIT(df_spotFIT, IDs)
            summ_df_frame_i = spotMAX_data.add_agg_spotFIT(summ_df_frame_i)
            frame_i_li.append(frame_i)
            summary_dfs.append(summ_df_frame_i)

            if do_save:
                key = f'frame_{frame_i}'
                spotFIT_store_HDF.append(key, df_spotFIT)

            """End of for loop iterating frames"""

        summary_df = pd.concat(summary_dfs, keys=frame_i_li,
                               names=['frame_i', 'Cell_ID'])
        # Close hdf stores, move spotfit data from temp to data folder
        # and delete temp folder
        spots_ch_data.store_HDF.close()
        if do_save:
            summary_df.to_csv(spots_ch_data.summ_df_path)
            spotMAX_data.save_spotFIT_done()
            spotFIT_store_HDF.close()
            dst = os.path.join(spots_ch_data.data_path, spotFIT_filename)
            shutil.move(HDF_temp_path, dst)
            temp_dir = os.path.dirname(HDF_temp_path)
            shutil.rmtree(temp_dir)


        """End of for loop iterating positions"""

    tend_exp = time()

    # print(f'Analysis of experiment {TIFFs_path} done!\n'
    #       f'Execution time: {tend_exp-t0_exp:.3f} s')
    # print('##################################################')
    # print('')

    """End of for loop iterating experiment folders"""

    pbar.update()

tend_tot = time()

pbar.close()

print('All experiments have been analysed!')
print('---------------')
exec_time = tend_tot-t0_tot
exec_time_min = exec_time/60
exec_time_delta = datetime.timedelta(seconds=exec_time)
print(f'Total execution time: {exec_time:.2f} s '
      f'({exec_time_delta} HH:mm:ss)')
exec_time_per_exp = exec_time/len(main_paths)
exec_time_per_exp_delta = datetime.timedelta(seconds=exec_time/len(main_paths))
print(f'Average execution time per experiment {exec_time_per_exp:.2f} s '
      f'({exec_time_per_exp_delta} HH:mm:ss)')
exec_time_per_pos = exec_time_per_exp/(pe-ps)
exec_time_per_pos_delta = datetime.timedelta(seconds=exec_time_per_exp/(pe-ps))
print(f'Average execution time per position {exec_time_per_pos:.2f} s '
      f'({exec_time_per_pos_delta} HH:mm:ss)')
print('---------------')
