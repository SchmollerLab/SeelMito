import os
import shutil
import subprocess
import traceback
import argparse
import tkinter as tk
from time import time
import datetime
import pandas as pd
import numpy as np
import cv2, re
import matplotlib.pyplot as plt
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
from tqdm import tqdm
from natsort import natsorted
try:
    from pyglet.canvas import Display
except:
    pass
from scipy import stats
from tkinter import filedialog as fd
from tkinter import Tk
from skimage import io, img_as_float
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import (
      threshold_otsu, threshold_local, threshold_multiotsu, gaussian,
      apply_hysteresis_threshold, threshold_minimum,  threshold_yen,
      threshold_li, threshold_isodata, threshold_triangle, unsharp_mask)
from skimage.filters import meijering, sato, frangi, hessian
from skimage.color import label2rgb
from skimage.exposure import histogram, equalize_adapthist
from skimage.measure import label, regionprops

import matplotlib

matplotlib.use('Agg')

from load import load_data, load_positive_control
import core, prompts, apps, load

"""NOTE: z_resolution_limit:
                     Resolution limit in μm. With confocal this is about
                     2-3 times larger than yx_resolution limit.
                     A good estimate is given by ZEN software in the
                     Z-stack menu by pressing 'Optimal'. Write None if
                     you don't need a different z resolution limit
"""

# Parse arguments for command line
ap = argparse.ArgumentParser(description='spotMAX inputs')
ap.add_argument('-v', '--verbose', type=int, default=1,
                help='Verbosity level. Choose between 0, 1, 2, 3')
ap.add_argument('-i', '--inspect', type=int, default=0,
                help='Visualize intermediate results. '
                     'Available: 0 (no inspection), 1 (inspect global results)'
                     '2 (inspect at single segemnted object level)')
ap.add_argument('-t', '--testing', type=int, default=0,
                help='Used for analysing a smaller dataset when testing '
                     'parameters. Type 0 or 1 for testing mode OFF or ON')
ap.add_argument('-e', '--experimenting', type=int, default=0,
                help='Used for experimenting with new features without changing'
                ' default tested behaviour of the script. Type 0 or 1 for '
                'experimental mode OFF or ON')
ap.add_argument('-d', '--debugging', type=int, default=0,
                help='Used for debugging. Test code with "if debug: ..."')

slice_by_slice = True

args = vars(ap.parse_args())

# Inititalize breakpoint (needed for debugging)
bp = apps.tk_breakpoint()

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 12)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# plt initial settings
plt.style.use('dark_background')
plt.rc('axes', edgecolor='0.1')

# Script version (to append for file saving)
script_name = os.path.basename(__file__)
script_dirpath = os.path.dirname(os.path.realpath(__file__))
NUM = re.findall('v(\d+).py', script_name)[0]
vNUM = f'v{NUM}'

# Initial variables
load_shifts = True
load_segm_npy = True
verbose = args['verbose']
inspect = args['inspect'] > 0
inspect_deep = args['inspect'] == 2
testing = args['testing'] == 1
experimenting = args['experimenting'] == 1
debug = args['debugging'] == 1

# Determine if user is running scripting mode without inspection
# to set Save Yes by default
areArgsDefault = (
    args['inspect'] == 0 and args['testing'] == 0 and
    args['experimenting'] == 0 and args['debugging'] == 0
)


# Prompt the user to select the image file
# Folder dialog
selected_path = prompts.folder_dialog(title=
    'Select folder with multiple experiments, the TIFFs folder or '
    'a specific Position_n folder')

if not selected_path:
    exit('Execution aborted.')

(main_paths, prompts_pos_to_analyse, run_num, tot,
is_pos_path, is_TIFFs_path) = load.get_main_paths(selected_path, vNUM)

# Channel name. The load_data selector is used by load_data class to
# prompt to select a channel name from available channel names in the metadata
# if the channel name selected by the user is not present.
ch_name_selector = prompts.select_channel_name(which_channel='spots')
load_data_ch0_selector = prompts.select_channel_name()

ref_ch_name_selector = prompts.select_channel_name(which_channel='ref')
load_data_ref_ch_selector = prompts.select_channel_name()

IDs_subset = prompts.analyse_subset_IDswidget()

# Class used to determine if positive control images are present
load_PC = load_positive_control()
is_first_call = True
load_cca_df = True

t0_tot = time()
"""Iterate experiment folders"""
for exp_idx, main_path in enumerate(main_paths):
    t0_exp = time()

    dirname = os.path.basename(main_path)

    if dirname == 'TIFFs':
        TIFFs_path = main_path
        print('')
        print('##################################################')
        print('')
        print(f'Analysing experiment: {os.path.dirname(main_path)}')
        folders_main_path = [f for f in natsorted(os.listdir(main_path))
                               if os.path.isdir(f'{main_path}/{f}')
                               and f.find('Position_')!=-1]

        paths = []
        for i, d in enumerate(folders_main_path):
            images_path = f'{main_path}/{d}/Images'
            if not os.path.exists(images_path):
                print('')
                print('==========================================')
                print('')
                print(f'WARNING: Folder {images_path} does not '
                'exist. Skipping it.')
                print('')
                print('==========================================')
                print('')
                continue
            filenames = os.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names = ch_name_selector.get_available_channels(filenames)
                ch_name_selector.prompt(ch_names)
            channel_name = ch_name_selector.channel_name
            user_ch_name = ch_name_selector.channel_name
            spots_ch_aligned_found = False
            spots_ch_tif_found = False
            for j, f in enumerate(filenames):
                if f.find(f'{user_ch_name}_aligned.npy') != -1:
                    spots_ch_aligned_found = True
                    aligned_i = j
                elif f.find(f'{user_ch_name}.tif') != -1:
                    spots_ch_tif_found = True
                    tif_i = j
            if spots_ch_aligned_found:
                spots_ch_path = os.path.join(images_path, filenames[aligned_i])
            elif spots_ch_tif_found:
                spots_ch_path = os.path.join(images_path, filenames[tif_i])
            else:
                print('')
                print('==========================================')
                print('')
                print(f'WARNING: Folder {images_path} does not '
                f'contain requested file "_{user_ch_name}.tif". Skipping it.')
                print('')
                print('==========================================')
                print('')
                continue
            paths.append(spots_ch_path)

        ps, pe = 0, len(paths)
        if exp_idx == 0:
            spotMAX_inputs = prompts.spotMAX_inputs_widget(areArgsDefault)
            if prompts_pos_to_analyse:
                ps, pe = core.num_pos_toQuant_tk(len(paths)).frange

    elif is_pos_path:
        TIFFs_path = os.path.dirname(main_path)
        pos_path = main_path
        images_path = f'{pos_path}/Images'
        if not os.path.exists(images_path):
            print('')
            print('==========================================')
            print('')
            print(f'WARNING: Folder {images_path} does not '
            'exist. Skipping it.')
            print('')
            print('==========================================')
            print('')
            continue
        filenames = os.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names = ch_name_selector.get_available_channels(filenames)
            ch_name_selector.prompt(ch_names)
            channel_name = ch_name_selector.channel_name
            user_ch_name = ch_name_selector.channel_name
        spots_ch_aligned_found = False
        spots_ch_tif_found = False
        for j, f in enumerate(filenames):
            if f.find(f'_{user_ch_name}_aligned.npy') != -1:
                spots_ch_aligned_found = True
                aligned_i = j
            elif f.find(f'_{user_ch_name}.tif') != -1:
                spots_ch_tif_found = True
                tif_i = j
        if spots_ch_aligned_found:
            spots_ch_path = os.path.join(images_path, filenames[aligned_i])
        elif spots_ch_tif_found:
            spots_ch_path = os.path.join(images_path, filenames[tif_i])
        else:
            print('')
            print('==========================================')
            print('')
            print(f'WARNING: Folder {images_path} does not '
            f'contain requested file "_{user_ch_name}.tif". Skipping it.')
            print('')
            print('==========================================')
            print('')
            continue
        paths = [spots_ch_path]
        ps, pe = 0, len(paths)
        if exp_idx == 0:
            spotMAX_inputs = prompts.spotMAX_inputs_widget(areArgsDefault)
    else:
        raise FileNotFoundError(f'The path {main_path} is not a valid path!')

    user_ch_name = channel_name


    """Iterate Position folders"""
    num_pos_total = len(paths[ps:pe])
    for pos_idx, path in enumerate(paths[ps:pe]):
        t0_pos = time()
        pos_path = os.path.dirname(os.path.dirname(path))
        pos_foldername = os.path.basename(pos_path)

        t0 = time()

        #Load tif file images and metadata
        print('Loading data...')
        spots_ch_data = load_data(path,
                               channel_name=channel_name,
                               user_ch_name=user_ch_name,
                               load_shifts=load_shifts,
                               load_segm_npy=load_segm_npy,
                               load_cca_df=load_cca_df,
                               create_data_folder=False,
                               ch_name_selector=load_data_ch0_selector,
                               ask_metadata_not_found=False)

        if spots_ch_data.frames is None:
            print('')
            print('==========================================')
            print('')
            continue

        elif spots_ch_data.segm_npy is None:
            print('')
            print('==========================================')
            print('')
            print(f'WARNING: Position {pos_foldername} does not have '
            'a segmmentation file. Skipping it.')
            print('')
            print('==========================================')
            print('')
            continue
        elif not spots_ch_data.segm_npy.any():
            print('')
            print('==========================================')
            print('')
            print(f'WARNING: Segmentation mask at position {pos_foldername} '
            'is empty. Skipping it.')
            print('')
            print('==========================================')
            print('')
            continue

        data_path = spots_ch_data.data_path
        channel_name = spots_ch_data.channel_name
        if spots_ch_data.cca_df is not None:
            if is_first_call:
                root = tk.Tk()
                root.withdraw()
                use_cca_df = tk.messagebox.askquestion(
                    'Cell cycle annotations detected',
                    'The system detected cell cycle annotations '
                    '(e.g. mother-daughter information).\n\n'
                    'Do you want to use this information?\n'
                    'NOTE: reccomended only for yeast cells.',
                    master=root)
                root.quit()
                root.destroy()
            if use_cca_df == 'yes':
                cca_df = spots_ch_data.cca_df
                load_cca_df = True
            else:
                cca_df = None
                load_cca_df = False
        else:
            cca_df = None

        is_segm_3D = spots_ch_data.is_segm_3D

        print('Data successfully loaded.')
        print('')

        # # Check shape of data
        # error = prompts.check_img_shape_vs_metadata(
        #                                     spots_ch_data.frames.shape,
        #                                     spots_ch_data.num_frames,
        #                                     spots_ch_data.SizeT,
        #                                     spots_ch_data.SizeZ)
        # if error:
        #     spots_ch_data.store_HDF.close()
        #     raise IndexError(f'Shape mismatch. {error}')

        # Show spotMAX input variables to the user
        if spotMAX_inputs.show:
            if spots_ch_data.zyx_vox_dim is not None:
                vox_dim = [round(r, 6) for r in  spots_ch_data.zyx_vox_dim]
            else:
                vox_dim = None
            spotMAX_inputs.run(
                      title='spotMAX analysis inputs',
                      channel_name=channel_name,
                      zyx_voxel_size=f'{vox_dim}',
                      zyx_voxel_size_float=spots_ch_data.zyx_vox_dim,
                      z_resolution_limit='1',
                      numerical_aperture=f'{spots_ch_data.NA}',
                      em_wavelength=f'{spots_ch_data.wavelen}',
                      gauss_sigma='0.75', is_segm_3D=is_segm_3D
            )

            local_maxima_thresh_func = globals()[spotMAX_inputs
                                                 .local_max_thresh_func]
            ref_ch_thresh_func = globals()[spotMAX_inputs
                                          .ref_ch_thresh_func]
            zyx_vox_dim = literal_eval(spotMAX_inputs.zyx_vox_size)
            z_resolution_limit = float(spotMAX_inputs.z_resol_limit)
            NA = float(spotMAX_inputs.NA)
            wavelen = float(spotMAX_inputs.em_wavel)
            sigma = float(spotMAX_inputs.gauss_sigma)
            yx_resolution_multiplier = float(spotMAX_inputs
                                             .yx_resolution_multiplier)

            gop_how = spotMAX_inputs.gop_how
            which_effsize = spotMAX_inputs.which_effsize
            gop_thresh_val = [float(spotMAX_inputs.gop_limit_txt)]
            do_bootstrap = gop_how.find('bootstrap') != -1

            calc_ref_ch_len = spotMAX_inputs.calc_ref_ch_len
            do_spotSIZE = spotMAX_inputs.do_spotSIZE
            do_gaussian_fit = spotMAX_inputs.do_gaussian_fit

            load_ref_ch = spotMAX_inputs.load_ref_ch
            is_ref_single_obj = spotMAX_inputs.is_ref_single_obj
            filter_by_ref_ch = spotMAX_inputs.filter_by_ref_ch

            filter_z_bound = spotMAX_inputs.filter_z_bound

            make_spots_sharper = spotMAX_inputs.make_sharper

            local_or_global_thresh = spotMAX_inputs.local_or_global_thresh

            segm_info = spotMAX_inputs.segm_info

            is_segm_3D = segm_info == '3D'

            spotsize_limits_pxl = spotMAX_inputs.spotsize_limits_pxl

            do_save = spotMAX_inputs.save

        # Prompt to select reference channel name
        if load_ref_ch and ref_ch_name_selector.is_first_call:
            images_path = os.path.dirname(path)
            filenames = os.listdir(images_path)
            ch_names = ref_ch_name_selector.get_available_channels(filenames)
            ch_names = [c for c in ch_names if c.find(user_ch_name)==-1]
            ref_ch_name_selector.prompt(ch_names,
                                        message='Select reference channel name')
            ref_channel_name = ref_ch_name_selector.channel_name
            user_ref_ch_name = ref_ch_name_selector.channel_name
        elif not load_ref_ch:
            ref_channel_name = None
            user_ref_ch_name = None

        # Align frames according to saved shifts (calculated from the phase contrast)
        # Skipped if align_shift.npy not found
        if not spots_ch_data.already_aligned:
            print('Aligning frames...')
            spots_ch_aligned = core.align_frames(spots_ch_data.frames,
                                         path=spots_ch_data.path,
                                         save_aligned=True, register=False,
                                         saved_shifts=spots_ch_data.shifts)
            if load_ref_ch:
                ref_ch_filename = f'{user_ref_ch_name}.tif'
        elif spots_ch_data.skip_alignment:
            if load_ref_ch:
                ref_ch_filename = f'{user_ref_ch_name}.tif'
        else:
            if load_ref_ch:
                ref_ch_filename = f'{user_ref_ch_name}_aligned.npy'

        if load_ref_ch:
            #Get path to mKate file
            position_n_path = os.path.dirname(path)
            found = False
            for filename in os.listdir(position_n_path):
                if filename.find(ref_ch_filename)>0:
                    found=True
                    break
            if not found:
                raise FileNotFoundError(ref_ch_filename+' Not found.')
            ref_ch_path = os.path.join(position_n_path, filename)

            #Load reference channel data and metadata
            ref_ch_data = load_data(
                               ref_ch_path, ref_channel_name, user_ref_ch_name,
                               ch_name_selector=load_data_ref_ch_selector)

            #Align frames according to saved shifts (calculated from the phase contrast)
            if not spots_ch_data.already_aligned:
                ref_ch_aligned = core.align_frames(ref_ch_data.frames,
                                             path=ref_ch_data.path,
                                             save_aligned=True, register=False,
                                             saved_shifts=spots_ch_data.shifts)
                print('Frames aligned')

        # Save analysis inputs
        if spotMAX_inputs.show:
            spotMAX_inputs.entry_labels.append('Spots channel name:')
            spotMAX_inputs.entry_txts.append(channel_name)
            spotMAX_inputs.entry_labels.append('Spots file name:')
            spotMAX_inputs.entry_txts.append(user_ch_name)
            if load_ref_ch:
                spotMAX_inputs.entry_labels.append('Reference channel name:')
                spotMAX_inputs.entry_txts.append(ref_channel_name)
                spotMAX_inputs.entry_labels.append('Reference ch. file name:')
                spotMAX_inputs.entry_txts.append(user_ref_ch_name)

        df_inputs = (pd.DataFrame({
                         'Description': spotMAX_inputs.entry_labels,
                         'Values': spotMAX_inputs.entry_txts}
                          ).set_index('Description')
                           .sort_index())

        df_inputs.at['Analysis date:', 'Values'] = datetime.datetime.now()

        print('\n**********************')
        print('Analysis inputs:')
        print(df_inputs)
        print('**********************\n')

        print('----------------------------------')
        print(f'Analysing experiment {exp_idx+1}/{len(main_paths)}, '
              f'{pos_foldername} ({pos_idx+1}/{num_pos_total})')
        print(f'{pos_path}...')
        print('')

        # Store last status of the inputs widget to load it next time
        if spotMAX_inputs.show:
            df_inputs.to_csv(os.path.join(script_dirpath,
                                          'last_status_inputs_widget.csv'))

        replace = False
        if os.path.exists(data_path) and do_save and spotMAX_inputs.show:
            if len(os.listdir(data_path)) > 5:
                df_inputs = pd.read_csv(os.path.join(script_dirpath,
                                              'last_status_inputs_widget.csv'),
                                        index_col='Description'
                                        ).astype('string')
                do_save, replace = (spotMAX_inputs
                        .check_if_same_of_prev_run(data_path,
                                                   vNUM, df_inputs))

        run_num = spotMAX_inputs.run_num

        if do_save:
            spotMAX_inputs_path = spotMAX_inputs.save_analysis_inputs(
                                                data_path, vNUM, df_inputs)

        spotMAX_inputs.show = False

        if verbose>0 and verbose<=3:
            print('')
            print(f'Data shape = {spots_ch_data.frames.shape}')
            print(f'SizeT = {spots_ch_data.SizeT}, '
                  f'SizeZ = {spots_ch_data.SizeZ}')
            print('')

        # Calculate Airy disc radius = resolution limit in yx.
        # see https://www.microscopyu.com/tutorials/imageformation-airyna
        (zyx_resolution,
        zyx_resolution_pxl,
        airy_radius) = core.calc_resolution_limited_vol(wavelen, NA,
                                                    yx_resolution_multiplier,
                                                    zyx_vox_dim,
                                                    z_resolution_limit)
        if verbose>0 and verbose<=3:
            print(f'Resolution limit (Airy disk radius) = {airy_radius:.2f} nm')
            airy_radii_print = [round(r, 3) for r in zyx_resolution_pxl]
            print(f'Airy sphere radii = {tuple(airy_radii_print)} pixels')
            print('')

        #Assign the correct data depending on alignment status (True or False)
        if spots_ch_data.already_aligned:
            spots_ch_aligned = spots_ch_data
            if load_ref_ch:
                ref_ch_aligned = ref_ch_data

        #Initialize core.spotMAX classes (4 dataframes + 3 summary dataframes)
        #subprocess.Popen('explorer "{}"'.format(os.path.normpath(data_path)))
        num_frames = spots_ch_data.num_frames
        finterval = spots_ch_data.finterval
        timestamp = spots_ch_data.timestamp

        # Initialize single nucleoids analysis dataframes --> will be saved to HDF
        orig_data = core.spotMAX()
        orig_data.init(finterval, num_frames, data_path,
                       '0_Orig_data.h5', hdf=True,
                        do_save=do_save,
                        replace=replace, vNUM=vNUM,
                        run_num=run_num)
        ellip_data = core.spotMAX()
        ellip_data.init(finterval, num_frames, data_path,
                        '1_ellip_test_data.h5', hdf=True,
                        do_save=do_save,
                        replace=replace, vNUM=vNUM,
                        run_num=run_num)
        p_data = core.spotMAX()
        p_data.init(finterval, num_frames, data_path,
                    '2_p-_test_data.h5', hdf=True,
                    do_save=do_save,
                    replace=replace, vNUM=vNUM,
                    run_num=run_num)
        p_ellip_data = core.spotMAX()
        p_ellip_data.init(finterval, num_frames, data_path,
                            '3_p-_ellip_test_data.h5', hdf=True,
                            do_save=do_save,
                            replace=replace, vNUM=vNUM,
                            run_num=run_num)

        spotQUANT_data = core.spotMAX()
        spotQUANT_data.init(finterval, num_frames,
                            data_path,
                            '4_spotFIT_data.h5',
                            hdf=do_spotSIZE, do_save=do_save,
                            replace=replace, vNUM=vNUM,
                            run_num=run_num,
                            do_spotSIZE=do_spotSIZE,
                            do_gaussian_fit=do_gaussian_fit)

        # Intialize cell analysis dataframes --> will be saved to CSV
        orig_summary = core.spotMAX()
        orig_summary.init(finterval, num_frames, data_path,
                            '0_Orig_data_Summary.csv',
                            do_save=do_save,
                            replace=replace, vNUM=vNUM,
                            run_num=run_num,
                            do_ref_chQUANT=True,
                            ref_ch_loaded=load_ref_ch,
                            calc_ref_ch_len=calc_ref_ch_len)
        ellip_summary = core.spotMAX()
        ellip_summary.init(finterval, num_frames, data_path,
                            '1_ellip_test_data_Summary.csv',
                            do_save=do_save,
                            replace=replace, vNUM=vNUM,
                            run_num=run_num,
                            ref_ch_loaded=load_ref_ch)
        p_summary = core.spotMAX()
        p_summary.init(finterval, num_frames, data_path,
                                        '2_p-_test_data_Summary.csv',
                                        do_save=do_save,
                                        replace=replace, vNUM=vNUM,
                                        run_num=run_num,
                                        ref_ch_loaded=load_ref_ch)
        p_ellip_summary = core.spotMAX()
        p_ellip_summary.init(finterval, num_frames, data_path,
                            '3_p-_ellip_test_data_Summary.csv',
                            do_save=do_save,
                            replace=replace, vNUM=vNUM,
                            run_num=run_num,
                            ref_ch_loaded=load_ref_ch)

        if do_gaussian_fit:
            spotfit_summary = core.spotMAX()
            spotfit_summary.init(finterval, num_frames, data_path,
                                '4_spotfit_data_Summary.csv',
                                do_save=do_save,
                                replace=replace, vNUM=vNUM,
                                run_num=run_num,
                                ref_ch_loaded=load_ref_ch)

        """Start for loop iterating frames"""
        if load_ref_ch:
            ref_ch_thresh_frames = np.zeros(ref_ch_aligned.frames.shape, bool)
        spots_lab_frames_dict = {}
        spots_ch_processed_frames = np.zeros(spots_ch_aligned.frames.shape)
        num_segm_frames = spots_ch_data.num_segm_frames
        if load_ref_ch:
            ref_ch_processed_frames = np.zeros(ref_ch_aligned.frames.shape)
        if 'frames_toQuant' in globals():
            # Number of frames was already prompted. Ask again only if the User
            # didn't click on "Ok for all"
            if not frames_toQuant.ok_for_all and num_segm_frames > 1:
                frames_toQuant = core.num_frames_toQuant_tk(num_segm_frames)
                frange = frames_toQuant.frange
        elif prompts_pos_to_analyse and num_segm_frames > 1:
            # First call for core.num_frames_toQuant_tk
            frames_toQuant = core.num_frames_toQuant_tk(num_segm_frames)
            frange = frames_toQuant.frange
        else:
            frange = 0, num_segm_frames


        """Iterate frames"""
        for frame_i in range(*frange):
            t0_frame = time()
            segm_npy_frame = None
            if num_segm_frames > 1:
                print('Analysing frame {}/{}...'.format(frame_i+1, frange[1]))
            if spots_ch_aligned.SizeT > 1:
                V_spots = spots_ch_aligned.frames[frame_i]
                if load_ref_ch:
                    V_ref = ref_ch_aligned.frames[frame_i]
                else:
                    V_ref = np.zeros_like(V_spots)
                if load_segm_npy and spots_ch_data.segm_npy is not None:
                    segm_npy_frame = spots_ch_data.segm_npy[frame_i]
            else:
                V_spots = spots_ch_aligned.frames
                if load_ref_ch:
                    V_ref = ref_ch_aligned.frames
                else:
                    V_ref = np.zeros_like(V_spots)
                if load_segm_npy and spots_ch_data.segm_npy is not None:
                    segm_npy_frame = spots_ch_data.segm_npy

            # Initialize segm 3D and cca_df
            shape_3D = (V_spots.shape[0], 1, 1)
            if segm_npy_frame is not None:
                if segm_info == '2D':
                    if segm_npy_frame.ndim == 2:
                        segm_npy_3D = np.tile(segm_npy_frame, shape_3D)
                    else:
                        raise IndexError('The loaded segmentation is not a '
                        '2D mask as selected. '
                        f'The shape is {segm_npy_frame.shape}')
                else:
                    if segm_npy_frame.ndim == 3:
                        segm_npy_3D = segm_npy_frame.copy()
                    else:
                        raise IndexError('The loaded segmentation is not a '
                        '3D mask as selected. '
                        f'The shape is {segm_npy_frame.shape}')

                    if testing:
                        rp_segm_3D = regionprops(segm_npy_3D)
                        IDs = [obj.label for obj in rp_segm_3D]
                        if IDs_subset.testing_mode_ON:
                            if load_ref_ch:
                                intensity_imgs = [V_spots, V_ref]
                            else:
                                intensity_imgs = [V_spots]
                            IDs_subset.run(segm_npy_3D, rp_segm_3D,
                                           intensity_imgs=intensity_imgs)
                            selection_mode = IDs_subset.selection_mode
                            if not IDs_subset.testing_mode_ON:
                                ids = []
                            elif selection_mode=='manual selection':
                                ids = IDs_subset.selection_entry_txt.split(',')
                                ids = [int(id) for id in ids]
                            elif selection_mode=='manual selection':
                                size = int(IDs_subset.selection_entry_txt)
                                ids = np.random.choice(IDs, size, replace=False)
                            print(f'Analysing only labels {ids}...')
                            for ID in IDs:
                                if ID not in ids:
                                    segm_npy_3D[segm_npy_3D==ID] = 0

                rp_segm_3D = regionprops(segm_npy_3D)
                IDs = [obj.label for obj in rp_segm_3D]
                if cca_df is None:
                    cca_df = core.dummy_cc_stage_df(IDs)
            else:
                segm_npy_3D = np.ones(V_spots.shape, np.uint8)
                rp_segm_3D = regionprops(segm_npy_3D)
                IDs = [1]

            # Z-projection of original images before gaussian filter
            z_proj_V_spots = V_spots.max(axis=0)
            if load_ref_ch:
                z_proj_V_ref = V_ref.max(axis=0)
            else:
                _Z, _Y, _X = V_spots.shape
                z_proj_V_ref = np.zeros((_Y, _X), np.uint8)

            print('Preprocessing image...')
            # Gaussian filter of the 3D images
            if sigma > 0:
                V_spots_raw = img_as_float(V_spots)
                V_spots = gaussian(V_spots, sigma)
            else:
                V_spots = img_as_float(V_spots)
                V_spots_raw = V_spots
            if load_ref_ch:
                if sigma > 0:
                    V_ref_gauss = gaussian(V_ref, sigma)
                    V_ref_raw = img_as_float(V_ref)
                else:
                    V_ref = img_as_float(V_ref)
                    V_ref_raw = V_ref

            V_spots_masked_by_ref = np.copy(V_spots)
            if load_ref_ch:
                V_ref_masked_by_ref = np.copy(V_ref)
            if spots_ch_aligned.SizeT > 1:
                spots_ch_processed_frames[frame_i] = V_spots
            else:
                spots_ch_processed_frames = V_spots

            if load_ref_ch:
                """
                1. Segment reference channel per each cell
                """
                print('Segmenting reference channel...')
                ref_mask = np.zeros(V_ref.shape, bool)
                z_proj_V_ref_gauss = V_ref_gauss.max(axis=0)

                if inspect:
                    matplotlib.use('TkAgg')
                    img = z_proj_V_ref_gauss
                    fig, ax, _ = apps.my_try_all_threshold(img)
                    suptitle = ('NOTE: These are the results of thresholding '
                        'the entire frame. The final result will be slightly '
                        'more accurate\n because a single threshold will be '
                        'calculated for each cell if a segmentation was '
                        'performed.')
                    fig.suptitle(suptitle)
                    plt.show()
                    matplotlib.use('Agg')
                    bp.pausehere()

                if experimenting:
                    ridge_operator=frangi
                else:
                    ridge_operator=None

                if segm_npy_frame is not None:
                    for obj in rp_segm_3D:
                        ID = obj.label
                        # Skip buds
                        if cca_df.at[ID, 'Relationship'] == 'bud':
                            continue
                        (V_ref_local, slice_3D,
                        local_obj_mask) = core.preprocessing_ref(
                                          V_ref_gauss, cca_df,
                                          segm_npy_3D, ID,
                                          inspect=inspect_deep,
                                          ridge_operator=ridge_operator,
                                          zyx_resolution_pxl=zyx_resolution_pxl,
                                          bp=bp
                                          )

                        # Compute threshold for the local reference channel
                        lowT_ref = ref_ch_thresh_func(V_ref_local.max(axis=0))
                        ref_mask_local = V_ref_local > lowT_ref

                        if inspect_deep:
                            matplotlib.use('TkAgg')
                            img = V_ref_local.max(axis=0)
                            fig, ax, _ = apps.my_try_all_threshold(img)
                            suptitle = ''
                            fig.suptitle(suptitle)
                            plt.show()
                            matplotlib.use('Agg')

                            apps.imshow_tk(V_ref_local.max(axis=0),
                                   additional_imgs=[ref_mask_local.max(axis=0)],
                                   titles=['Original reference channel z-proj.',
                                           'Reference channel mask z-proj.'])
                            bp.pausehere()

                        # EXPERIMENTAL: for dapi keep only one/two brightest objects
                        if is_ref_single_obj:
                            cc_stage = cca_df.at[ID, 'Cell cycle stage']
                            ref_mask_local = core.keep_only_one_obj(
                                                    ref_mask_local, cc_stage)

                        # Remove segmented objects outside of cell
                        ref_mask_local[~local_obj_mask] = False
                        remove_small_objects(ref_mask_local, min_size=10,
                                             in_place=True)

                        # Insert segmented object into main mask
                        (ref_mask[slice_3D]
                                 [local_obj_mask]) = ref_mask_local[
                                                                 local_obj_mask]

                else:
                    V_ref_proj = V_ref_gauss.max(axis=0)
                    thresh_val_ref = ref_ch_thresh_func(V_ref_proj)
                    ref_mask = V_ref_gauss > thresh_val_ref

                if spots_ch_aligned.SizeT > 1:
                    ref_ch_thresh_frames[frame_i] = ref_mask
                else:
                    ref_ch_thresh_frames = ref_mask
                    if inspect_deep:
                        print('Final reference channel mask')
                        apps.imshow_tk(
                            V_ref_gauss,
                            additional_imgs=[ref_mask.max(axis=0)],
                            titles=['Original reference channel z-proj.',
                                    'Reference channel mask z-proj.'])
                        bp.pausehere()

                V_ref = V_ref_gauss

                if spots_ch_aligned.SizeT > 1:
                    ref_ch_processed_frames[frame_i] = V_ref
                else:
                    ref_ch_processed_frames = V_ref
            else:
                ref_mask = np.ones(V_spots.shape, bool)

            if load_ref_ch and filter_by_ref_ch:
                """
                2. Normalize reference channel by dividing for the median
                inside each cell and outside of the spots

                NOTE:
                The biological assumption is that where we have clusters
                of higher intensities reference probes/fluorophores we
                expect higher intensities of spots probes/fluorophores.
                This could lead to false positives. If this assumption
                doesn't hold for your application do not use the reference
                channel for filtering false positives.
                """
                # print('mKate threshold values = {0:.3f}, {1:.3f}'
                #       .format(background_threshold,hyst_high_thresh))
                (df_norm_ref_ch,
                 df_norm_spots_INref_ch) = orig_data.normalize_ref_ch(
                                             V_ref, ref_mask, segm_npy_3D,
                                             rp_segm_3D, V_spots=V_spots)
                V_ref_masked_by_ref[ref_mask == False] = 0
                if filter_by_ref_ch:
                    # Set to 0 pixels outside of segmented reference channel
                    # only if the reference channels is used for filtering spots
                    V_spots_masked_by_ref[ref_mask == False] = 0
                V_mKate_mtNet = V_ref[ref_mask]
                V_mNeon_mtNet = V_spots[ref_mask]
                median_ref_ch = np.median(V_mKate_mtNet)
                mean_ref_ch = np.mean(V_mKate_mtNet)
                mean_spots = np.median(V_mNeon_mtNet)
                if verbose>0 and verbose<=3:
                    print(f'Ref. channel median value = {median_ref_ch:.3f}')
                    print(f'Ref. channel mean value = {mean_ref_ch:.3f}')
                    print(f'Spots channel median value = {mean_spots:.3f}')
                    print('')
            else:
                df_norm_ref_ch = pd.DataFrame({'Cell_ID': IDs,
                                               'ref_ch norm.': [1]*len(IDs)}
                                               ).set_index('Cell_ID')
                df_norm_spots_INref_ch = pd.DataFrame({'Cell_ID': IDs,
                                              'spots_ch norm.': [1]*len(IDs)}
                                              ).set_index('Cell_ID')

            """
            3a. Find local maxima
            """
            local_max_coords = []
            print('Detecting spots...')
            # Initialize V_spots sharp
            if make_spots_sharper and segm_npy_frame is not None:
                V_spots_sharp = np.zeros_like(V_spots)
            else:
                V_spots_sharp = None

            if segm_npy_frame is not None:
                V_locals = []
                V_pos_controls = []
                backgr_vals_li = []
                local_obj_masks = []
                slice_bboxs_lower = []
                slices_3D = []
                if local_or_global_thresh == 'Local':
                    # Check if V_spots positive control is present
                    # otherwise load_PC.V_local_spots_PC is None
                    if load_PC.is_first_call:
                        load_PC.load_PC_df(TIFFs_path)
                        if load_PC.PC_df is not None:
                            load_PC.load_single_PC(TIFFs_path, user_ch_name,
                                                   sigma)
                    (local_max_coords,
                    V_spots_sharp) = core.spot_detection_local(
                                  V_spots, rp_segm_3D, segm_npy_3D, cca_df,
                                  zyx_resolution_pxl, local_maxima_thresh_func,
                                  ref_mask, zyx_vox_dim, zyx_resolution,
                                  filter_by_ref_ch=filter_by_ref_ch,
                                  make_spots_sharper=make_spots_sharper,
                                  V_spots_sharp=V_spots_sharp,
                                  inspect_deep=inspect_deep, gop_how=gop_how,
                                  gop_limit=gop_thresh_val[0], bp=bp,
                                  V_local_spots_PC=load_PC.V_local_spots_PC,
                                  local_mask_PC_3D=load_PC.local_mask_PC_3D,
                                  experimenting=experimenting)

                elif local_or_global_thresh == 'Global':
                    (local_max_coords,
                    V_spots_sharp) = core.spot_detection_global(
                                        rp_segm_3D, V_spots,
                                        local_maxima_thresh_func,
                                        zyx_resolution_pxl, ref_mask,
                                        segm_npy_3D,
                                        make_sharper=make_spots_sharper,
                                        filter_by_ref_ch=filter_by_ref_ch,
                                        inspect=inspect,
                                        inspect_deep=inspect_deep, bp=bp)

            else:
                if make_spots_sharper:
                    V_spots_blur = gaussian(V_spots, sigma=zyx_resolution_pxl)
                    V_spots_sharp = V_spots-V_spots_blur
                    V_detect = V_spots_sharp
                else:
                    V_detect = V_spots

                if inspect:
                    matplotlib.use('TkAgg')
                    img = V_detect.max(axis=0)
                    fig, ax, _ = apps.my_try_all_threshold(img)
                    suptitle = ('Automatic thresholding to outline areas '
                                'where spots will be searched.\n')
                    fig.suptitle(suptitle)
                    plt.show()
                    matplotlib.use('Agg')
                    bp.pausehere()

                local_maxima_thresh = local_maxima_thresh_func(
                                       V_detect.max(axis=0))

                z,y,x = V_spots.shape
                slice_3D = (slice(0,z), slice(0,y), slice(0,x))

                local_obj_mask = np.ones(V_spots.shape, bool)

                local_max_coords_ID = core.spot_detection_local(
                                    V_spots, rp_segm_3D, segm_npy_3D, cca_df,
                                    zyx_resolution_pxl, local_maxima_thresh_func,
                                    ref_mask, zyx_vox_dim, zyx_resolution,
                                    filter_by_ref_ch=filter_by_ref_ch,
                                    make_spots_sharper=make_spots_sharper,
                                    V_spots_sharp=V_spots_sharp,
                                    inspect_deep=inspect_deep, gop_how=gop_how,
                                    gop_limit=gop_thresh_val[0], bp=bp,
                                    V_local_spots_PC=load_PC.V_local_spots_PC,
                                    local_mask_PC_3D=load_PC.local_mask_PC_3D,
                                    experimenting=experimenting
                )
                if inspect:
                    apps.imshow_tk(V_detect.max(axis=0),
                        dots_coords=local_max_coords_ID,
                        x_idx=2)

                local_max_coords = local_max_coords_ID

            if verbose>0 and verbose<=3:
                print(f'Total number of peaks found = {len(local_max_coords)}')


            if inspect:
                inspect_app = apps.inspect_effect_size_app('All detected spots '
                              '(before filtering peaks that are too close)')
                inspect_app.run(load_ref_ch, V_spots, segm_npy_3D, IDs,
                                local_max_coords, channel_name, V_ref,
                                ref_mask, ref_channel_name,
                                zyx_resolution_pxl[0],
                                local_max_coords, sharp_V_spots=V_spots_sharp)
                if inspect_app.next:
                    continue



            if len(local_max_coords) == 0:
                print(f'No spots found at {pos_foldername}, '
                      f'frame {frame_i}')
                orig_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    predict_cell_cycle=is_ref_single_obj)
                ellip_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                p_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                p_ellip_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                if do_gaussian_fit:
                    spotfit_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                        zyx_vox_dim, timestamp, finterval, frame_i,
                                        ref_mask, None, df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        ref_chQUANT_data=orig_summary,
                                        predict_cell_cycle=is_ref_single_obj)
                orig_summary.df_zyx = []
                continue


            """"
            4a. Ellipsoid test
            A valid point is defined as not lying inside of any ellipsoid centred
            at points with higher intensity.
            The ellipsoid size is determined by yx and z resolution limits.
            """
            ellips_test = core.filter_points_resol_limit(local_max_coords,
                                        zyx_resolution_pxl, V_spots.shape,
                                        filter_z_bound=filter_z_bound)
            ellips_test_idx = ellips_test.get_valid_points_idx(local_max_coords)

            local_max_coords_e_test = ellips_test.valid_points

            if verbose>0 and verbose<=3:
                print('')
                print('Total number of peaks after '
                      f'ellipsoid test: {len(local_max_coords_e_test)}')

            """
            4b. Calculate spots metrics
            """
            print('')
            (df_spots, local_max_coords_e_test,
            spots_mask) = core.metrics_spots().calc_metrics_spots(
                                   V_spots_raw, V_ref, local_max_coords_e_test,
                                   df_norm_ref_ch, zyx_resolution,
                                   zyx_vox_dim, segm_npy_3D,
                                   is_segm_3D=segm_info=='3D',
                                   df_spots_ch_norm=df_norm_spots_INref_ch,
                                   orig_data=orig_data,
                                   ref_ch_mask=ref_mask,
                                   do_bootstrap=do_bootstrap,
                                   filter_by_ref_ch=filter_by_ref_ch,
                                   V_spots_sharp=V_spots_sharp)



            if inspect:
                if filter_by_ref_ch:
                    cols = ['vox_spot', '|norm|_spot', '|norm|_ref',
                            which_effsize, 'z', 'y', 'x',
                            '|spot|:|ref| t-value',
                            '|spot|:|ref| p-value (t)',
                            'effsize_cohen_s']
                else:
                    cols = ['vox_spot', '|norm|_spot', '|norm|_ref',
                            which_effsize, 'z', 'y', 'x',
                            'peak_to_background ratio',
                            'backgr_INcell_OUTspot_mean',
                            'backgr_INcell_OUTspot_std',
                            'backgr_INcell_OUTspot_median']
                print(df_spots.sort_values('vox_spot')[cols])
                bp.pausehere()


            """
            5. Assign Cell IDs as MultiIndex to allow for Multi-level indexing
            """
            orig_df_MulIDs = core.df_MultiIndex_IDs(df_spots)
            ellips_df_MulIDs = core.df_MultiIndex_IDs(df_spots)

            if verbose>1 and verbose<=3:
                print('Data that passed ellipsoid test:')
                print(ellips_df_MulIDs.df_IDs)


            """
            6. Iteratively repeat the a), b) and c) steps (see below) until
            the number of nucleoids stops decreasing.
                a) Normalize mNeon channel by the mean outside the nucleoids
                   and inside the mito network
                b) Filter good peaks by goodness-of-peak (gop) test
                   between normalized intensities of mNeon and mKate signals
                   inside the resolution limited volume centered at each peak.
                c) Repeat a), b) until the number of peaks doesn't stops
                   decreasing.
            """
            num_nucl = -1
            peaks_coords_gop_test = local_max_coords_e_test.copy()
            num_peaks = len(peaks_coords_gop_test)
            df = df_spots.copy()
            count_iter = 0
            pbar = tqdm(desc='Iterating gop-test', total=num_peaks,
                        unit=' spot', leave=True, position=0, ncols=100)
            while num_nucl < num_peaks and num_nucl != 0:
                if count_iter > 0:
                    num_peaks = num_nucl
                """
                6a. Estimate mNeon background fluorescence intensity by mean
                or median of the intensities outside of nucleoids and within
                the mtNetwork for each cell.
                """
                if load_ref_ch and filter_by_ref_ch:
                    df_norm_spots, spots_mask = ellip_data.normalize_spots_ch(
                                           V_spots, segm_npy_3D, rp_segm_3D,
                                           zyx_vox_dim, zyx_resolution,
                                           peaks_coords_gop_test, ref_mask,
                                           df_ellips_test=None)
                    df_norm_ref_ch = ellip_data.normalize_ref_ch(V_ref,
                                              ref_mask, segm_npy_3D,
                                              rp_segm_3D,
                                              use_outside_spots_mask=True)
                else:
                    df_norm_spots = pd.DataFrame({'Cell_ID': IDs,
                                                  'spots_ch norm.': [1]*len(IDs)}
                                                  ).set_index('Cell_ID')
                OUT_spots_mask = np.invert(spots_mask)
                OUT_spots_IN_ref_mask = np.logical_and(ref_mask, OUT_spots_mask)
                V_spots_OUTspots_INref = V_spots[OUT_spots_IN_ref_mask]
                mean_OUTspots_INref = np.mean(V_spots_OUTspots_INref)
                median_OUTspots_INref = np.median(V_spots_OUTspots_INref)
                if verbose>1 and verbose<=3:
                    print('')
                    print('Mean of mNeon outside of nucleoids '
                          f'(and within mtNetwork): {mean_OUTspots_INref:.3f}')
                    print('Median of mNeon outside of nucleoids '
                          f'(and within mtNetwork): {median_OUTspots_INref:.3f}')

                    print(df_norm_ref_ch)
                    print(df_norm_spots)

                # print(df_norm_ref_ch)
                # apps.imshow_tk(segm_npy_3D.max(axis=0))
                # import pdb; pdb.set_trace()
                """
                6b. Compute metrics needed for the gop test
                """
                if count_iter > 0:
                    df, _, spots_mask = (core.metrics_spots()
                                        .calc_metrics_spots(
                                        V_spots_raw, V_ref,
                                        peaks_coords_gop_test, df_norm_ref_ch,
                                        zyx_resolution, zyx_vox_dim,
                                        segm_npy_3D,
                                        df_spots_ch_norm=df_norm_spots,
                                        orig_data=orig_data,
                                        ref_ch_mask=ref_mask,
                                        do_bootstrap=do_bootstrap,
                                        filter_by_ref_ch=filter_by_ref_ch,
                                        V_spots_sharp=V_spots_sharp))

                if inspect:
                    gop_bounds = (df[which_effsize].min(),
                                  df[which_effsize].max())
                    inspect_app = apps.inspect_effect_size_app(
                                  f'Detected spots after {count_iter} '
                                  'iterations of goodness-of-peak test '
                                  '(after filtering peaks that are too close)')
                    inspect_app.run(load_ref_ch, V_spots, segm_npy_3D, IDs,
                                    peaks_coords_gop_test, channel_name, V_ref,
                                    ref_mask, ref_channel_name,
                                    zyx_resolution_pxl[0],
                                    local_max_coords_e_test,
                                    gop_bounds=gop_bounds, df_spots=df,
                                    gop_how=gop_how,
                                    which_effsize=which_effsize,
                                    prev_df_spots=df_spots,
                                    sharp_V_spots=V_spots_sharp)
                    if inspect_app.next:
                        break
                    # Testing
                    apps.inspect_effect_size(gop_thresh_val, df, filter_by_ref_ch,
                                        do_bootstrap, peaks_coords_gop_test,
                                        V_spots)

                """
                6c. Goodness-of-peak (gop) test
                """
                df_gop_test = core.filter_good_peaks(
                                      df, gop_thresh_val,
                                      how=gop_how,
                                      which_effsize=which_effsize
                )
                peaks_coords_gop_test = df_gop_test[['z', 'y', 'x']].to_numpy()

                if verbose == 3:
                    print(df_gop_test)

                num_nucl = len(peaks_coords_gop_test)
                count_iter += 1
                if verbose>1 and verbose<=3:
                    print(f'    Iteration number = {count_iter}')
                    print(f'    Number of peaks previous iter. = {num_peaks}')
                    print(f'    Number of peaks current iter. = {num_nucl}')
                # bp.pausehere()
                pbar.update(num_peaks-num_nucl)
                """End of gop test while loop"""

            pbar.update(num_nucl)
            pbar.close()

            if inspect:
                if inspect_app.next:
                    continue

            if verbose>1 and verbose<=3:
                print(df_gop_test)

            if verbose>0 and verbose<=3:
                print('')
                print('Total number of peaks after gop-test = '
                      f'{num_nucl}')
                print('')

            if num_nucl == 0:
                print('')
                print(f'No VALID spots found at {pos_foldername}, '
                      f'frame {frame_i}')
                orig_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    predict_cell_cycle=is_ref_single_obj)
                ellip_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                p_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                p_ellip_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                    zyx_vox_dim, timestamp, finterval, frame_i,
                                    ref_mask, None, df_norm_ref_ch, cca_df,
                                    V_spots.shape, rp_segm_3D, is_segm_3D,
                                    ref_chQUANT_data=orig_summary,
                                    predict_cell_cycle=is_ref_single_obj)
                if do_gaussian_fit:
                    spotfit_summary.generate_summary_df(IDs, 0, segm_npy_3D,
                                        zyx_vox_dim, timestamp, finterval, frame_i,
                                        ref_mask, None, df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        ref_chQUANT_data=orig_summary,
                                        predict_cell_cycle=is_ref_single_obj)
                z_proj_V_spots_norm = V_spots.max(axis=0)
                z_proj_V_ref_norm = V_ref.max(axis=0)
                local_max_coords_p_ellips_test = np.zeros((1,3), int)
                orig_summary.df_zyx = []
                continue

            """
            7. Ellipsoid test on data that passed the gop-test
            """
            p_ellips_test = core.filter_points_resol_limit(
                                            peaks_coords_gop_test,
                                            zyx_resolution_pxl, V_spots.shape,
                                            filter_z_bound=filter_z_bound)
            p_ellips_test_idx = p_ellips_test.get_valid_points_idx(
                                                        peaks_coords_gop_test)
            df_gop_test_dist = df_gop_test.filter(p_ellips_test_idx, axis=0)
            local_max_coords_p_ellips_test = p_ellips_test.valid_points


            """
            8a. spotQUANT on data that passed first the p-test and then the
            ellipsoid test
            """
            p_df_MulIDs = core.df_MultiIndex_IDs(df_gop_test)
            p_ellips_df_MulIDs = core.df_MultiIndex_IDs(df_gop_test_dist,
                                                              verb=False)

            num_nucl_ellip_test = len(p_ellips_df_MulIDs.df_IDs)

            if num_nucl_ellip_test == 0:
                print('')
                print(f'No VALID spots found at {pos_foldername}, '
                      f'frame {frame_i} after ellipsoid test')
                orig_summary.generate_summary_df(orig_df_MulIDs.IDs_unique,
                                                orig_df_MulIDs.num_spots,
                                                segm_npy_3D, zyx_vox_dim,
                                                timestamp, finterval, frame_i,
                                                ref_mask, df_norm_spots,
                                                df_norm_ref_ch, cca_df,
                                                V_spots.shape, rp_segm_3D,
                                                is_segm_3D,
                                                predict_cell_cycle=is_ref_single_obj)
                ellip_summary.generate_summary_df(ellips_df_MulIDs.IDs_unique,
                                            ellips_df_MulIDs.num_spots,
                                            segm_npy_3D, zyx_vox_dim,
                                            timestamp, finterval, frame_i,
                                            ref_mask, df_norm_spots,
                                            df_norm_ref_ch, cca_df,
                                            V_spots.shape, rp_segm_3D, is_segm_3D,
                                            ref_chQUANT_data=orig_summary,
                                            predict_cell_cycle=is_ref_single_obj)
                p_summary.generate_summary_df(p_df_MulIDs.IDs_unique,
                                            p_df_MulIDs.num_spots,
                                            segm_npy_3D, zyx_vox_dim,
                                            timestamp, finterval, frame_i,
                                            ref_mask, df_norm_spots,
                                            df_norm_ref_ch, cca_df,
                                            V_spots.shape, rp_segm_3D, is_segm_3D,
                                            ref_chQUANT_data=orig_summary,
                                            predict_cell_cycle=is_ref_single_obj)
                p_ellip_summary.generate_summary_df(p_ellips_df_MulIDs.IDs_unique,
                                            p_ellips_df_MulIDs.num_spots,
                                            segm_npy_3D, zyx_vox_dim,
                                            timestamp, finterval, frame_i,
                                            ref_mask, df_norm_spots,
                                            df_norm_ref_ch, cca_df,
                                            V_spots.shape, rp_segm_3D, is_segm_3D,
                                            V_spots_raw=V_spots_raw,
                                            ref_chQUANT_data=orig_summary,
                                            spots_mask=spots_mask,
                                            predict_cell_cycle=is_ref_single_obj,
                                            filter_by_ref_ch=filter_by_ref_ch,
                                            debug=True)
                if do_gaussian_fit:
                    spotfit_summary.generate_summary_df(
                                        p_ellips_df_MulIDs.IDs_unique,
                                        p_ellips_df_MulIDs.num_spots,
                                        segm_npy_3D, zyx_vox_dim,
                                        timestamp, finterval, frame_i,
                                        ref_mask, df_norm_spots,
                                        df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        V_spots_raw=V_spots_raw,
                                        ref_chQUANT_data=orig_summary,
                                        gaussian_fit_done=do_gaussian_fit,
                                        spots_mask=spots_mask,
                                        predict_cell_cycle=is_ref_single_obj,
                                        filter_by_ref_ch=filter_by_ref_ch)

            if verbose>0 and verbose<=3:
                print('')
                print('Total number of peaks after ellipsoid- and gop-test = '
                      f'{num_nucl_ellip_test}')
                print('')

            if verbose>1 and verbose<=3:
                print('Data that passed gop-test and ellipsoid test:')
                print(p_ellips_df_MulIDs.df_IDs)

            if do_spotSIZE:
                print(f'Performing spotQUANT on {pos_path}...')
            df_spotFIT, spots_3D_lab = spotQUANT_data.spotQUANT(
                                       p_ellips_df_MulIDs, V_spots_raw,
                                       ref_mask, rp_segm_3D, segm_npy_3D, IDs,
                                       zyx_vox_dim, zyx_resolution,
                                       verbose=verbose,
                                       filter_by_ref_ch=filter_by_ref_ch,
                                       inspect=inspect_deep)

            if do_spotSIZE:
                spots_mask = spots_3D_lab > 0

            if verbose>1 and verbose<=3 and do_spotSIZE:
                print('spotQUANT data:')
                print(df_spotFIT)

            """
            8b. Filter spots by size based on spotfit results
            """
            if do_gaussian_fit:
                if debug or inspect>0:
                    _t = 'Inspect effect of spot size limits'
                    apps.inspect_spotFIT_app(_t).run(
                                         V_spots, segm_npy_3D,
                                         IDs, df_spotFIT, channel_name,
                                         spotsize_limits_pxl,
                                         spotQUANT_data.spots_3D_labs,
                                         spotQUANT_data.ID_bboxs_lower,
                                         spotQUANT_data.ID_3Dslices,
                                         spotQUANT_data.IDs_with_spots,
                                         spotQUANT_data.dfs_intersect,
                                         sharp_V_spots=V_spots_sharp,
                                         which_ax1_data='sigma_yx_mean')
                df_spotFIT = spotQUANT_data.filter_spots_by_size(
                                               df_spotFIT, spotsize_limits_pxl)
                spotfit_MultiIdx_IDs = core.df_MultiIndex_IDs(df_spotFIT)
                print('')
                print('Total number of peaks after spotFIT = '
                      f'{len(spotfit_MultiIdx_IDs.df_IDs)}')
                print('')


            """"
            9. Append frame dataframes to HDF5 file with key = 'frame_{frame_i}'
            """
            orig_data.appn_HDF(orig_df_MulIDs.df_IDs, timestamp,
                                                    finterval, frame_i)
            ellip_data.appn_HDF(ellips_df_MulIDs.df_IDs, timestamp,
                                                    finterval, frame_i)
            p_data.appn_HDF(p_df_MulIDs.df_IDs, timestamp,
                                                    finterval, frame_i)
            p_ellip_data.appn_HDF(p_ellips_df_MulIDs.df_IDs, timestamp,
                                                    finterval, frame_i)
            if do_spotSIZE:
                spotQUANT_data.appn_HDF(df_spotFIT, timestamp,
                                        finterval, frame_i)

            """
            10. Create summary dataframes and append them
            (they will be concantenated into csv)
            """
            orig_summary.generate_summary_df(orig_df_MulIDs.IDs_unique,
                                            orig_df_MulIDs.num_spots,
                                            segm_npy_3D, zyx_vox_dim,
                                            timestamp, finterval, frame_i,
                                            ref_mask, df_norm_spots,
                                            df_norm_ref_ch, cca_df,
                                            V_spots.shape, rp_segm_3D,
                                            is_segm_3D,
                                            predict_cell_cycle=is_ref_single_obj)
            ellip_summary.generate_summary_df(ellips_df_MulIDs.IDs_unique,
                                        ellips_df_MulIDs.num_spots,
                                        segm_npy_3D, zyx_vox_dim,
                                        timestamp, finterval, frame_i,
                                        ref_mask, df_norm_spots,
                                        df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        ref_chQUANT_data=orig_summary,
                                        predict_cell_cycle=is_ref_single_obj)
            p_summary.generate_summary_df(p_df_MulIDs.IDs_unique,
                                        p_df_MulIDs.num_spots,
                                        segm_npy_3D, zyx_vox_dim,
                                        timestamp, finterval, frame_i,
                                        ref_mask, df_norm_spots,
                                        df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        ref_chQUANT_data=orig_summary,
                                        predict_cell_cycle=is_ref_single_obj)
            p_ellip_summary.generate_summary_df(p_ellips_df_MulIDs.IDs_unique,
                                        p_ellips_df_MulIDs.num_spots,
                                        segm_npy_3D, zyx_vox_dim,
                                        timestamp, finterval, frame_i,
                                        ref_mask, df_norm_spots,
                                        df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        V_spots_raw=V_spots_raw,
                                        ref_chQUANT_data=orig_summary,
                                        spots_mask=spots_mask,
                                        predict_cell_cycle=is_ref_single_obj,
                                        filter_by_ref_ch=filter_by_ref_ch,
                                        debug=True)

            if do_gaussian_fit:
                spotfit_summary.generate_summary_df(
                                        spotfit_MultiIdx_IDs.IDs_unique,
                                        spotfit_MultiIdx_IDs.num_spots,
                                        segm_npy_3D, zyx_vox_dim,
                                        timestamp, finterval, frame_i,
                                        ref_mask, df_norm_spots,
                                        df_norm_ref_ch, cca_df,
                                        V_spots.shape, rp_segm_3D, is_segm_3D,
                                        V_spots_raw=V_spots_raw,
                                        ref_chQUANT_data=orig_summary,
                                        df_spotFIT=spotfit_MultiIdx_IDs.df_IDs,
                                        gaussian_fit_done=do_gaussian_fit,
                                        spots_mask=spots_mask,
                                        predict_cell_cycle=is_ref_single_obj,
                                        filter_by_ref_ch=filter_by_ref_ch)

            """Append skeleton coordinates to H5 database"""
            if load_ref_ch:
                orig_summary.appn_HDF_mtNet(timestamp, finterval, frame_i)

            tend_frame = time()

            # TODO replace with progress bar
            print('')
            print(f'Frame {frame_i+1}/{frange[1]} done! '
                  f'Execution time: {tend_frame-t0_frame:.3f} s')

            """End of for loop iterating frames"""

        """Save data"""
        if do_save:
            print('')
            print('Saving data...')
            if load_ref_ch:
                ref_mask_path = os.path.join(
                     spots_ch_data.images_path,
                     f'{spots_ch_data.basename}_{user_ref_ch_name}_mask.npz')
                np.savez_compressed(ref_mask_path, ref_ch_thresh_frames)
            spots_labs_path = os.path.join(
                                 spots_ch_data.images_path,
                                 f'{spots_ch_data.basename}_spot_labels.npz')
            # np.savez_compressed(spots_labs_path, **spots_lab_frames_dict)

            # Remove older files versions
            # old_spots_labs_path = os.path.join(
            #                       spots_ch_data.images_path,
            #                       f'{spots_ch_data.basename}_mNeon_labels.npy')
            # old_ref_mask_path_1 = os.path.join(
            #      spots_ch_data.images_path,
            #      f'{spots_ch_data.basename}_ref_ch_mask.npy')
            # old_ref_mask_path_2 = os.path.join(
            #      spots_ch_data.images_path,
            #      f'{spots_ch_data.basename}_mKate_thresh.npy')
            # if os.path.isfile(old_spots_labs_path):
            #     os.remove(old_spots_labs_path)
            # if os.path.isfile(old_spots_labs_path):
            #     os.remove(old_ref_mask_path_1)
            # if os.path.isfile(old_spots_labs_path):
            #     os.remove(old_ref_mask_path_2)

            # np.save(mNeon_processed_path, spots_ch_processed_frames)
            # np.save(mKate_processed_path, ref_ch_processed_frames)
        orig_summary.write_to_csv('w') #concatenate frames and the write to csv
        ellip_summary.write_to_csv('w')
        p_summary.write_to_csv('w')
        p_ellip_summary.write_to_csv('w')
        if do_gaussian_fit:
            spotfit_summary.write_to_csv('w')
        orig_data.close_HDF()
        ellip_data.close_HDF()
        p_data.close_HDF()
        p_ellip_data.close_HDF()
        if do_spotSIZE:
            spotQUANT_data.close_HDF()
        if do_spotSIZE:
            if do_gaussian_fit:
                spotQUANT_data.save_spotFIT_done()
            else:
                spotQUANT_data.save_spotFIT_NOT_done()
        orig_summary.close_HDF_mtNetZYX()
        orig_summary.write_header_info()
        if do_save:
            print('Files saved to ' + data_path)

        tend_pos = time()

        print(f'{pos_foldername} done! '
              f'Execution time: {tend_pos-t0_pos:.3f} s')

        print('----------------------------------')
        print('')

        """End of for loop iterating Position_n folders"""

        is_first_call = False

    print('Generating all positions concantenated data...')

    # Create spotMAX folder for all positions concantenated data
    AllPos_summary_df = core.spotMAX_concat_pos(TIFFs_path, vNUM=vNUM,
                                                  run_num=run_num,
                                                  do_save=do_save)

    # Load all positions dataframes from HDD
    saveConcat = do_save
    try:
        AllPos_summary_df.load_df_from_allpos(vNUM=vNUM, run_num=run_num)

        (ellips_test_df_moth, ellips_test_df_bud,
        ellips_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.ellips_test_df_li)

        (p_test_df_moth, p_test_df_bud,
        p_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.p_test_df_li)

        (p_ellips_test_df_moth, p_ellips_test_df_bud,
        p_ellips_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.p_ellips_test_df_li)

        if AllPos_summary_df.spotfit_df_li:
            (spotfit_df_moth, spotfit_df_bud,
            spotfit_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.spotfit_df_li)
    except:
        saveConcat = False
        traceback.print_exc()
        print('IGNORE error or open an issue on GitHub')

    print('Done!')

    if saveConcat:
        print('Saving all positions concantenated data...')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_moth,
                                         '1_AllPos_ellip_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_bud,
                                         '1_AllPos_ellip_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_tot,
                                         '1_AllPos_ellip_test_TOT_data.csv')

        AllPos_summary_df.save_AllPos_df(p_test_df_moth,
                                         '2_AllPos_p-_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(p_test_df_bud,
                                         '2_AllPos_p-_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(p_test_df_tot,
                                         '2_AllPos_p-_test_TOT_data.csv')

        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_moth,
                                         '3_AllPos_p-_ellip_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_bud,
                                         '3_AllPos_p-_ellip_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_tot,
                                         '3_AllPos_p-_ellip_test_TOT_data.csv')

        if AllPos_summary_df.spotfit_df_li:
            AllPos_summary_df.save_AllPos_df(spotfit_df_moth,
                                         '4_AllPos_spotfit_MOTH_data.csv')
            AllPos_summary_df.save_AllPos_df(spotfit_df_bud,
                                         '4_AllPos_spotfit_BUD_data.csv')
            AllPos_summary_df.save_AllPos_df(spotfit_df_tot,
                                         '4_AllPos_spotfit_TOT_data.csv')

        AllPos_summary_df.save_ALLPos_analysis_inputs(spotMAX_inputs_path)
        print('Done!')

    tend_exp = time()

    print(f'Analysis of experiment {TIFFs_path} done!\n'
          f'Execution time: {tend_exp-t0_exp:.3f} s')
    print('##################################################')
    print('')

    """End of for loop iterating experiment folders"""

tend_tot = time()

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

"""Create some z proj for plotting"""
z_proj_V_spots_norm = (V_spots/mean_OUTspots_INref).max(axis=0)
if load_ref_ch:
    median_ref_ch = np.median(V_ref)
    z_proj_V_ref_norm = (V_ref/median_ref_ch).max(axis=0)
    z_proj_V_ref_black_bg = (V_ref_masked_by_ref).max(axis=0)
    z_proj_norm_ref_ch = z_proj_V_ref_black_bg/median_ref_ch
else:
    _Z, _Y, _X = V_spots.shape
    z_proj_V_ref_norm = np.zeros((_Y, _X), np.uint8)
    z_proj_V_ref_black_bg = np.zeros((_Y, _X), np.uint8)
    z_proj_norm_ref_ch = np.zeros((_Y, _X), np.uint8)


#Plot images
matplotlib.use('TkAgg')
fig, ax = plt.subplots(2,3)

ax[0,0].imshow(z_proj_V_spots)
ax[0,0].axis('off')
ax[0,0].set_title('Spots z-projection')

ax[0,1].imshow(z_proj_V_ref, cmap='gist_heat')
ax[0,1].axis('off')
ax[0,1].set_title('Reference channel z-projection')

ax[0,2].imshow(z_proj_V_spots)
ax[0,2].plot(local_max_coords[:, 2],
             local_max_coords[:, 1], 'r.') #add red dots at peaks positions
ax[0,2].axis('off')
ax[0,2].set_title('Spots plus local maxima')

ax10_img = z_proj_V_mNeon_norm if V_spots_sharp is None else V_spots_sharp.max(axis=0)
ax[1,0].imshow(ax10_img)
ax[1,0].axis('off')
ax[1,0].set_title('Spots normalized gaussian filtered z-projection')

ax[1,1].imshow(z_proj_norm_ref_ch, cmap='gist_heat')
ax[1,1].axis('off')
ax[1,1].set_title('Reference channel normalized gaussian filtered z-projection')

ax[1,2].imshow(z_proj_V_spots_norm)
ax[1,2].plot(local_max_coords_p_ellips_test[:, 2],
             local_max_coords_p_ellips_test[:, 1],
             'r.')  # add red dots at peaks positions
ax[1,2].axis('off')
ax[1,2].set_title('Spots plus valid local maxima')

# for a in ax.ravel():
#     a.set_xlim(110, 250)
#     a.set_ylim(300, 110)

# Plot mtNetwork
if load_ref_ch:
    core.plot_mtNet(orig_summary.df_zyx, ax, lw=1.5)

fig_path = '{}/{}_plt.pdf'.format(data_path, spots_ch_data.basename)

folder_name = os.path.basename(os.path.dirname(data_path))
fig.suptitle('Total execution time = {0:.3f} s\nFolder name: {1}'
            .format(tend_tot-t0_tot, folder_name), y=0.97, size=14)

try:
    #Display plots maximized window
    mng = plt.get_current_fig_manager()
    screens = Display().get_screens()
    num_screens = len(screens)
    if num_screens==1:
        mng.window.state('zoomed')  # display plots window maximized
    else:
        width = screens[0].width
        height = screens[0].height - 70
        left = width-7
        geom = "{}x{}+{}+0".format(width,height,left)
        mng.window.wm_geometry(geom)  # move GUI window to second monitor
                                      # with string "widthxheight+x+y"
except:
    try:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    except:
        pass

def resize_cb(event):
    ax02_l, ax02_b, ax02_r, ax02_t = ax[0,2].get_position().get_points().flatten()
    fig.text((ax02_r+ax02_l)/2, ax02_b-0.015,
         'Number of spots = {}'.format(len(local_max_coords)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    ax12_l, ax12_b, ax12_r, ax12_t = ax[1,2].get_position().get_points().flatten()
    fig.text((ax12_r+ax12_l)/2, ax12_b-0.015,
         'Number of spots = {}'.format(len(local_max_coords_p_ellips_test)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    ax11_l, ax11_b, ax11_r, ax11_t = ax[1,1].get_position().get_points().flatten()
    fig.text((ax11_r+ax11_l)/2, ax11_b-0.015, 'Reference channel length (um) = {0:.2f}'
                             .format(p_ellip_summary.get_tot_mtNet_len()),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    ax11_l, ax11_b, ax11_r, ax11_t = ax[1,1].get_position().get_points().flatten()
    fig.text((ax11_r+ax11_l)/2, ax11_b-0.015, 'Reference channel length (um) = {0:.2f}'
                             .format(p_ellip_summary.get_tot_mtNet_len()),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)

exp_name = os.path.basename(os.path.dirname(
                            os.path.dirname(
                            os.path.dirname(data_path))))
fig.canvas.mpl_connect('resize_event', resize_cb)
fig.canvas.manager.set_window_title(f'spotMAX on {exp_name}')
plt.show()
