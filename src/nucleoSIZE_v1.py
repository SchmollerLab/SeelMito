import os
import shutil
import re
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from sys import exit
from tkinter.filedialog import folder_dialog
from tkinter.messagebox import tk_breakpoint
from skimage import io
from skimage.draw import circle
from skimage.filters import gaussian
from natsort import natsorted
from time import time
from mitoQUANT_Lib_v1 import (load_data, spheroid, single_entry_messagebox,
                   select_exp_folder, beyond_listdir_pos)

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)

plt.dark()

class num_pos_toQuant_tk:
    def __init__(self, tot_frames):
        root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        root.geometry('+800+400')
        root.attributes("-topmost", True)
        tk.Label(root,
                 text="How many positions do you want to QUANT?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        tk.Label(root,
                 text="(There are a total of {} positions).".format(tot_frames),
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start position",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=tk.E, padx=4)
        tk.Label(root,
                 text="# of positions to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                            textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(1))
        sv_sf.trace_add("write", self.set_all)
        self.start_frame = start_frame
        start_frame.grid(row=2, column=1, pady=8, sticky=tk.W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=tk.W)
        tk.Button(root,
                  text='All',
                  command=self.set_all,
                  width=8).grid(row=3,
                                 column=2,
                                 pady=4, padx=4)
        tk.Button(root,
                  text='OK',
                  command=self.ok,
                  width=12).grid(row=4,
                                 column=0,
                                 pady=8,
                                 columnspan=3)
        root.bind('<Return>', self.ok)
        start_frame.focus_force()
        start_frame.selection_range(0, tk.END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            rightRange = self.tot_frames - startf + 1
            self.num_frames.delete(0, tk.END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf + 1
                self.num_frames.delete(0, tk.END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def ok(self, event=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf-1, stopf-1)
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

def gaussian_3D(coeffs, ZYX):
    z, y, x = ZYX
    s_z, s_xy, mu_z, mu_y, mu_x = coeffs
    model = np.exp(-((x-mu_x)**2+(y-mu_y)**2)/())

bp = tk_breakpoint()

ref_z = 15

# Folder dialog
selected_path = folder_dialog(title="Select folder containing valid experiments")
selector = select_exp_folder()
selector.do_all = True
selector.do_remaining = False

vNUM = single_entry_messagebox(entry_label='Script version number (e.g. v9)',
                               input_txt='v10', toplevel=False).entry_txt
run_num = single_entry_messagebox(entry_label='Analysis run number: ',
                               input_txt='1', toplevel=False).entry_txt

if not os.listdir(selected_path)[0].find('Position_') != -1:
    beyond_listdir_pos = beyond_listdir_pos(selected_path, vNUM=vNUM,
                                            run_num=run_num)
    main_paths = selector.run_widget(beyond_listdir_pos.all_exp_info,
                             title='mitoQUANT: Select experiment to analyse',
                             label_txt='Select experiment to analyse',
                             full_paths=beyond_listdir_pos.TIFFs_path,
                             showinexplorer_button=True,
                             all_button=True,
                             remaining_button=True)
else:
    # The selected path is already the fodler containing Position_n folders
    main_paths = [selected_path]

if selector.do_remaining:
    main_paths = main_paths[selector.current_idx:]

if not selector.do_all and not selector.do_remaining:
    main_paths = [main_paths]

data_filename = '3_p-_ellip_test_data'
h5_df_name = f'{run_num}_{data_filename}_{vNUM}.h5'
summ_df_name = f'{run_num}_{data_filename}_Summary_{vNUM}.csv'
nuclSIZE_h5_name = f'{run_num}_4_p-_ellip_size_test_data_{vNUM}.h5'

user_path = os.path.expanduser("~")

num_exp = len(main_paths)

for exp_idx, path in enumerate(main_paths):
    print(f'NucleoSIZEing: {path}')
    pos_foldnames = natsorted(os.listdir(path))
    """Load metadata (zyx resolution limit and voxel dimensions) from mNeon tif"""
    img_fold_path = f'{path}/{pos_foldnames[0]}/Images'
    img_filenames = os.listdir(img_fold_path)
    for f, img_fn in enumerate(img_filenames):
        if img_fn.find('_mNeon.tif') != -1:
            break
    nucleodata_path = f'{path}/{pos_foldnames[0]}/NucleoData'
    analysis_inputs_name = f'{run_num}_{vNUM}_analysis_inputs.csv'
    analysis_inputs_path = f'{nucleodata_path}/{analysis_inputs_name}'
    if os.path.exists(analysis_inputs_path):
        df_inputs = pd.read_csv(analysis_inputs_path,
                                index_col='Description')
        sigma = float(df_inputs.at['Gaussian filter sigma:', 'Values'])
        z_resolution_limit = float(df_inputs.loc['Z resolution limit (μm):',
                                                 'Values'])
    else:
        z_resolution_limit = float(single_entry_messagebox(
                      entry_label=u'Z resolution limit (\u03bcm):',
                      input_txt='1', toplevel=False).entry_txt)

    mNeon_path = f'{img_fold_path}/{img_fn}'
    mNeon_data = load_data(mNeon_path, 'mNeon', load_mask=False)
    zyx_vox_dim = mNeon_data.zyx_vox_dim
    NA = mNeon_data.NA
    wavelen = mNeon_data.wavelen
    airy_radius = (1.22 * wavelen)/(2*NA)
    sphere_radius = airy_radius*1E-3 #convert nm to µm
    zyx_resolution = np.asarray([z_resolution_limit, sphere_radius,
                                                     sphere_radius])
    zyx_resolution_print = [round(r, 3) for r in zyx_resolution]
    print(f'zyx resolution limit = {zyx_resolution_print}')

    # Read h5 DataFrame and loop each peak
    if exp_idx == 0 and num_exp==1:
        pstart, pend = num_pos_toQuant_tk(len(pos_foldnames)).frange
    else:
        pstart, pend = 0, len(pos_foldnames)
    t0 = time()
    sigma_ok = False
    """Iterate positions"""
    for p, pfn in enumerate(pos_foldnames[pstart:pend]):
        print(f'NucleoSIZEing experiment n. {exp_idx+1}/{num_exp}, '
              f'{pfn} ({p+1}/{pend-pstart})...')
        img_fold_path = f'{path}/{pfn}/Images'
        nucleodata_path = f'{path}/{pfn}/NucleoData'
        mNeon_proc_found = False
        mKate_proc_found = False
        files_in_img_fold = os.listdir(img_fold_path)
        for k, img_fn in enumerate(files_in_img_fold):
            if img_fn.find('_mNeon_processed.npy') != -1:
                mN_proc_i = k
                mNeon_proc_found = True
            elif img_fn.find('_mKate_processed.npy') != -1:
                mK_proc_i = k
                mKate_proc_found = True
            elif img_fn.find('_mNeon.tif') != -1:
                mN_tif = k
            elif img_fn.find('_mKate.tif') != -1:
                mK_tif = k
        if not mNeon_proc_found or not mKate_proc_found:
            if not os.path.exists(analysis_inputs_path):
                if not sigma_ok:
                    sigma = float(single_entry_messagebox(
                              entry_label='Sigma for mNeon gaussian filter: ',
                              input_txt='0.75', toplevel=False).entry_txt)
            sigma_ok = True
        if mNeon_proc_found:
            mNeon_path = f'{img_fold_path}/{files_in_img_fold[mN_proc_i]}'
            mNeon_data = np.load(mNeon_path)
        else:
            mNeon_path = f'{img_fold_path}/{files_in_img_fold[mN_tif]}'
            mNeon_data = io.imread(mNeon_path)

        if mKate_proc_found:
            mKate_path = f'{img_fold_path}/{files_in_img_fold[mK_proc_i]}'
            mKate_data = np.load(mKate_path)
        else:
            mKate_path = f'{img_fold_path}/{files_in_img_fold[mK_tif]}'
            mKate_data = io.imread(mKate_path)

        h5_df_path = f'{nucleodata_path}/{h5_df_name}'
        nuclSIZE_h5_path = f'{nucleodata_path}/{nuclSIZE_h5_name}'
        # NOTE: if you are working with data on Dropbox or Google Drive the h5
        # database cannot be created directly there. It is safer to create
        # it in the user folder and move it after it has been closed.
        nuclSIZE_h5_temp = f'{user_path}/Documents/{nuclSIZE_h5_name}'
        summ_df_path = f'{nucleodata_path}/{summ_df_name}'
        summ_df = pd.read_csv(summ_df_path, index_col=['frame_i', 'Cell_ID'])
        df_store = pd.HDFStore(h5_df_path, mode='r')
        keys = natsorted(df_store.keys())
        num_frames = len(keys)
        nuclSIZE_store = pd.HDFStore(nuclSIZE_h5_temp, mode='w')
        """Iterate frames"""
        for key in keys:
            df = df_store.get(key)
            frame_i = int(re.search('/frame_(\d+)', key).group(1))
            print(f'    NucleoSIZEing frame {frame_i}/{len(keys)-1} '
                  f'({pfn} ({p+1}/{pend-pstart}), '
                  f'exp. n. {exp_idx+1}/{num_exp})...')
            if num_frames == 1:
                V_mNeon = mNeon_data.copy()
            else:
                V_mNeon = mNeon_data[frame_i]
            if num_frames == 1:
                V_mKate = mKate_data.copy()
            else:
                V_mKate = mKate_data[frame_i]
            if not mNeon_proc_found:
                V_mNeon = gaussian(V_mNeon,sigma,preserve_range=True)
            if not mKate_proc_found:
                V_mKate = gaussian(V_mKate, sigma, preserve_range=True)
            summ_df_frame = summ_df.loc[frame_i]
            df_ID_li = []
            # print(summ_df_frame)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(V_mNeon.max(axis=0))
            # ax[1].imshow(V_mKate.max(axis=0))
            # plt.show()
            # exit()
            ID_groups = df.groupby('Cell_ID')
            # print(df)
            """Iterate cells"""
            for ID_idx, (ID, df_ID) in enumerate(ID_groups):
                print(f'        NucleoSIZEing cell ID {ID}/{len(ID_groups)} '
                      f'(frame {frame_i}/{len(keys)-1}, '
                      f'{pfn} ({p+1}/{pend-pstart}), '
                      f'exp. n. {exp_idx+1}/{num_exp})...')
                df_ID_copy = df_ID.copy()
                zyx_centers = df_ID[['z','y','x']].to_numpy()
                i = 0  # nucleoid index in h5 dataframe
                sph = spheroid(V_mNeon)
                semiax_len = sph.calc_semiax_len(i, zyx_vox_dim, zyx_resolution)
                # print(semiax_len)
                summ_df_ID = summ_df_frame.loc[ID]
                min_int = summ_df_ID['mNeon norm. value']
                mKate_norm = summ_df_ID['mKate norm. value']
                print('          Background mNeon intensity '
                      f'(cell ID {ID}) = {min_int:.2f}')
                print('          Background mKate intensity '
                      f'(cell ID {ID}) = {mKate_norm:.2f}')
                grow_prev = [True]*len(zyx_centers)
                grow = sph.grow_cond(i, semiax_len, zyx_centers, grow_prev,
                                     V_mNeon, min_int)
                sph_xy_sizes = [sph.xys if not g1 else 0 for g1 in grow]
                sph_z_sizes = [sph.zs if not g1 else 0 for g1 in grow]
                sph_volumes = [sph.volume() if not g1 else 0 for g1 in grow]
                foregr_sum = [sph.calc_foregr_sum(j, V_mNeon, min_int)
                                            if not g1 else 0
                                            for j, g1 in enumerate(grow)]
                mNeon_mKate_sum = [sph.calc_mNeon_mKate_sum(j, V_mNeon,
                                            V_mKate, min_int, mKate_norm)
                                            if not g1 else 0
                                            for j, g1 in enumerate(grow)]
                sph_sizes_pxl = [semiax_len if not g1 else (0,0)
                                            for g1 in grow]
                while any(grow):
                    i += 1
                    semiax_len = sph.calc_semiax_len(i, zyx_vox_dim,
                                                     zyx_resolution)
                    grow_next = sph.grow_cond(i, semiax_len, zyx_centers, grow,
                                              V_mNeon, min_int, verb=False)
                    # check if we should stop growing in some positions
                    idx = [i for i, (g1, g2) in enumerate(zip(grow, grow_next))
                                                                   if g1 != g2]
                    # Store the semiaxis lengths for positions where
                    # we stopped growing
                    if idx:
                        for j in idx:
                            sph_xy_sizes[j] = sph.xys
                            sph_z_sizes[j] = sph.zs
                            sph_volumes[j] = sph.volume()
                            foregr_sum[j] = sph.calc_foregr_sum(j, V_mNeon,
                                                                   min_int)
                            mNeon_mKate_sum[j] = sph.calc_mNeon_mKate_sum(j,
                                                          V_mNeon, V_mKate,
                                                          min_int, mKate_norm)
                            sph_sizes_pxl[j] = semiax_len
                    grow = grow_next.copy()
                col_loc = len(df_ID_copy.columns)-4
                df_ID_copy.insert(col_loc, 'xy radius (µm)', sph_xy_sizes)
                df_ID_copy.insert(col_loc, 'z radius (µm)', sph_z_sizes)
                df_ID_copy['Nucl. vol (µm^3)'] = sph_volumes
                df_ID_copy['Foregr. sum int'] = foregr_sum
                df_ID_copy['mNeon-mKate sum int'] = mNeon_mKate_sum
                print(df_ID_copy)
                df_ID_li.append(df_ID_copy)
            df_frame_nuclSIZE = pd.concat(df_ID_li)
            print(df_frame_nuclSIZE)
            bp.pausehere()
            nuclSIZE_store.append(key, df_frame_nuclSIZE)
            # """Test plots"""
            # V_segm = np.zeros(V_mNeon.shape, int)
            # ID = 1
            # for (z, yc, xc), (radius, _) in zip(zyx_centers, sph_sizes_pxl):
            #     rr, cc = circle(yc, xc, radius, shape=(V_mNeon.shape[1], V_mNeon.shape[2]))
            #     V_segm[ref_z, rr, cc] = ID
            #     ID += 1
            # fig, ax = plt.subplots(1,2)
            #
            # ax[0].imshow(z_proj_max(V_mNeon))
            # ax[0].plot(zyx_centers[:,2], zyx_centers[:,1], 'r.')
            # ax[1].imshow(V_segm[ref_z])
            # plt.win_size(swap_screen=True)
            # plt.show()
            # exit()
        df_store.close()
        nuclSIZE_store.close()
        shutil.move(nuclSIZE_h5_temp, nuclSIZE_h5_path)

    tf = time()

    print(f'Experiment n. {exp_idx+1}/{num_exp}: process finished! '
          f'Total execution time {tf-t0:.2f} s')
