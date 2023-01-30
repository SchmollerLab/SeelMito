import os
import sys
import traceback
import re
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from skimage import io
import skimage.filters
from datetime import datetime
from tifffile import TiffFile
from natsort import natsorted
import skimage.measure
import apps, prompts

class load_positive_control:
    def __init__(self):
        self.is_first_call = True
        self.V_local_spots_PC = None
        self.local_mask_PC_3D = None

    def load_PC_df(self, TIFFs_path):
        self.PC_df = None
        listdir_TIFFs = os.listdir(TIFFs_path)
        files = [f for f in listdir_TIFFs
                 if f == 'Positive_control_analysis_info.csv']
        if files:
            load = prompts.askyesno(title='Load pos. control?',
            message='It looks like you performed a positive control detection.\n'
                    'Do you want to load that information?')
            if load:
                self.PC_df = pd.read_csv(os.path.join(TIFFs_path, files[0])
                             ).set_index('PC_IDs'
                             ).sort_values('areas', ascending=False)
        self.is_first_call = False

    def load_single_PC(self, TIFFs_path, user_ch_name, sigma):
        PC_ID = self.PC_df.index[0]
        pos = self.PC_df.iloc[0]['Position_n']
        images_path = os.path.join(TIFFs_path, pos, 'Images')
        for f in os.listdir(images_path):
            if f.endswith(f'{user_ch_name}.tif'):
                V_spots_PC = io.imread(os.path.join(images_path, f))
                print('')
                print(f'Positive control: {pos}, ID {PC_ID}')
            elif f.endswith('segm.npy'):
                PC_lab = np.load(os.path.join(images_path, f))
        PC_rp = skimage.measure.regionprops(PC_lab)
        IDs = [obj.label for obj in PC_rp]
        idx = IDs.index(PC_ID)
        PC_obj = PC_rp[idx]
        V_local_spots_PC = V_spots_PC[:, PC_obj.slice[0], PC_obj.slice[1]]
        if sigma > 0:
            V_local_spots_PC = skimage.filters.gaussian(V_local_spots_PC, sigma)
        self.V_local_spots_PC = V_local_spots_PC
        self.local_mask_PC_3D = PC_obj.image
        if self.local_mask_PC_3D.ndim == 2:
            _tile = [self.local_mask_PC_3D]*len(V_spots_PC)
            self.local_mask_PC_3D = np.array(_tile)




class load_data:
    """
    Load tif file into numpy array and read file metadata:

    - zyx_vox_dim from pixel depth, width, height
    - Emission wavelength (wavelen) of channel name
    - All channels' name and number (fluor_names)

    """
    def __init__(self, path, channel_name=None, user_ch_name=None,
                 load_shifts=False, load_segm_npy=False, load_metadata=True,
                 create_data_folder=False, load_cca_df=False,
                 load_bkgr_mask=False,
                 data_foldername='spotMAX_output', load_analysis_inputs=False,
                 run_num=1, vNUM='v1', load_df_h5=False,
                 which_h5='_spotFIT_', ch_name_selector=None,
                 which_ch='Spots', load_ch_mask=False,
                 load_summary_df=False, ask_metadata_not_found=True,
                 which_summmary_df='3_p-_ellip_test_data_Summary'):
        if channel_name is None:
            self.pos_path = path
            self.images_path = os.path.join(self.pos_path, 'Images')
        else:
            self.images_path = os.path.dirname(path)
            self.pos_path = os.path.dirname(self.images_path)

        if ch_name_selector is None:
            ch_name_selector = prompts.select_channel_name()

        # If the channel name is not provided attempt to read it from
        # analysis inputs.csv otherwise promp user to select channel name
        prompt_channel_name = False
        if load_analysis_inputs or channel_name is None:
            data_path = f'{self.pos_path}/{data_foldername}'
            nucleodata_path = f'{self.pos_path}/NucleoData'
            if os.path.exists(data_path) or os.path.exists(nucleodata_path):
                if os.path.exists(nucleodata_path):
                    os.rename(nucleodata_path, data_path)
                analysis_inputs_path = os.path.join(
                        data_path, f'{run_num}_{vNUM}_analysis_inputs.csv'
                )
                if os.path.exists(analysis_inputs_path):
                    df_inputs = pd.read_csv(analysis_inputs_path)
                    df_inputs['Description'] = (
                        df_inputs['Description'].str.replace('\u03bc', 'u')
                    )
                    df_inputs.set_index('Description', inplace=True)
                    self.df_inputs = df_inputs
                    if which_ch == 'Spots':
                        filename_idx = 'Spots file name:'
                        ch_name_idx = 'Spots channel name:'
                    elif which_ch == 'Reference':
                        filename_idx = 'Reference ch. file name:'
                        ch_name_idx = 'Reference channel name:'
                    if filename_idx in df_inputs.columns:
                        v = 'Values'
                        user_ch_name = df_inputs.at[filename_idx, v]
                        channel_name = df_inputs.at[ch_name_idx, v]
                    else:
                        prompt_channel_name = True
                else:
                    raise FileNotFoundError(f'Analysis inputs file not found. '
                    f'Path searched: {analysis_inputs_path}')
            else:
                if load_analysis_inputs:
                    raise FileNotFoundError(f'Data folder {data_path} not found!')
                else:
                    prompt_channel_name = True
        if prompt_channel_name and ch_name_selector.is_first_call:
            filenames = os.listdir(self.images_path)
            ch_names = ch_name_selector.get_available_channels(filenames)
            ch_name_selector.prompt(ch_names)
            channel_name = ch_name_selector.channel_name
            user_ch_name = ch_name_selector.channel_name
            ch_name_selector.user_ch_name = user_ch_name
            path = self._search_user_ch_name_file(user_ch_name)
            # Keep ch_name_selector usable if the channel name is not found
            ch_name_selector.is_first_call = True
        elif prompt_channel_name and not ch_name_selector.is_first_call:
            user_ch_name = ch_name_selector.user_ch_name
            channel_name = ch_name_selector.metadata_ch_name
            path = self._search_user_ch_name_file(user_ch_name)
        self.user_ch_name = user_ch_name
        self.filename = os.path.basename(path)

        npy_selected = self.filename.find('.npy') != -1
        data_path = f'{self.pos_path}/{data_foldername}'
        self.data_path = data_path
        if npy_selected:  # .npy selected --> we need to read metadata from tif
            tif_path = self.get_substring_file(path, user_ch_name+'.tif',
                                               self.images_path)[0]
            self.tif_path = tif_path
            frames = np.load(path)
        else:
            self.tif_path = path
            frames = io.imread(path)
        slice_path, slice_found = self.get_substring_file(path, 'slice_segm.csv',
                                                          self.images_path)
        if slice_found:
            df_slices = pd.read_csv(slice_path)
            self.slices = df_slices['Slice used for segmentation'].to_list()
        else:
            slice_path, slice_found = self.get_substring_file(path,
                                                              'slice_segm.txt',
                                                              self.images_path)
            if slice_found:
                with open(slice_path, 'r') as slice_txt:
                    slice = slice_txt.read()
                    slice = int(slice)
                self.slices = [slice]
            else:
                pass
                # print('WARNING: slice used for segmentation text file not found!')
        tif_filename = os.path.basename(self.tif_path)
        basename_idx = tif_filename.find(f'_{user_ch_name}.tif')
        self.basename = tif_filename[0:basename_idx]
        self.frames = frames
        if self.frames.ndim == 2:
            self.frames = None
            print('')
            print('==========================================')
            print('')
            print(f'File {self.tif_path} is 2D.\n2D images are not supported yet. '
            'Please open an ticket on GitHub for a feature request.')
            return
        self.path = path
        self.already_aligned = self.aligned(path)
        if channel_name == 'mNeon':
            self.channel_name = 'EGFP'
        elif channel_name == 'mCitrine':
            self.channel_name = 'TagYFP'
        else:
            self.channel_name = channel_name

        self.zyx_vox_dim_found = False
        self.wavelen_found = False
        self.NA_found = False
        if load_metadata:
            self.info, self.metadata_found = self.metadata(self.tif_path)
            if self.metadata_found:
                try:
                    self.SizeT, self.SizeZ = self.data_dimensions(self.info)
                except Exception as e:
                    traceback.print_exc()
                    print('IGNORE error')
                    if ask_metadata_not_found:
                        self.SizeT, self.SizeZ = self.dimensions_entry_widget()
                    else:
                        self.SizeT, self.SizeZ = 1, 1
            else:
                self.SizeT, self.SizeZ = 1, 1
            self.num_segm_frames = self.SizeT
            prompt_user = False
            self.timestamp = datetime.now()
            if self.metadata_found and not load_analysis_inputs:
                try:
                    self.zyx_vox_dim = self.zyx_vox_dim()
                    self.zyx_vox_dim_txt = ', '.join(str((round(e, 4)))
                                                     for e in self.zyx_vox_dim)
                    self.zyx_vox_dim_found = True
                except Exception as e:
                    if sys.exc_info()[0] == SystemExit:
                        exit('Execution aborted.')
                    # traceback.print_exc()
                    prompt_user = True
                    self.zyx_vox_dim_txt = None
                    self.zyx_vox_dim = None
                try:
                    (self.fluor_names,
                    self.ch_num) = self.fluor_names(ch_name_selector)
                    self.wavelen = self.emission_wavelengths(self.ch_num)
                    self.em_wavel_txt = f'{self.wavelen}'
                    self.wavelen_found = True
                except Exception as e:
                    if sys.exc_info()[0] == SystemExit:
                        exit('Execution aborted.')
                    traceback.print_exc()
                    prompt_user = True
                    self.em_wavel_txt = None
                    self.wavelen = None
                try:
                    self.NA = self.numerical_aperture(self.info)
                    self.NA_txt = f'{self.NA}'
                    self.NA_found = True
                except Exception as e:
                    if sys.exc_info()[0] == SystemExit:
                        exit('Execution aborted.')
                    traceback.print_exc()
                    prompt_user = True
                    self.NA_txt = None
                    self.NA = None
                try:
                    self.param_manual_entry = False
                    self.timestamp = self.get_timestamp(self.info)
                except Exception as e:
                    if sys.exc_info()[0] == SystemExit:
                        exit('Execution aborted.')
                    traceback.print_exc()
                    self.timestamp = datetime.now()
                if prompt_user and ask_metadata_not_found:
                    self.metadata_entry_widget(
                            zyx_vox_dim_txt=self.zyx_vox_dim_txt,
                            em_wavel_txt=self.em_wavel_txt,
                            NA_txt=self.NA_txt)
                    self.param_manual_entry = True
                else:
                    self.NA = self.NA
                    self.wavelen = self.wavelen
                    self.zyx_vox_dim = self.zyx_vox_dim
                    self.param_manual_entry = False
            elif not load_analysis_inputs:
                if prompt_user and ask_metadata_not_found:
                    self.metadata_entry_widget()
                    self.param_manual_entry = True
                else:
                    self.NA = None
                    self.wavelen = None
                    self.zyx_vox_dim = None
                    self.param_manual_entry = False
            else:
                ch_name_selector.is_first_call = False
                ch_name_selector.metadata_ch_name = channel_name

        else:
            ch_name_selector.is_first_call = False
            ch_name_selector.metadata_ch_name = channel_name
            self.param_manual_entry = False
            self.metadata_found = False
        if load_shifts:
            if self.already_aligned:
                # print('WARNING: Frames are already aligned, no need to load shifts')
                self.skip_alignment = False
            else:
                shift_npy_path, shifts_npy_found = self.get_substring_file(path,
                                                              'align_shift.npy',
                                                              self.images_path)
                if shifts_npy_found:
                    self.shifts = np.load(shift_npy_path)
                    self.skip_alignment = False
                else:
                    # print('WARNING: Shifts file for alignment not found. '
                    #       'No alignment can be applied!')
                    self.already_aligned = True
                    self.skip_alignment = True
        self.segm_npy = None
        self.is_segm_3D = False
        if load_segm_npy:
            segm_npz_path, segm_npy_found = self.get_substring_file(
                                           path, 'segm.npz', self.images_path)
            if not segm_npy_found:
                segm_npz_path, segm_npy_found = self.get_substring_file(
                                           path, 'segm.npy', self.images_path)
            if segm_npy_found:
                segm_npy = np.load(segm_npz_path)
                try:
                    segm_npy = segm_npy['arr_0']
                    print('npz found')
                except Exception as e:
                    segm_npy = segm_npy
                if segm_npy.ndim == 2 or self.frames.ndim == 3:
                    num_segm_frames = 1
                else:
                    num_segm_frames = len(segm_npy)
                self.segm_npy = segm_npy
                self.num_segm_frames = num_segm_frames
                shape_match = False
                if self.frames.ndim == 4:
                    if segm_npy.ndim == 4 or segm_npy.ndim == 3:
                        shape_match = True
                        if segm_npy.ndim == 4:
                            self.is_segm_3D = True
                elif self.frames.ndim == 3:
                    if segm_npy.ndim == 3 or segm_npy.ndim == 2:
                        shape_match = True
                        if segm_npy.ndim == 3:
                            self.is_segm_3D = True
                if not shape_match:
                    raise IndexError('Shape mismatch!'
                    ' The intensity image has shape '
                    f'{self.frames.shape}, but the segmentation file has shape '
                    f'{self.segm_npy.shape}.\n'
                    'The order of the dimensions in the files MUST be TZYX '
                    'where T (number of frames) will be set '
                    'to 1 if you do not have frames.\n'
                    'In general:\n'
                    '- Intensity image 4D (i.e. 3D over time) --> segmentation '
                    'can be either 3D (i.e. 2D over time) or 4D (i.e. 3D over time)\n'
                    '- Intensity image 3D (i.e. single 3D, no time) --> '
                    'segmentation can be either 3D (i.e. single 3D, no time) '
                    'or 2D (i.e. single 2D, no time)\n')
            else:
                print('WARNING: Segmentation file not found. '
                'Sub-cellular features will be assigned to a single object '
                'with ID = 1.')
        self.cca_df = None
        if load_cca_df:
            cca_df_path, cca_df_found = self.get_substring_file(path,
                                                                'cc_stage.csv',
                                                               self.images_path)
            if cca_df_found:
                self.cca_df = pd.read_csv(cca_df_path, index_col='Cell_ID')
            else:
                cca_df_path, cca_df_found = self.get_substring_file(
                                    path, 'acdc_output.csv', self.images_path)
                if cca_df_found:
                    acdc_df = pd.read_csv(cca_df_path)
                    self.cca_df = self.acdc_df_To_cca_df(acdc_df)
                else:
                    print('WARNING: Cell cycle analysis file not found. '
                    'Cell cycle information (e.g. cell cycle stage) cannot be '
                    'assigned. Ignore you do not need cell cycle info.')
        if load_bkgr_mask:
            mask_path, mask_found = self.get_substring_file(path,
                                                            '_mask.npy',
                                                            self.images_path)
            if mask_found:
                self.mask_npy = np.load(mask_path)
            else:
                self.mask_npy = None
        if create_data_folder:
            if not os.path.exists(data_path):
                os.mkdir(data_path)
        if load_df_h5:
            # (.*) means find any char (except new line) zero or more times
            pattern = f'{run_num}(.*){which_h5}(.*){vNUM}.h5'
            parent_path = os.path.join(self.pos_path, data_foldername)
            df_h5_path = self.search_by_regex(data_path, pattern)
            self.store_HDF = pd.HDFStore(df_h5_path, mode='r')
        if load_ch_mask:
            ch_mask_filenames = ['_mKate_thresh.npy', '_ref_ch_maks.npy',
                                 f'_{user_ch_name}_mask.npy']
            for f in ch_mask_filenames:
                ch_mask_path, ch_mask_found = self.get_substring_file(
                                                            path, f,
                                                            self.images_path)
                if ch_mask_found:
                    break
            if ch_mask_found:
                self.ch_mask = np.load(ch_mask_path)
            else:
                self.ch_mask = None
        if load_summary_df:
            fn = f'{run_num}_{which_summmary_df}_{vNUM}.csv'
            csv_path = os.path.join(self.data_path, fn)
            if os.path.exists(csv_path):
                self.summ_df_path = csv_path
                self.summary_df = (pd.read_csv(csv_path)
                                   .set_index(['frame_i', 'Cell_ID']))
            else:
                raise FileNotFoundError('Summary df not found! '
                                       f'Path requested: {csv_path}')

    def acdc_df_To_cca_df(self, acdc_df):
        if 'cell_cycle_stage' in acdc_df.columns:
            cca_df_colNames = [
                'Cell_ID',
                'cell_cycle_stage',
                'generation_num',
                'relative_ID',
                'relationship'
            ]
            cca_df = acdc_df[cca_df_colNames]
            cca_df = cca_df.rename(
                columns={
                    'cell_cycle_stage': 'Cell cycle stage',
                    'generation_num': '# of cycles',
                    'relative_ID': "Relative's ID",
                    'relationship': 'Relationship'
                }
            ).set_index('Cell_ID')
            cca_df['OF'] = False
            return cca_df
        else:
            print('WARNING: Cell cycle analysis file not found. '
            'Cell cycle information (e.g. cell cycle stage) cannot be '
            'assigned. Ignore you do not need cell cycle info.')
            return None


    def _search_user_ch_name_file(self, user_ch_name):
        filenames = os.listdir(self.images_path)
        ch_aligned_found = False
        for j, f in enumerate(filenames):
            if f.find(f'_{user_ch_name}_aligned.npy') != -1:
                ch_aligned_found = True
                aligned_i = j
            elif f.find(f'_{user_ch_name}.tif') != -1:
                tif_i = j
        if ch_aligned_found:
            ch_path = os.path.join(self.images_path, filenames[aligned_i])
        else:
            ch_path = os.path.join(self.images_path, filenames[tif_i])
        return ch_path

    def search_by_regex(self, parent_path, pattern):
        filenames = os.listdir(parent_path)
        re_path = None
        for file in filenames:
            if re.findall(pattern, file):
                re_path = os.path.join(parent_path, file)
                break
        return re_path


    def data_dimensions(self, info):
        SizeT = int(re.findall('SizeT = (\d+)', info)[0])
        SizeZ = int(re.findall('SizeZ = (\d+)', info)[0])
        return SizeT, SizeZ


    def remove_salt_pepper_noise(self):
        # see https://scikit-image.org/docs/dev/auto_examples/applications/plot_3d_image_processing.html#sphx-glr-auto-examples-applications-plot-3d-image-processing-py
        vmin, vmax = np.percentile(self.frames, q=(0.5, 99.9999))
        self.frames = exposure.rescale_intensity(
                            self.frames,
                            in_range=(vmin, vmax),
                            out_range=np.float32)

    def dimensions_entry_widget(self):
        root = tk.Tk()
        root.geometry("+800+400")
        tk.Label(root,
                 text="Data dimensions not found in metadata.\n"
                      "Provide the following sizes.",
                 font=(None, 12)).grid(row=0, column=0, columnspan=2, pady=4)
        tk.Label(root,
                 text="Number of frames (SizeT)",
                 font=(None, 10)).grid(row=1, pady=4)
        tk.Label(root,
                 text="Number of slices (SizeZ)",
                 font=(None, 10)).grid(row=2, pady=4, padx=8)

        SizeT_entry = tk.Entry(root, justify='center')
        SizeZ_entry = tk.Entry(root, justify='center')

        # Default texts in entry text box
        SizeT_entry.insert(0, '1')
        SizeZ_entry.insert(0, '1')

        SizeT_entry.grid(row=1, column=1, padx=8)
        SizeZ_entry.grid(row=2, column=1, padx=8)

        tk.Button(root,
                  text='OK',
                  command=root.quit,
                  width=10).grid(row=3,
                                 column=0,
                                 pady=16,
                                 columnspan=2)
        SizeT_entry.focus()

        tk.mainloop()

        SizeT = int(SizeT_entry.get())
        SizeZ = int(SizeZ_entry.get())
        root.destroy()
        return SizeT, SizeZ

    def aligned(self,path):
        filename = os.path.basename(path)
        filename_noEXT, filename_extension = os.path.splitext(filename)
        already_aligned = True if filename_extension == '.npy' else False
        return already_aligned


    def metadata(self, tif_path):
        with TiffFile(tif_path) as tif:
            metadata = tif.imagej_metadata
        try:
            self.finterval = metadata['finterval']
        except Exception as e:
            self.finterval = 0
        try:
            self.num_frames = metadata['frames']
        except Exception as e:
            self.num_frames = 1
        try:
            self.num_slices = metadata['slices']
        except Exception as e:
            self.num_slices = 0
        try:
            metadata_found = True
            metadata = metadata['Info']
        except KeyError:
            metadata_found = False
            metadata = []
        return metadata, metadata_found


    def zyx_vox_dim(self):
        info = self.info
        try:
            scalint_str = "Scaling|Distance|Value #"
            len_scalin_str = len(scalint_str) + len("1 = ")
            px_x_start_i = info.find(scalint_str + "1 = ") + len_scalin_str
            px_x_end_i = info[px_x_start_i:].find("\n") + px_x_start_i
            px_x = float(info[px_x_start_i:px_x_end_i])*1E6 #convert m to µm
            px_y_start_i = info.find(scalint_str + "2 = ") + len_scalin_str
            px_y_end_i = info[px_y_start_i:].find("\n") + px_y_start_i
            px_y = float(info[px_y_start_i:px_y_end_i])*1E6
            px_z_start_i = info.find(scalint_str + "3 = ") + len_scalin_str
            px_z_end_i = info[px_z_start_i:].find("\n") + px_z_start_i
            px_z = float(info[px_z_start_i:px_z_end_i])*1E6
        except Exception as e:
            x_res_match = re.findall('XResolution = ([0-9]*[.]?[0-9]+)', info)
            px_x = 1/float(x_res_match[0])
            y_res_match = re.findall('YResolution = ([0-9]*[.]?[0-9]+)', info)
            px_y = 1/float(y_res_match[0])
            z_spac_match = re.findall('Spacing = ([0-9]*[.]?[0-9]+)', info)
            px_z = float(z_spac_match[0])
        return [px_z, px_y, px_x]


    def scan_info_string(self, search_str, info):
        start_i = info.find(search_str)
        info_from_start = info[start_i:]
        str_found = start_i != -1
        all_info = []
        name_found = False
        num = 0
        while str_found:
            next_line_i = info_from_start.find('\n')+1
            line_info = info_from_start[len(search_str):next_line_i-1]
            if not name_found:
                num += 1
                name_found = line_info.find(self.channel_name) != -1
            info_from_start = info_from_start[next_line_i:]
            str_found = info_from_start.find(search_str) != -1
            all_info.append(line_info)
        return all_info, num, name_found


    def get_timestamp(self, info):
        start_i = info.find('Information|TimelineTrack|TimelineElement|Time')
        info = info[start_i:]
        end_i = info.find('\n')
        time_line = info[:end_i]
        start_eq = info.find('=')+2
        timestamp_str = time_line[start_eq:]
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError as v:
            if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                timestamp_str = timestamp_str[:-(len(v.args[0]) - 26)]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
            else:
                timestamp = datetime.now()
        # timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%uZ')
        # Alternatively
        # timestamp = datetime.fromisoformat(re.sub('[0-9]Z', '+00:00', timestamp_str))
        return timestamp


    def fluor_names(self, ch_name_selector):
        raise_error = False
        info = self.info
        search_str = "Information|Image|Channel|Fluor #"
        ch_num = None
        all_fluor_info, _ch_num, ch_name_found = self.scan_info_string(
                                                            search_str, info)
        if not ch_name_found:
            if all_fluor_info:
                msg = (f'Channel name {self.user_ch_name} not found in '
                      'file information.\n These are the fluorophore names found '
                      f'in file metadata information:')
                if ch_name_selector is not None:
                    if ch_name_selector.is_first_call:
                        channel_names = [re.findall('(\d+) = (.*)', ch)[0][-1]
                                         for ch in all_fluor_info]
                        ch_name_selector.prompt(channel_names, message=msg)
                    self.channel_name = ch_name_selector.channel_name
                    ch_name_selector.metadata_ch_name = self.channel_name
                    (all_fluor_info, _ch_num,
                    ch_name_found) = self.scan_info_string(search_str, info)
                    if not ch_name_found:
                        raise_error = True
                else:
                    raise_error = True
            else:
                raise_error = True
        else:
            ch_num = _ch_num
            ch_name_selector.is_first_call = False
        if raise_error:
            raise NameError(f'Channel name {self.user_ch_name} not found in '
                  'file information. These are the fluorophores name found '
                  f'in file metadata information: {all_fluor_info}')
        return all_fluor_info, ch_num


    def emission_wavelengths(self, ch_num_to_search):
        info = self.info
        search_str = "Information|Image|Channel|EmissionWavelength #"
        all_wavelen, ch_num, ch_name_found = self.scan_info_string(search_str,
                                                                   info)
        for wavelen_info in all_wavelen:
            ch_num_found = wavelen_info.find(str(ch_num_to_search)) != -1
            if ch_num_found:
                break
        m = re.search('= (\d+)',wavelen_info)
        ch_em_wavelen = m.group(1)
        return float(ch_em_wavelen)


    def numerical_aperture(self, info):
        search_str = "Information|Instrument|Objective|LensNA #"
        string = self.scan_info_string(search_str, info)[0][0]
        m = re.search('= (\d+\.\d+)',string)
        NA = m.group(1)
        return float(NA)


    def get_substring_file(self, path, substring, parent_path):
        substring_found = False
        for filename in os.listdir(parent_path):
            if filename.find(substring)>0:
                substring_found = True
                break
        substring_path = os.path.join(parent_path, filename)
        return substring_path, substring_found

    def metadata_entry_widget(self, zyx_vox_dim_txt=None, em_wavel_txt=None,
                                    NA_txt=None):
        root = tk.Tk()
        root.geometry("+800+400")
        tk.Label(root,
                 text="Metadata was not found.\n"
                      "Provide the following constants.",
                 font=(None, 12)).grid(row=0, column=0,
                                       columnspan=2, pady=4)
        tk.Label(root,
                 text="z, y, x pixel width (µm)",
                 font=(None, 10)).grid(row=1, pady=4)
        tk.Label(root,
                 text="Emission wavelength (nm)",
                 font=(None, 10)).grid(row=2, pady=4)
        tk.Label(root,
                 text="Numerical aperture",
                 font=(None, 10)).grid(row=3, pady=4)

        zyx_vox_dim_entry = tk.Entry(root, justify='center', width=30)
        emission_wavelength_entry = tk.Entry(root, justify='center', width=30)
        NA_entry = tk.Entry(root, justify='center', width=30)
        self.zyx_vox_dim_entry = zyx_vox_dim_entry
        self.emission_wavelength_entry = emission_wavelength_entry
        self.NA_entry = NA_entry

        # Default texts in entry text box
        if zyx_vox_dim_txt is None:
            zyx_vox_dim_txt = '0.125, 0.08, 0.08'
        zyx_vox_dim_entry.insert(0, zyx_vox_dim_txt)
        if em_wavel_txt is None:
            em_wavel_txt = '509'
        emission_wavelength_entry.insert(0, '509')
        if NA_txt is None:
            NA_txt = '1.3'
        NA_entry.insert(0, '1.3')

        zyx_vox_dim_entry.grid(row=1, column=1, padx=(4,10))
        emission_wavelength_entry.grid(row=2, column=1, padx=(4,10))
        NA_entry.grid(row=3, column=1, padx=(4,10))

        tk.Button(root,
                  text='OK',
                  command=self._close,
                  width=10).grid(row=4,
                                 column=0,
                                 pady=16,
                                 columnspan=2)
        zyx_vox_dim_entry.focus()

        self.NA_entry = NA_entry
        self.emission_wavelength_entry = emission_wavelength_entry
        self.zyx_vox_dim_entry = zyx_vox_dim_entry

        root.protocol("WM_DELETE_WINDOW", self._abort)

        self.root = root

        try:
            self._load_last_status()
        except Exception as e:
            # traceback.print_exc()
            pass

        root.mainloop()

    def _load_last_status(self):
        _src_path = os.path.dirname(os.path.realpath(__file__))
        _temp_path = os.path.join(_src_path, 'temp')
        _csv_path = os.path.join(_temp_path, 'metadata_df.csv')
        _df = pd.read_csv(_csv_path).set_index('Description')
        zyx_vox_dim = _df.at['zyx_vox_dim', 'Values']
        NA = _df.at['num_aperture', 'Values']
        wavelen = _df.at['wavelen', 'Values']
        self.zyx_vox_dim_entry.delete(0, tk.END)
        self.zyx_vox_dim_entry.insert(0, zyx_vox_dim)
        self.emission_wavelength_entry.delete(0, tk.END)
        self.emission_wavelength_entry.insert(0, wavelen)
        self.NA_entry.delete(0, tk.END)
        self.NA_entry.insert(0, NA)

    def _close(self):
        self.NA = float(self.NA_entry.get())
        self.wavelen = float(self.emission_wavelength_entry.get())
        m = re.findall('([0-9]*[.]?[0-9]+)',
                       self.zyx_vox_dim_entry.get())
        self.zyx_vox_dim = [float(f) for f in m]
        _df = pd.DataFrame({
            'Description': ['zyx_vox_dim', 'num_aperture', 'wavelen'],
            'Values': [self.zyx_vox_dim, self.NA, self.wavelen]}
                            ).set_index('Description')
        _src_path = os.path.dirname(os.path.realpath(__file__))
        _temp_path = os.path.join(_src_path, 'temp')
        _csv_path = os.path.join(_temp_path, 'metadata_df.csv')
        _df.to_csv(_csv_path)
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        exit('Execution aborted by the user')
        self.root.quit()
        self.root.destroy()


class beyond_listdir_pos:
    def __init__(self, folder_path, vNUM=None, run_num=None):
        self.bp = apps.tk_breakpoint()
        self.folder_path = folder_path
        self.TIFFs_path = []
        self.count_recursions = 0
        self.listdir_recursion(folder_path)
        if not self.TIFFs_path:
            raise FileNotFoundError(f'Path {folder_path} is not valid!')
        self.all_exp_info = self.count_analysed_pos(vNUM, run_num)

    def listdir_recursion(self, folder_path):
        if os.path.isdir(folder_path):
            listdir_folder = natsorted(os.listdir(folder_path))
            contains_pos_folders = any([name.find('Position_')!=-1 for name in listdir_folder])
            if not contains_pos_folders:
                contains_TIFFs = any([name=='TIFFs' for name in listdir_folder])
                contains_CZIs = any([name=='CZIs' for name in listdir_folder])
                if contains_TIFFs:
                    self.TIFFs_path.append(f'{folder_path}/TIFFs')
                elif contains_CZIs:
                    self.TIFFs_path.append(f'{folder_path}')
                else:
                    for name in listdir_folder:
                        subfolder_path = f'{folder_path}/{name}'
                        self.listdir_recursion(subfolder_path)
            else:
                self.TIFFs_path.append(folder_path)

    def get_rel_path(self, path):
        rel_path = ''
        parent_path = path
        count = 0
        while parent_path != self.folder_path or count==10:
            if count > 0:
                rel_path = f'{os.path.basename(parent_path)}/{rel_path}'
            parent_path = os.path.dirname(parent_path)
            count += 1
        rel_path = f'.../{rel_path}'
        return rel_path

    def count_analysed_pos(self, vNUM=None, run_num=None):
        all_exp_info = []
        for path in self.TIFFs_path:
            foldername = os.path.basename(path)
            if foldername == 'TIFFs':
                pos_foldernames = os.listdir(path)
                rel_path = self.get_rel_path(path)
                num_analysed_pos = 0
                num_nuclSIZED_pos = 0
                num_pos = len(pos_foldernames)
                for pos_foldername in pos_foldernames:
                    spotmax_path = f'{path}/{pos_foldername}/spotMAX_output'
                    if not os.path.exists(spotmax_path):
                        spotmax_path = os.path.join(path, pos_foldername,
                                                    'NucleoData')
                    if os.path.exists(spotmax_path):
                        num_analysed_pos += 1
                        if vNUM is not None:
                            nuclSIZE_name = '4_p-_ellip_size_test_data'
                            if run_num is not None:
                                to_find = f'{run_num}_{nuclSIZE_name}_{vNUM}.h5'
                            else:
                                to_find = f'{nuclSIZE_name}_{vNUM}.h5'
                            filenames = os.listdir(spotmax_path)
                            spotSIZE_found = any([f.find(to_find)!=-1
                                                  for f in filenames])
                            if spotSIZE_found:
                                num_nuclSIZED_pos += 1
                if num_analysed_pos < num_pos:
                    if num_analysed_pos != 0:
                        if vNUM is None:
                            exp_info = (f'{rel_path} '
                                f'(N. of analysed pos: {num_analysed_pos})')
                        else:
                            exp_info = (f'{rel_path} '
                                f'(N. of analysed pos: {num_analysed_pos}, '
                                f'N. of NucleSIZED pos: {num_nuclSIZED_pos})')
                    else:
                        exp_info = (f'{rel_path} '
                                    '(NONE of the pos have been analysed)')
                elif num_analysed_pos == num_pos:
                    if vNUM is None:
                        exp_info = (f'{rel_path} (All pos analysed)')
                    else:
                        if num_nuclSIZED_pos == num_pos:
                            exp_info = (f'{rel_path} (All pos analysed, '
                                        f'All pos analysed NucleSIZED)')
                        else:
                            exp_info = (f'{rel_path} (All pos analysed, '
                                f'N. of NucleSIZED pos: {num_nuclSIZED_pos})')
                elif num_analysed_pos > num_pos:
                    num_analysed_pos = (f'{rel_path} (WARNING:'
                                   'multiple "spotMAX_output" folders found!)')
                else:
                    exp_info = rel_path
            else:
                rel_path = self.get_rel_path(path)
                exp_info = f'{rel_path} (FIJI macro not executed!)'
            all_exp_info.append(exp_info)
        return all_exp_info

class select_exp_folder:
    def run_widget(self, values, current=0,
                   title='Select Position folder',
                   label_txt="Select \'Position_n\' folder to analyze:",
                   showinexplorer_button=False,
                   all_button=False,
                   remaining_button=True,
                   full_paths=None,
                   rel_paths=None,
                   toplevel=False,
                   selected_path=None,
                   NONspotCOUNTED_pos=[],
                   NONspotSIZED_pos=[],
                   NONspotFIT_pos=[],
                   total_pos=1):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        width = max([len(value) for value in values])
        root.geometry('+800+400')
        root.title(title)
        root.lift()
        root.attributes("-topmost", True)
        self.root = root
        self.full_paths=full_paths
        self.rel_paths=rel_paths
        self.NONspotCOUNTED_pos = NONspotCOUNTED_pos
        self.NONspotSIZED_pos = NONspotSIZED_pos
        self.NONspotFIT_pos = NONspotFIT_pos
        self.selected_path = selected_path
        self.total_pos = total_pos
        # Label
        ttk.Label(root, text = label_txt,
                  font = (None, 10)).grid(column=0, row=0, padx=10, pady=10)

        # More info on non analysed pos button
        if selected_path is not None:
            show_NONanalysed_button = ttk.Button(self.root,
                                          text='More info',
                                          comman=self.show_NONanalysed_widget)
            show_NONanalysed_button.grid(column=4, row=1, pady=10)
            self.show_NONanalysed_button = show_NONanalysed_button
            self.show_NONanalysed_button.grid_remove()

        # Combobox
        pos_n_sv = tk.StringVar()
        if selected_path is not None:
            pos_n_sv.trace_add('write', self.combob_cb)
        self.pos_n_sv = pos_n_sv
        self.values = values
        pos_b_combob = ttk.Combobox(root, textvariable=pos_n_sv, width=width)
        pos_b_combob['values'] = values
        pos_b_combob.grid(column=1, row=0, padx=10, columnspan=4)
        pos_b_combob.current(current)

        # Ok button
        ok_b = ttk.Button(root, text='Ok!', comman=self.ok_cb)
        ok_b.grid(column=0, row=1, pady=10, sticky=tk.E)

        # Analys all button
        self.do_all = False
        if all_button:
            all_b = ttk.Button(root, text='Apply to all!',
                                          comman=self.all_cb)
            all_b.grid(column=1, row=1, pady=10)

        # Analyse remaning button
        self.do_remaining = False
        if remaining_button:
            remaining_b = ttk.Button(root, text='Analyse from selected',
                                          comman=self.remaining_cb)
            remaining_b.grid(column=2, row=1, pady=10)

        # Show in explorer button
        if showinexplorer_button:
            show_expl_button = ttk.Button(root, text='Show in explorer',
                                          comman=self.open_path_explorer)
            show_expl_button.grid(column=3, row=1, pady=10)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root = root

        root.mainloop()

    def ok_cb(self):
        idx = self.values.index(self.pos_n_sv.get())
        self.paths = [self.full_paths[idx]]
        if self.rel_paths is not None:
            self.selected_rel_paths = [self.rel_paths[idx]]
        self._close()

    def combob_cb(self, name=None, index=None, mode=None):
        sv_txt = self.pos_n_sv.get()
        sv_idx = self.values.index(sv_txt)
        rel_path_idx = len(self.selected_path)
        self.NONspotCOUNTED_paths = self.NONspotCOUNTED_pos[sv_idx]
        self.NONspotSIZED_paths = self.NONspotSIZED_pos[sv_idx]
        self.NONspotFIT_paths = self.NONspotFIT_pos[sv_idx]
        NONspotCOUNTED_li = [f'..{abs_path[rel_path_idx:]}' for abs_path
                                          in self.NONspotCOUNTED_paths]
        NONspotSIZED_li = [f'..{abs_path[rel_path_idx:]}' for abs_path
                                          in self.NONspotSIZED_paths]
        NONspotFIT_li = [f'..{abs_path[rel_path_idx:]}' for abs_path
                                          in self.NONspotFIT_paths]
        if NONspotCOUNTED_li or NONspotSIZED_li or NONspotFIT_li:
            self.NONspotCOUNTED_items = NONspotCOUNTED_li
            self.NONspotSIZED_items = NONspotSIZED_li
            self.NONspotFIT_items = NONspotFIT_li
            self.show_NONanalysed_button.grid()
        else:
            self.show_NONanalysed_button.grid_remove()

    def show_NONanalysed_widget(self):
        more_info_root = tk.Toplevel(self.root)
        more_info_root.attributes("-topmost", True)
        more_info_root.focus_force()

        #Non spotCOUNTED positions listbox
        if self.NONspotCOUNTED_items:
            font = (None, 10,'bold')
        else:
            font = (None, 10,'normal')
        tk.Label(more_info_root,
                 text='NON-spotCOUNTED Positions:', font=font
                 ).grid(row=0, column=0, pady=(10, 0))
        NONspotCOUNTED_var = tk.StringVar()
        NONspotCOUNTED_var.set(self.NONspotCOUNTED_items)

        w = 30
        if self.NONspotCOUNTED_items:
            w = len(self.NONspotCOUNTED_items[0])+10
        NONspotCOUNTED_listbox = tk.Listbox(
                                        more_info_root,
                                        listvariable=NONspotCOUNTED_var,
                                        width=w,
                                        selectmode='extended',
                                        exportselection=0)
        NONspotCOUNTED_listbox.grid(column=0, row=1, pady=(4,10), padx=(10, 0))
        NnC_scrollbar = tk.Scrollbar(more_info_root)
        NnC_scrollbar.grid(column=1, row=1, pady=4, padx=(0, 10), sticky=tk.NS)
        NONspotCOUNTED_listbox.config(yscrollcommand = NnC_scrollbar.set)
        NnC_scrollbar.config(command = NONspotCOUNTED_listbox.yview)
        self.NONspotCOUNTED_listbox = NONspotCOUNTED_listbox
        ttk.Button(more_info_root,
                  text='Analyse selected',
                  command=self.analysed_selected_NONspotCOUNTED_cb,
                  width=20).grid(row=2, column=0, pady=(0,10))

        #Non spotSIZED positions listbox
        if self.NONspotSIZED_items:
            font = (None, 10,'bold')
        else:
            font = (None, 10,'normal')
        tk.Label(more_info_root,
                 text='NON-spotSIZED Positions:', font=font
                 ).grid(row=0, column=2, pady=(10, 0))
        NONspotSIZED_var = tk.StringVar()
        NONspotSIZED_var.set(self.NONspotSIZED_items)

        w = 30
        if self.NONspotSIZED_items:
            w = len(self.NONspotSIZED_items[0])+10
        NONspotSIZED_listbox = tk.Listbox(
                                        more_info_root,
                                        listvariable=NONspotSIZED_var,
                                        width=w,
                                        selectmode='extended',
                                        exportselection=0)
        NONspotSIZED_listbox.grid(column=2, row=1, pady=(4,10), padx=(10, 0))
        NnS_scrollbar = tk.Scrollbar(more_info_root)
        NnS_scrollbar.grid(column=3, row=1, pady=4, padx=(0, 10), sticky=tk.NS)
        NONspotSIZED_listbox.config(yscrollcommand = NnS_scrollbar.set)
        NnS_scrollbar.config(command = NONspotSIZED_listbox.yview)
        self.NONspotSIZED_listbox = NONspotSIZED_listbox
        ttk.Button(more_info_root,
                  text='Analyse selected',
                  command=self.analysed_selected_NONspotSIZED_cb,
                  width=20).grid(row=2, column=2, pady=(0,10))

        #Non nucleoGAUSS-fit positions listbox
        if self.NONspotFIT_items:
            font = (None, 10,'bold')
        else:
            font = (None, 10,'normal')
        tk.Label(more_info_root,
                 text='NON-NucleoGAUSS-fit Positions:', font=font
                 ).grid(row=0, column=4, pady=(10, 0))
        NONspotFIT_var = tk.StringVar()
        NONspotFIT_var.set(self.NONspotFIT_items)

        w = 30
        if self.NONspotFIT_items:
            w = len(self.NONspotFIT_items[-1])+10
        NONspotFIT_listbox = tk.Listbox(
                                        more_info_root,
                                        listvariable=NONspotFIT_var,
                                        width=w,
                                        selectmode='extended',
                                        exportselection=0)
        NONspotFIT_listbox.grid(column=4, row=1, pady=(4,10), padx=(10, 0))
        NnGf_scrollbar = tk.Scrollbar(more_info_root)
        NnGf_scrollbar.grid(column=5, row=1, pady=4, padx=(0, 10), sticky=tk.NS)
        NONspotFIT_listbox.config(yscrollcommand = NnGf_scrollbar.set)
        NnGf_scrollbar.config(command = NONspotFIT_listbox.yview)
        self.NONspotFIT_listbox = NONspotFIT_listbox
        ttk.Button(more_info_root,
                  text='Show in Explorer',
                  command=self.show_NONspotFIT_in_explorer_cb,
                  width=20).grid(row=2, column=4, pady=(0,10))
        ttk.Button(more_info_root,
                  text='Analyse selected',
                  command=self.analysed_selected_NONspotFIT_cb,
                  width=20).grid(row=3, column=4, pady=(0,10))

        more_info_root.bind('<Escape>', self.clear_selection)

    def show_NONspotFIT_in_explorer_cb(self):
        idx = self.NONspotFIT_listbox.curselection()[-1]
        path = self.NONspotFIT_paths[idx]
        subprocess.Popen(f'explorer "{os.path.normpath(path)}"')

    def analysed_selected_NONspotCOUNTED_cb(self):
        sel_idx = self.NONspotCOUNTED_listbox.curselection()
        self.paths = [[os.path.dirname(self.NONspotCOUNTED_paths[idx])
                       for idx in sel_idx]]
        self._close()

    def analysed_selected_NONspotSIZED_cb(self):
        sel_idx = self.NONspotSIZED_listbox.curselection()
        self.paths = [[os.path.dirname(self.NONspotSIZED_paths[idx])
                       for idx in sel_idx]]
        self._close()

    def analysed_selected_NONspotFIT_cb(self):
        sel_idx = self.NONspotFIT_listbox.curselection()
        self.paths = [[os.path.dirname(self.NONspotFIT_paths[idx])
                       for idx in sel_idx]]
        self._close()

    def clear_selection(self, key_info):
        self.NONspotFIT_listbox.selection_clear(0, tk.END)

    def all_cb(self):
        self.paths = self.full_paths
        if self.rel_paths is not None:
            self.selected_rel_paths = self.rel_paths
        self._close()

    def remaining_cb(self):
        idx = self.values.index(self.pos_n_sv.get())
        self.paths = self.full_paths[idx:]
        if self.rel_paths is not None:
            self.selected_rel_paths = [self.rel_paths[idx:]]
        self._close()

    def open_path_explorer(self):
        if self.full_paths is None:
            path = self.pos_n_sv.get()
            subprocess.Popen('explorer "{}"'.format(os.path.normpath(path)))
        else:
            sv_txt = self.pos_n_sv.get()
            sv_idx = self.values.index(sv_txt)
            path = self.full_paths[sv_idx]
            subprocess.Popen('explorer "{}"'.format(os.path.normpath(path)))

    def get_values_segmGUI(self, exp_path):
        pos_foldernames = natsorted(os.listdir(exp_path))
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            last_tracked_i_found = False
            images_path = f'{exp_path}/{pos}/Images'
            filenames = os.listdir(images_path)
            for filename in filenames:
                if filename.find('_last_tracked_i.txt') != -1:
                    last_tracked_i_found = True
                    last_tracked_i_path = f'{images_path}/{filename}'
                    with open(last_tracked_i_path, 'r') as txt:
                        last_tracked_i = int(txt.read())
            if last_tracked_i_found:
                values.append(f'{pos} (Last tracked frame: {last_tracked_i})')
            else:
                values.append(pos)
        self.values = values
        return values

    def _close(self):
        try:
            tot_pos = [self.total_pos[self.full_paths.index(p)] for p in self.paths]
            self.tot_pos = sum(tot_pos)
        except TypeError:
            pass
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        exit('Execution aborted by the user')

class beyond_listdir_spotMAX:
    def __init__(self, folder_path, vNUM=None, run_num=None, multi_run_msg=None):
        print('Scanning folders...')
        self.multi_run_msg = multi_run_msg
        self.bp = apps.tk_breakpoint()
        self.folder_path = folder_path
        self.TIFFs_path = []
        self.count_recursions = 0
        self.vNUM = vNUM
        self.listdir_recursion(folder_path)
        if not self.TIFFs_path:
            raise FileNotFoundError(f'Path {folder_path} is not valid!')
        run_num = self.get_run_nums()
        self.run_num = run_num
        self.all_exp_info = self.count_analysed_pos(vNUM, run_num)

    def listdir_recursion(self, folder_path):
        if os.path.isdir(folder_path):
            listdir_folder = natsorted(os.listdir(folder_path))
            contains_pos_folders = any([name.find('Position_')!=-1
                                        for name in listdir_folder])
            if not contains_pos_folders:
                contains_TIFFs = any([name=='TIFFs' for name in listdir_folder])
                contains_CZIs = any([name=='CZIs' for name in listdir_folder])
                if contains_TIFFs:
                    self.TIFFs_path.append(f'{folder_path}/TIFFs')
                elif contains_CZIs:
                    self.TIFFs_path.append(f'{folder_path}')
                else:
                    for name in listdir_folder:
                        subfolder_path = f'{folder_path}/{name}'
                        self.listdir_recursion(subfolder_path)
            else:
                self.TIFFs_path.append(folder_path)

    def get_rel_path(self, path):
        rel_path = ''
        parent_path = path
        count = 0
        while parent_path != self.folder_path or count==10:
            if count > 0:
                rel_path = f'{os.path.basename(parent_path)}/{rel_path}'
            parent_path = os.path.dirname(parent_path)
            count += 1
        rel_path = f'.../{rel_path}'
        return rel_path

    def get_run_nums(self):
        run_nums_li = []
        scanned_TIFFs_paths = []
        scanned_pos_paths = []
        all_runs = []
        spotmax_paths = []
        for i, path in enumerate(self.TIFFs_path):
            foldername = os.path.basename(path)
            if foldername == 'TIFFs':
                pos_foldernames = natsorted(
                                     [p for p in os.listdir(path)
                                     if p.find('Position_')!=-1
                                     and os.path.isdir(os.path.join(path, p))])
                for p in pos_foldernames:
                    pos_path = os.path.join(path, p)
                    spotmax_path = os.path.join(pos_path, 'spotMAX_output')
                    if not os.path.exists(spotmax_path):
                        spotmax_path = os.path.join(pos_path, 'NucleoData')
                    if os.path.exists(spotmax_path):
                        spotmax_paths.append(spotmax_path)
                        filenames = os.listdir(spotmax_path)
                        run_nums = [re.findall('(\d+)_(\d)_', f)
                                             for f in filenames]
                        run_nums = np.unique(
                                   np.array(
                                    [int(m[0][0]) for m in run_nums if m], int))
                        run_nums_li.append(run_nums)
                        all_runs.extend(run_nums)
                    else:
                        spotmax_paths.append('')
                        run_nums_li.append(np.nan)
                    scanned_TIFFs_paths.append(path)
                    scanned_pos_paths.append(pos_path)
        df_run_nums = pd.DataFrame({'TIFFs_path': scanned_TIFFs_paths,
                                    'pos_path': scanned_pos_paths,
                                    'run_nums': run_nums_li}
                                    ).set_index('TIFFs_path')
        unique_runs = np.unique(np.array(all_runs, int))
        is_multi_run = len(unique_runs) > 1
        multi_run_msg = self.multi_run_msg
        if multi_run_msg is None:
            multi_run_msg = ('Select run number to scan:\n'
                   '(The software will determine if size and/or\n'
                   'gaussian fits were performed\n'
                   'for the selected run)')
        if is_multi_run:
            self.spotmax_paths = spotmax_paths
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.title('Multiple runs detected!')
            root.geometry("+800+400")
            tk.Label(root,
                     text=multi_run_msg,
                     font=(None, 10),
                     justify='right').grid(row=1, column=0, pady=10, padx=10)

            tk.Button(root, text='Ok', width=20,
                            command=self._close).grid(row=3, column=1,
                                                      columnspan=2,
                                                      pady=(0,10), padx=10)

            run_num_Intvar = tk.IntVar()
            run_num_combob = ttk.Combobox(root, width=15, justify='center',
                                          textvariable=run_num_Intvar)
            run_num_combob.option_add('*TCombobox*Listbox.Justify', 'center')
            run_num_combob['values'] = list(unique_runs)
            run_num_combob.grid(column=1, row=1, padx=10, pady=(10, 0))
            run_num_combob.current(0)

            print_b = tk.Button(root, text='Print analysis inputs', width=20,
                                command=self._print_analysis_inputs)
            print_b.grid(row=2, column=1, columnspan=2, pady=(0, 5), padx=10)
            print_b.config(font=(None, 9, 'italic'))

            root.protocol("WM_DELETE_WINDOW", self._abort)
            self.run_num_Intvar = run_num_Intvar
            self.root = root
            root.mainloop()
            run_num = self.run_num_Intvar.get()
        elif unique_runs:
            run_num = unique_runs[0]
        else:
            run_num = 1
        self.df_run_nums = df_run_nums
        return run_num

    def _print_analysis_inputs(self):
        run_num = self.run_num_Intvar.get()
        for spotmax_path in self.spotmax_paths:
            analysis_inputs_path = os.path.join(
                    spotmax_path,
                    f'{run_num}_{self.vNUM}_analysis_inputs.csv'
            )
            if os.path.exists(analysis_inputs_path):
                df_inputs = pd.read_csv(analysis_inputs_path)
                df_inputs['Description'] = df_inputs['Description'].str.replace(
                                                                '\u03bc', 'u')
                df_inputs.set_index('Description', inplace=True)
                print('================================')
                print(f'Analysis inputs for run number {run_num}:')
                print('')
                print(df_inputs)
                print('================================')
                break

    def _close(self):
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        exit('Execution aborted by the user')
        self.root.quit()
        self.root.destroy()


    def count_analysed_pos(self, vNUM=None, run_num=None):
        all_exp_info = []
        NONspotCOUNTED_pos_info = []
        NONspotSIZED_pos_info = []
        NONspotFIT_pos_info = []
        tot_exp = len(self.TIFFs_path)
        self.tot_pos = []
        for path in tqdm(self.TIFFs_path, unit=' folders', leave=False):
            foldername = os.path.basename(path)
            if foldername == 'TIFFs':
                pos_foldernames = [p for p in os.listdir(path)
                                     if p.find('Position_')!=-1
                                     and os.path.isdir(os.path.join(path, p))]
                self.tot_pos.append(len(pos_foldernames))
                rel_path = self.get_rel_path(path)
                num_analysed_pos = 0
                num_spotSIZED_pos = 0
                num_spotFIT_pos = 0
                num_pos = len(pos_foldernames)
                spotFIT_filename = '4_spotFIT_data'
                NONspotCOUNTED_pos = []
                NONspotSIZED_pos = []
                NONspotFIT = []
                for pos_foldername in pos_foldernames:
                    spotmax_path = f'{path}/{pos_foldername}/spotMAX_output'
                    if not os.path.exists(spotmax_path):
                        spotmax_path = os.path.join(path, pos_foldername,
                                                    'NucleoData')
                    dir_exists = False
                    if os.path.exists(spotmax_path):
                        dir_exists = True
                        not_empty = len(os.listdir(spotmax_path)) > 5
                    if dir_exists and not_empty:
                        num_analysed_pos += 1
                        # Check which positions have been spotSIZED
                        if vNUM is not None:
                            if run_num is not None:
                                basename = f'{run_num}_{spotFIT_filename}_{vNUM}'
                                to_find = f'{basename}.h5'
                            else:
                                basename = f'{spotFIT_filename}_{vNUM}'
                                to_find = f'{basename}.h5'
                        else:
                            basename = f'{spotFIT_filename}'
                            to_find = f'{basename}.h5'
                        filenames = os.listdir(spotmax_path)
                        find_li = [f.find(to_find)!=-1 for f in filenames]
                        spotSIZE_found = any(find_li)
                        if spotSIZE_found:
                            num_spotSIZED_pos += 1
                            # Check if gauss fit was performed
                            spotFIT_done_fname = f'{basename}_spotFIT_done.txt'
                            spotFIT_done_path = os.path.join(
                                                        spotmax_path,
                                                        spotFIT_done_fname)
                            spotFIT_done = any([f.find(spotFIT_done_fname)!=-1
                                              for f in filenames])

                            spotFIT_NOT_done_fname = f'{basename}_spotFIT_NOT_done.txt'
                            spotFIT_NOT_done_path = os.path.join(
                                                        spotmax_path,
                                                        spotFIT_NOT_done_fname)
                            spotFIT_NOT_done = any([
                                            f.find(spotFIT_NOT_done_fname)!=-1
                                            for f in filenames
                            ])
                            if spotFIT_done:
                                num_spotFIT_pos += 1
                            elif spotFIT_NOT_done:
                                NONspotFIT.append(spotmax_path)
                            else:
                                found_idx = [i for i,f in enumerate(find_li)
                                                                       if f][0]
                                filename = filenames[found_idx]
                                spotSIZED_path = f'{spotmax_path}/{filename}'
                                try:
                                    df = pd.read_hdf(spotSIZED_path,
                                                     key='frame_0')
                                    if 'sigma_z' in df.columns:
                                        num_spotFIT_pos += 1
                                        with open(spotFIT_done_path, 'w') as txt:
                                            txt.write('3D Gaussian fit was '
                                                      'performed!')
                                    else:
                                        with open(spotFIT_NOT_done_path, 'w') as txt:
                                            txt.write('3D Gaussian fit NOT '
                                                      'performed!')
                                        NONspotFIT.append(spotmax_path)
                                except Exception as e:
                                    NONspotFIT.append(spotmax_path)
                        else:
                            NONspotSIZED_pos.append(spotmax_path)
                    else:
                        NONspotCOUNTED_pos.append(f'{path}/{pos_foldername}')
                NONspotCOUNTED_pos_info.append(NONspotCOUNTED_pos)
                NONspotSIZED_pos_info.append(NONspotSIZED_pos)
                NONspotFIT_pos_info.append(NONspotFIT)
                if num_analysed_pos < num_pos:
                    if num_analysed_pos != 0:
                        if vNUM is None:
                            exp_info = (f'{rel_path} '
                                f'(N. of analysed pos.: {num_analysed_pos})')
                        else:
                            exp_info = (f'{rel_path} '
                            f'(N. of spotCOUNTED pos.: {num_analysed_pos}, '
                            f'N. of spotSIZED pos.: {num_spotSIZED_pos}, '
                            f'N. of spotFITTED pos.: {num_spotFIT_pos})')
                    else:
                        exp_info = (f'{rel_path} '
                                    '(NONE of the pos. have been analysed)')
                elif num_analysed_pos == num_pos:
                    if vNUM is None:
                        exp_info = (f'{rel_path} (All pos. analysed)')
                    else:
                        if num_spotSIZED_pos == num_pos:
                            if num_spotFIT_pos == num_pos:
                                exp_info = (f'{rel_path} (All pos. spotCOUNTED, '
                                            f'All pos. spotSIZED, '
                                            'All pos. spotFITTED)')
                            else:
                                exp_info = (f'{rel_path} (All pos. spotCOUNTED, '
                                            f'All pos. spotSIZED, '
                                            'N. of spotFITTED pos.: '
                                            f'{num_spotFIT_pos})')
                        else:
                            exp_info = (f'{rel_path} (All pos. spotCOUNTED, '
                                f'N. of spotSIZED pos: {num_spotSIZED_pos},'
                                ' N. of spotFITTED pos.: '
                                f'{num_spotFIT_pos}')
                elif num_analysed_pos > num_pos:
                    num_analysed_pos = (f'{rel_path} (WARNING:'
                                   'multiple "spotMAX_output" folders found!)')
                else:
                    exp_info = rel_path
            else:
                rel_path = self.get_rel_path(path)
                exp_info = f'{rel_path} (FIJI macro not executed!)'
            all_exp_info.append(exp_info)
            self.NONspotCOUNTED_pos_info = NONspotCOUNTED_pos_info
            self.NONspotSIZED_pos_info = NONspotSIZED_pos_info
            self.NONspotFIT_pos_info = NONspotFIT_pos_info
        return all_exp_info


def spotfit_checkpoint(pos_path):
    skip = False
    nucleodata_path = os.path.join(pos_path, 'NucleoData')
    spotmax_path = os.path.join(pos_path, 'spotMAX_output')
    if os.path.exists(nucleodata_path):
        os.rename(nucleodata_path, spotmax_path)
    spotCOUNT_data_path = os.path.join(spotmax_path,
                                       '1_3_p-_ellip_test_data_v1.h5')
    if not os.path.exists(spotCOUNT_data_path):
        skip = True
    return skip

def get_main_paths(selected_path, vNUM):
    is_images_path = os.path.basename(selected_path) == 'Images'
    if is_images_path:
        selected_path = os.path.dirname(selected_path)
    is_pos_path = os.path.basename(selected_path).find('Position_') != -1
    is_TIFFs_path = any([item.find('Position_')!=-1 for item in os.listdir(selected_path)])
    multi_run_msg = ('Select run number to scan:\n\n'
                     '(The software will determine if size and/or\n'
                     'gaussian fits were performed\n'
                     'for the selected run)')
    if not is_pos_path and not is_TIFFs_path:
        selector = select_exp_folder()
        beyond_listdir = beyond_listdir_spotMAX(selected_path, vNUM=vNUM,
                                                multi_run_msg=multi_run_msg)
        selector.run_widget(
                 beyond_listdir.all_exp_info,
                 title='spotMAX: Select experiment to analyse',
                 label_txt='Select experiment to analyse',
                 full_paths=beyond_listdir.TIFFs_path,
                 showinexplorer_button=True,
                 all_button=True,
                 remaining_button=True,
                 selected_path=selected_path,
                 NONspotCOUNTED_pos=beyond_listdir.NONspotCOUNTED_pos_info,
                 NONspotSIZED_pos=beyond_listdir.NONspotSIZED_pos_info,
                 NONspotFIT_pos=beyond_listdir.NONspotFIT_pos_info,
                 total_pos=beyond_listdir.tot_pos)
        tot_pos = selector.tot_pos
        main_paths = selector.paths
        prompts_pos_to_analyse = False
        run_num = beyond_listdir.run_num
    elif is_TIFFs_path:
        # The selected path is already the folder containing Position_n folders
        prompts_pos_to_analyse = True
        main_paths = [selected_path]
        ls_selected_path = os.listdir(selected_path)
        pos_paths = [os.path.join(selected_path, p) for p in ls_selected_path
                          if p.find('Position_') != -1
                          and os.path.isdir(os.path.join(selected_path, p))]
        scan_run_num = prompts.scan_run_nums(vNUM)
        run_nums = scan_run_num.scan(pos_paths)
        tot_pos = len(pos_paths)
        if len(run_nums) > 1:
            run_num = scan_run_num.prompt(run_nums, msg=multi_run_msg)
        else:
            run_num = 1
    elif is_pos_path:
        tot_pos = 1
        prompts_pos_to_analyse = False
        main_paths = [selected_path]
        scan_run_num = prompts.scan_run_nums(vNUM)
        run_nums = scan_run_num.scan(main_paths)
        if len(run_nums) > 1:
            run_num = scan_run_num.prompt(run_nums, msg=multi_run_msg)
        else:
            run_num = 1

    return (main_paths, prompts_pos_to_analyse, run_num, tot_pos, is_pos_path,
            is_TIFFs_path)

if __name__ == '__main__':
    path = r'G:\My Drive\1_MIA_Data\Anika\WTs\SCD'
    pathScanner = beyond_listdir_spotMAX(path)
    print(pathScanner.TIFFs_path)
