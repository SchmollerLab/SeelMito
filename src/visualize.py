import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.measure
from MyWidgets import Slider, Button, MyRadioButtons
from natsort import natsorted
from ast import literal_eval

import prompts, core, apps

matplotlib.use('TkAgg')
plt.style.use('dark_background')
plt.rc('axes', edgecolor='0.1')

vNUM = 'v1'
TIFFs_path = prompts.folder_dialog(title='Select TIFFs folder')

if os.path.basename(TIFFs_path) == 'TIFFs':
    listdir_TIFFs = os.listdir(TIFFs_path)
    pos_foldernames = natsorted(
                       [p for p in listdir_TIFFs
                       if os.path.isdir(os.path.join(TIFFs_path, p))
                       and p.find('Position_')!=-1])
elif os.path.basename(TIFFs_path).find('Position_') != -1:
    pos_foldernames = [os.path.basename(TIFFs_path)]
    TIFFs_path = os.path.dirname(TIFFs_path)
else:
    raise FileNotFoundError(f'The path {TIFFs_path} is not a valid path. '
                            'Only "/TIFFs" or "/TIFFs/Potision_n" allowed.')

ch_name_selector = prompts.select_channel_name()
ref_ch_name_selector = prompts.select_channel_name()
scan_run_num = prompts.scan_run_nums(vNUM)
pos_folder_widget = prompts.single_combobox_widget()
h5_filename_widget = prompts.single_combobox_widget()

spots_img_li = [None]*len(pos_foldernames)
sharpen_spots_img_li = [None]*len(pos_foldernames)
ref_ch_img_li = [None]*len(pos_foldernames)
pos_to_visualize = None
pos_foldername_visualized = [None]*len(pos_foldernames)
h5_data_li = [None]*len(pos_foldernames)
segm_3D_li = [None]*len(pos_foldernames)
print('Loading and processing data...')
for p, pos in enumerate(pos_foldernames):
    if pos_to_visualize is not None:
        # Skip position folders to not visualise
        if pos != pos_to_visualize:
            continue
    pos_path = os.path.join(TIFFs_path, pos)
    images_path = os.path.join(pos_path, 'Images')
    spotMAX_path = os.path.join(pos_path, 'spotMAX_output')

    # Load analysis inputs
    if os.path.exists(spotMAX_path):
        if scan_run_num.is_first_call:
            run_nums = scan_run_num.scan(pos_path)
            if len(run_nums) > 1:
                run_num = scan_run_num.prompt(run_nums)
            else:
                run_num = 1
        analysis_inputs_path = os.path.join(
                spotMAX_path, f'{run_num}_{vNUM}_analysis_inputs.csv'
        )
        df_inputs = (pd.read_csv(analysis_inputs_path)
                       .set_index('Description'))
        v = 'Values'
        load_ref_ch = df_inputs.at['Load a reference channel?', v] == 'True'
    else:
        raise FileNotFoundError('spotMAX data not found. '
              f'The folder {spotMAX_path} does not exist')

    # Prompt user to select channel names
    filenames = os.listdir(images_path)
    if ch_name_selector.is_first_call:
        ch_names = ch_name_selector.get_available_channels(filenames)
        ch_name_selector.prompt(ch_names)
        channel_name = ch_name_selector.channel_name
    if ref_ch_name_selector.is_first_call and load_ref_ch:
        ref_ch_names = ref_ch_name_selector.get_available_channels(filenames)
        ref_ch_names = [c for c in ch_names if c.find(channel_name)==-1]
        ref_ch_name_selector.prompt(ref_ch_names,
                                    message='Select REFERENCE channel name')
        ref_channel_name = ref_ch_name_selector.channel_name

    # Load images and data
    spots_ch_aligned_found = False
    ref_ch_aligned_found = False
    segm_found = False
    spots_h5_available = []
    for j, f in enumerate(filenames):
        if f.find(f'_{channel_name}_aligned.npy') != -1:
            spots_ch_aligned_found = True
            spots_aligned_i = j
        elif f.find(f'_{channel_name}.tif') != -1:
            spots_tif_i = j
        elif f.find('_segm.npy') != -1:
            segm_found = True
            segm_i = j
        if load_ref_ch:
            if f.find(f'_{ref_channel_name}_aligned.npy') != -1:
                ref_ch_aligned_found = True
                ref_aligned_i = j
            elif f.find(f'_{ref_channel_name}.tif') != -1:
                ref_tif_i = j
    if spots_ch_aligned_found:
        spots_ch_path = os.path.join(images_path, filenames[spots_aligned_i])
        V_spots = np.load(spots_ch_path)
    else:
        spots_ch_path = os.path.join(images_path, filenames[spots_tif_i])
        V_spots = skimage.io.imread(spots_ch_path)
    if load_ref_ch:
        if ref_ch_aligned_found:
            ref_ch_path = os.path.join(images_path, filenames[ref_aligned_i])
            V_ref = np.load(ref_ch_path)
        else:
            ref_ch_path = os.path.join(images_path, filenames[ref_tif_i])
            V_ref = skimage.io.imread(ref_ch_path)
    if segm_found:
        segm_info = df_inputs.at['Segmentation info (ignore if not present):', v]
        segm_path = os.path.join(images_path, filenames[segm_i])
        segm_npy = np.load(segm_path)
        if segm_info == '2D':
            # Tile 2D segmentation into 3D
            segm_npy = np.array([segm_npy]*len(V_spots))
    else:
        segm_info = None

    # Check if data is 4D and prompt to select which position to visualize
    if V_spots.ndim > 4 and pos_folder_widget.is_first_call:
        pos_folder_widget.prompt(pos_foldernames,
                                 title='Select position folder',
                                 message='Data is 4D.\n'
                                 'Select which Position folder to visualize\n')
        pos_to_visualize = pos_folder_widget.selected_val
        continue

    # Preprocess images if needed
    gauss_sigma = float(df_inputs.at['Gaussian filter sigma:', v])
    if gauss_sigma > 0:
        V_spots = skimage.filters.gaussian(V_spots, sigma=gauss_sigma)
        if load_ref_ch:
            V_ref = skimage.filters.gaussian(V_ref, sigma=gauss_sigma)
    sharpen = df_inputs.at['Sharpen image prior spot detection?', v] == 'True'

    # Sharpen spots image if needed
    if sharpen:
        wavelen_mask = df_inputs.index.str.find('emission wavelength') > -1
        wavelen = float(df_inputs[wavelen_mask].iloc[0][v])
        NA = float(df_inputs.at['Numerical aperture:', v])
        yx_resol_multip = float(df_inputs.at['YX resolution multiplier:', v])
        zyx_vox_dim = literal_eval(df_inputs.at['ZYX voxel size (um):', v])
        z_resolution_limit = float(df_inputs.at['Z resolution limit (um):', v])
        (zyx_resolution,
        zyx_resolution_pxl, _) = core.calc_resolution_limited_vol(wavelen, NA,
                                                yx_resol_multip,
                                                zyx_vox_dim,
                                                z_resolution_limit)
        V_spots_blurred = skimage.filters.gaussian(V_spots,
                                                   sigma=zyx_resolution_pxl)
        V_spots_sharp = V_spots - V_spots_blurred
        sharpen_spots_img_li[p] = V_spots_sharp

    spots_img_li[p] = V_spots
    if segm_found:
        segm_3D_li[p] = segm_npy
    if load_ref_ch:
        ref_ch_img_li[p] = V_ref
    pos_foldername_visualized[p] = pos

    # Load spots .h5 data
    if h5_filename_widget.is_first_call:
        pattern = f'{run_num}_(.*)_data_{vNUM}.h5'
        spots_h5_available = [f for f in os.listdir(spotMAX_path)
                              if re.match(pattern, f) is not None]
        h5_filename_widget.prompt(spots_h5_available,
                                 title='Select .h5 filename',
                                 message='Select which spots data to visualize\n')
        h5_to_load = h5_filename_widget.selected_val

    for f in os.listdir(spotMAX_path):
        if f.find(h5_to_load) != -1:
            h5_path = os.path.join(spotMAX_path, h5_to_load)
            if V_spots.ndim < 4:
                df_h5 = pd.read_hdf(h5_path)
                h5_data_li[p] = df_h5
            else:
                raise NotImplementedError('4D images not supported yet.')

"""Initialize app"""
print('Starting spotMAX visualizer...')
class spotMAX_visualize_app:
    def __init__(self, fig, ax, spots_img_li, sharpen_spots_img_li,
                 ref_ch_img_li, pos_foldername_visualized,
                 h5_data_li, load_ref_ch, segm_3D_li, segm_info):
        self.spots_img_li = spots_img_li
        self.sharpen_spots_img_li = sharpen_spots_img_li
        self.ref_ch_img_li = ref_ch_img_li
        self.pos_foldername_visualized = pos_foldername_visualized
        self.h5_data_li = h5_data_li
        self.segm_3D_li = segm_3D_li
        self.i = 0
        self.num_frames = len(spots_img_li)
        self.fig = fig
        self.ax = ax
        self.load_ref_ch = load_ref_ch
        self.segm_info = segm_info
        self.is_max_proj = False

    def init_plots(self):
        fig, ax = self.fig, self.ax
        img = self.spots_img_li[0].max(axis=0)
        self.imshow_spots = ax[0].imshow(img)
        if self.load_ref_ch:
            img_ref = self.ref_ch_img_li[0].max(axis=0)
            self.imshow_ref = ax[1].imshow(img_ref)
        zz = self.h5_data_li[0]['z'].to_numpy()
        yy = self.h5_data_li[0]['y'].to_numpy()
        xx = self.h5_data_li[0]['x'].to_numpy()
        self.plot_spots_coords, = ax[0].plot(xx, yy, 'r.')
        IDs_cont_val = self.radiob_IDs_cont.value_selected
        contours_ON = (IDs_cont_val == 'Only contours'
                       or IDs_cont_val == 'Contours and IDs')
        IDs_ON = (IDs_cont_val == 'Only IDs'
                  or IDs_cont_val == 'Contours and IDs')
        if self.segm_info is not None:
            self.lab = self.segm_3D_li[0].max(axis=0)
            self.rp = skimage.measure.regionprops(self.lab)
            IDs = [obj.label for obj in self.rp]
            self.cont_plots = []
            self.contours = (apps.auto_select_slice()
                             .find_contours(self.lab, IDs, group=True))
            for cont in self.contours:
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                c, = ax[0].plot(x, y, c='r', alpha=0.5, lw=2, ls='--')
                c1, = ax[1].plot(x, y, c='r', alpha=0.5, lw=2, ls='--')
                self.cont_plots.append((c, c1))
            self.IDs_texts = []
            for obj in self.rp:
                y, x = obj.centroid
                txt = str(obj.label)
                t = ax[1].text(x, y, txt, fontsize=12, fontweight='semibold')
                self.IDs_texts.append(t)
            if not contours_ON:
                for c in self.cont_plots:
                    c[0].set_data([[],[]])
                    c[1].set_data([[],[]])
            if not IDs_ON:
                for t in self.IDs_texts:
                    t.set_text('')
        for a in ax:
            a.axis('off')

    def next_img(self):
        if self.i+1 < self.num_frames:
            self.i += 1
            self.update_plots()
        else:
            self.i = 0
            self.update_plots()

    def prev_img(self):
        if self.i > 0:
            self.i -= 1
            self.update_plots()
        else:
            self.i = len(self.spots_img_li)-1
            self.update_plots()

    def update_IDs_cont(self, label):
        IDs_cont_val = self.radiob_IDs_cont.value_selected
        contours_ON = (IDs_cont_val == 'Only contours'
                       or IDs_cont_val == 'Contours and IDs')
        IDs_ON = (IDs_cont_val == 'Only IDs'
                  or IDs_cont_val == 'Contours and IDs')
        if contours_ON:
            if self.segm_info is not None:
                for i, cont in enumerate(self.contours):
                    x = cont[:,1]
                    y = cont[:,0]
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    self.cont_plots[i][0].set_data(x, y)
                    self.cont_plots[i][1].set_data(x, y)
        else:
            for c in self.cont_plots:
                c[0].set_data([[],[]])
                c[1].set_data([[],[]])
        if IDs_ON:
            if self.segm_info is not None:
                for i, obj in enumerate(self.rp):
                    y, x = obj.centroid
                    txt = str(obj.label)
                    self.IDs_texts[i].set_text(txt)
        else:
            for t in self.IDs_texts:
                t.set_text('')
        self.fig.canvas.draw_idle()

    def update_plots(self, event=None):
        if self.is_max_proj:
            img = self.spots_img_li[self.i].max(axis=0)
        else:
            z = int(self.z_slice_slider.val)
            img = self.spots_img_li[self.i][z]
        self.imshow_spots.set_data(img)
        if self.load_ref_ch:
            if self.is_max_proj:
                img_ref = self.ref_ch_img_li[self.i].max(axis=0)
            else:
                z = int(self.z_slice_slider.val)
                img_ref = self.ref_ch_img_li[self.i][z]
            self.imshow_ref = ax[1].imshow(img_ref)
        zz = self.h5_data_li[self.i]['z'].to_numpy()
        yy = self.h5_data_li[self.i]['y'].to_numpy()
        xx = self.h5_data_li[self.i]['x'].to_numpy()
        if not self.is_max_proj:
            yy = yy[zz == int(self.z_slice_slider.val)]
            xx = xx[zz == int(self.z_slice_slider.val)]

        self.plot_spots_coords.set_data(xx, yy)
        self.update_IDs_cont(None)


num_img = 2 if load_ref_ch else 1

fig, ax = plt.subplots(1, num_img, sharex=True, sharey=True)
if num_img == 1:
    ax = [ax]

app = spotMAX_visualize_app(fig, ax, spots_img_li, sharpen_spots_img_li,
                            ref_ch_img_li, pos_foldername_visualized,
                            h5_data_li, load_ref_ch, segm_3D_li, segm_info)

# Widgets colors
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'
button_true_color = '0.4'

# Widgets axes
ax_z_slice_slider = plt.axes([0.1, 0.78, 0.25, 0.2])
ax_z_proj_button = plt.axes([0.1, 0.78, 0.25, 0.39])
ax_radiob_IDs_cont = plt.axes([0.1, 0.96, 0.25, 0.39])
if sharpen:
    ax_radiob_sharp = plt.axes([0.1, 0.97, 0.34, 0.2])

# Widgets
app.z_slice_slider = Slider(ax_z_slice_slider,
        'z-slice', 0, spots_img_li[0].shape[0],
        valinit=spots_img_li[0].shape[0]/2,
        valstep=1,
        orientation='horizontal',
        color=slider_color,
        init_val_line_color=hover_color,
        valfmt='%1.0f')
app.z_proj_b = Button(ax_z_proj_button, 'Max',
        color=axcolor, hovercolor=hover_color,
        presscolor=presscolor)

app.radiob_IDs_cont = MyRadioButtons(ax_radiob_IDs_cont,
              ('None', 'Only IDs', 'Only contours', 'Contours and IDs'),
              active = 0,
              activecolor = button_true_color,
              orientation = 'horizontal',
              size = 59,
              circ_p_color = button_true_color)

if sharpen:
    app.radiob_sharp = MyRadioButtons(ax_radiob_sharp,
                  ('Original', 'Sharp'),
                  active = 0,
                  activecolor = button_true_color,
                  orientation = 'horizontal',
                  size = 59,
                  circ_p_color = button_true_color)

def z_proj_b_callback(event):
    app.is_max_proj = True
    app.update_plots()
    app.is_max_proj = False


def key_down(event):
    key = event.key
    if key == 'right':
        app.next_img()
    elif key == 'left':
        app.prev_img()

def resize_widgets(self):
    # [left, bottom, width, height]
    H = 0.03
    W = 0.1
    spF = 0.01
    ax0 = ax[0]
    ax0_l, ax0_b, ax0_r, ax0_t = ax0.get_position().get_points().flatten()
    ax0_w = ax0_r-ax0_l
    ax0_c = ax0_l + ax0_w/2
    ax_radiob_IDs_cont.set_position([ax0_l, ax0_t+spF, ax0_w, H])
    b = ax0_b-spF-H
    ax_z_slice_slider.set_position([ax0_l, b, ax0_w, H])
    L = ax0_l+ax0_w+2*spF
    ax_z_proj_button.set_position([L, b, W/3, H])
    b1 = b-spF-H
    w = 1/3*ax0_w
    l1 = ax0_c - w/2
    ax_radiob_sharp.set_position([l1, b1, w, H])

# Connect to events
app.z_proj_b.on_clicked(z_proj_b_callback)
app.radiob_IDs_cont.on_clicked(app.update_IDs_cont)
app.z_slice_slider.on_changed(app.update_plots)
fig.canvas.mpl_connect('resize_event', resize_widgets)
fig.canvas.mpl_connect('key_press_event', key_down)

try:
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
except:
    pass

app.init_plots()

fig.canvas.set_window_title(f'spotMAX visualize results of {TIFFs_path}')
plt.show()

print('spotMAX visualizer closed')
