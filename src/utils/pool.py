import os
import sys
import numpy as np
import pandas as pd
from functools import reduce
import skimage.measure
import skimage.io
import scipy.ndimage
import skimage.filters
import skimage.morphology
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(script_dir)
sys.path.append(main_dir)

import apps, core, prompts

TIFFs_path = prompts.folder_dialog(
                       title='Select TIFFs path containing Position_n folders')

if os.path.basename(TIFFs_path).find('TIFFs') == -1:
    raise FileNotFoundError('Not a TIFFs folder. '
    f'The selected path {TIFFs_path} is not a TIFFs folder.')

listdir_TIFFs = os.listdir(TIFFs_path)

pos_foldernames = natsorted([p for p in listdir_TIFFs
                     if os.path.isdir(os.path.join(TIFFs_path, p))
                     and p.find('Position_') != -1])

if not pos_foldernames:
    raise FileNotFoundError('Empty folder. '
    f'The selected path {TIFFs_path} does not contain any Position folder.')

ch_name_selector = prompts.select_channel_name(which_channel='spots')
ref_ch_name_selector = prompts.select_channel_name(which_channel='ref')

load_ref_ch = prompts.askyesno(title='Load reference channel?',
              message='Do you also want to pool all the cells\n'
                      'from reference channel?')

print('Loading segmentation files...')
segm_obj_widths = []
segm_obj_hegihts = []
segm_obj_centers = [[] for _ in range(len(pos_foldernames))]
for p, pos in enumerate(tqdm(pos_foldernames, unit=' pos')):
    pos_path = os.path.join(TIFFs_path, pos)
    images_path = os.path.join(pos_path, 'Images')

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

    segm_npy = None
    cca_df = None
    for f in filenames:
        if f.endswith('segm.npy'):
            segm_path = os.path.join(images_path, f)
            segm_npy = np.load(segm_path)
            if segm_npy.ndim > 2:
                raise IndexError('Shape error. '
                'Segmentation files with more than 2 dimensions '
                'are not supported for pooling yet.')
        elif f.endswith('cc_stage.csv'):
            cca_df = pd.read_csv(os.path.join(images_path, f)).set_index('Cell_ID')

    if segm_npy is None:
        raise FileNotFoundError('Segmentation file not found! '
        'Cropping and pooling all the cells together is not supported without '
        'a segmentation file yet.')

    rp = skimage.measure.regionprops(segm_npy)
    IDs = [obj.label for obj in rp]
    for obj in rp:
        ID = obj.label
        y_min, x_min, y_max, x_max = obj.bbox
        h, w = (y_max-y_min), (x_max-x_min)
        yc = y_min+h/2
        xc = x_min+w/2
        if cca_df is not None:
            # skip buds
            if cca_df.at[ID, 'Relationship'] == 'bud':
                continue
            if cca_df.at[ID, 'Cell cycle stage'] == 'S':
                bud_ID = cca_df.at[ID, 'Relative\'s ID']
                _temp_segm = np.zeros(segm_npy.shape, bool).astype(np.uint8)
                _temp_segm[segm_npy==ID] = 1
                _temp_segm[segm_npy==bud_ID] = 1
                _temp_rp = skimage.measure.regionprops(_temp_segm)

                y_min, x_min, y_max, x_max = _temp_rp[0].bbox
                h, w = (y_max-y_min), (x_max-x_min)
                yc = y_min+h/2
                xc = x_min+w/2

        segm_obj_widths.append(w)
        segm_obj_hegihts.append(h)
        segm_obj_centers[p].append((yc, xc))

hH = int(np.ceil(max(segm_obj_hegihts)/2))
hW = int(np.ceil(max(segm_obj_widths)/2))

all_V_spots = []
all_V_ref = []
all_segm_crop = []

print('')
print('Loading and cropping images files...')
pos_ID = 0
all_IDs = []
for p, pos in enumerate(tqdm(pos_foldernames, unit=' pos')):
    pos_path = os.path.join(TIFFs_path, pos)
    images_path = os.path.join(pos_path, 'Images')

    filenames = os.listdir(images_path)
    for f in filenames:
        if f.endswith('segm.npy'):
            segm_path = os.path.join(images_path, f)
            segm_npy = np.load(segm_path)
        elif f.endswith(f'{channel_name}.tif'):
            spots_path = os.path.join(images_path, f)
            V_spots = skimage.io.imread(spots_path)
        elif f.endswith(f'{ref_channel_name}.tif') and load_ref_ch:
            ref_path = os.path.join(images_path, f)
            V_ref = skimage.io.imread(ref_path)

    Z, Y, X = V_spots.shape

    for yc, xc in segm_obj_centers[p]:
        y_min = int(yc)-hH-5
        x_min = int(xc)-hW-5
        y_max = int(yc)+hH+5
        x_max = int(xc)+hW+5

        pad_y_before = 0
        pad_x_before = 0
        pad_y_after = 0
        pad_x_after = 0
        # Check that cropping is within global image shape
        if y_min < 0:
            pad_y_before = -y_min
            y_min = 0
        if x_min < 0:
            pad_x_before = -x_min
            x_min = 0
        if y_max > Y:
            pad_y_after = y_max-Y
            y_max = Y
        if x_max > X:
            pad_x_after = x_max-Y
            x_max = X

        # Crop images
        V_spots_crop = V_spots[:, y_min:y_max, x_min:x_max]
        if load_ref_ch:
            V_ref_crop = V_ref[:, y_min:y_max, x_min:x_max]
        segm_npy_crop = segm_npy[y_min:y_max, x_min:x_max].copy()
        segm_npy_crop[segm_npy_crop!=0] += pos_ID

        # Pad images with zeros if the cropping box exceeded global shape
        pad_2D = ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
        pad_3D = ((0,0), *pad_2D)
        V_spots_crop = np.pad(V_spots_crop, pad_3D)
        if load_ref_ch:
            V_ref_crop = np.pad(V_ref_crop, pad_3D)
        segm_npy_crop = np.pad(segm_npy_crop, pad_2D)

        # Append for later stacking
        all_V_spots.append(V_spots_crop)
        if load_ref_ch:
            all_V_ref.append(V_ref_crop)
        all_segm_crop.append(segm_npy_crop)

    IDs = np.unique(segm_npy)
    IDs = IDs[IDs != 0]
    all_IDs.append((IDs+pos_ID, IDs))
    pos_ID += segm_npy.max()

# Stacking
num_cells = len(all_V_spots)
_, crop_h, crop_w = all_V_spots[0].shape
num_cells_sqrt = int(np.ceil(np.sqrt(num_cells)))
y_end = num_cells_sqrt*crop_h
x_end = num_cells_sqrt*crop_w

V_spots_pool = np.zeros((Z, y_end, x_end), dtype=V_spots.dtype)

if load_ref_ch:
    V_ref_pool = V_spots_pool.copy()
segm_npy_pool = np.zeros((y_end, x_end), dtype=segm_npy.dtype)

y_range = range(0, y_end, crop_h)
x_range = range(0, x_end, crop_w)
print('')
print('Squaring the pool....')
pbar = tqdm(total=len(all_V_spots), unit=' cell')
i = 0
for y in y_range:
    for x in x_range:
        if i < num_cells:
            V_spots_i = all_V_spots[i]
            V_spots_pool[:, y:y+crop_h, x:x+crop_w] = V_spots_i
            if load_ref_ch:
                V_ref_i = all_V_ref[i]
                V_ref_pool[:, y:y+crop_h, x:x+crop_w] = V_ref_i
            segm_i = all_segm_crop[i]
            segm_npy_pool[y:y+crop_h, x:x+crop_w] = segm_i
            i += 1
            pbar.update()
        else:
            break
pbar.close()

# Remove 0s
_proj = V_spots_pool.max(axis=0)
row_mask = np.all(_proj==0, axis=1)
col_mask = np.all(_proj==0, axis=0)
_3D_non0s_mask = np.ones(V_spots_pool.shape, bool)
_3D_non0s_mask[:, row_mask] = False
_3D_non0s_mask[:, :, col_mask] = False
_slice = scipy.ndimage.find_objects(_3D_non0s_mask.astype(np.uint8))[0]
_V_spots_pool = np.zeros(_3D_non0s_mask.shape, dtype=V_spots_pool.dtype)
V_spots_pool = V_spots_pool[_slice]
if load_ref_ch:
    V_ref_pool = V_ref_pool[_slice]
segm_npy_pool = segm_npy_pool[(_slice[1], _slice[2])]

# Replace leftover 0s with median
V_spots_pool[V_spots_pool==0] = np.median(V_spots_pool[V_spots_pool!=0])

# V_spots_pool = np.concatenate(all_V_spots, axis=1)

print(V_spots_pool.shape)

apps.imshow_tk(segm_npy_pool)

multiotsu_thresh_vals = skimage.filters.threshold_multiotsu(V_spots_pool.max(axis=0))
li_thresh_val = skimage.filters.threshold_li(V_spots_pool.max(axis=0),
                              initial_guess = lambda a: np.quantile(a, 0.95))



apps.imshow_tk(V_spots_pool,
               additional_imgs=[V_spots_pool>li_thresh_val,
               V_spots_pool>multiotsu_thresh_vals[-1]])

# Positive control mask
PC_mask = (V_spots_pool>multiotsu_thresh_vals[-1]).max(axis=0)
segm_npy_PC = segm_npy_pool.copy()
segm_npy_PC[~PC_mask] = 0
skimage.morphology.remove_small_objects(segm_npy_PC, min_size=5,
                                        in_place=True)
apps.imshow_tk(segm_npy_PC)

# fig, ax = skimage.filters.try_all_threshold(V_spots_pool.max(axis=0))
#
# plt.show()

save = prompts.askyesno(title='Save?',
                        message='Do you want to save pooled images info?')

if save:
    PC_rp = skimage.measure.regionprops(segm_npy_PC)
    areas = [obj.area for obj in PC_rp]
    PC_incr_IDs = [obj.label for obj in PC_rp]
    PC_IDs = [0]*len(PC_incr_IDs)
    pos = ['0']*len(PC_incr_IDs)
    for i, PC_increm_ID in enumerate(PC_incr_IDs):
        for p, (increm_ID, IDs) in enumerate(all_IDs):
            if PC_increm_ID in increm_ID:
                idx = list(increm_ID).index(PC_increm_ID)
                PC_IDs[i] = IDs[idx]
                pos[i] = pos_foldernames[p]
    df = (pd.DataFrame({'PC_IDs': PC_IDs,
                        'areas': areas,
                        'Position_n': pos})
            .sort_values('areas', ascending=False)).set_index('PC_IDs')
    csv_path = os.path.join(TIFFs_path, 'Positive_control_analysis_info.csv')
    df.to_csv(csv_path)
