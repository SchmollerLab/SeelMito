import numpy as np

import skimage.io

from utils import printl

import matplotlib.pyplot as plt

mkate_mask_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-02-18_Diploid_SCD_1\2020-02-18_ASY15-1_150nM\TIFFs\Position_42\Images\ASY15-1_150nM-42_s42_mKate_mask.npz"
mkate_data_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-02-18_Diploid_SCD_1\2020-02-18_ASY15-1_150nM\TIFFs\Position_42\Images\ASY15-1_150nM-42_s42_mKate.tif"
mitograph_mask = r"G:\My Drive\1_MIA_Data\Anika\MitoGraph\SCD\Diploids\2020-02-18_Diploid_SCD_1\200218_SCD_150nM_cells_new\C2-ASY15-1_150nM-42_041.png"

mkate_data = skimage.io.imread(mkate_data_path)
mkate_mask = np.load(mkate_mask_path)['arr_0']
mitograph_mask = skimage.io.imread(mitograph_mask)

vox_to_um3 =  0.35*0.06725*0.06725
spotmax_vol = np.count_nonzero(mkate_mask)*vox_to_um3
mitograph_vol = np.count_nonzero(mitograph_mask)*vox_to_um3

printl(
    f'spotMAX voxel volume = {spotmax_vol:.2f}\n'
    f'MitoGraph voxel volume = {mitograph_vol:.2f}'
)

fig, ax = plt.subplots(1,3)
ax[0].imshow(mkate_data.max(axis=0))
ax[1].imshow(mkate_mask.max(axis=0))
ax[2].imshow(mitograph_mask)
plt.show()