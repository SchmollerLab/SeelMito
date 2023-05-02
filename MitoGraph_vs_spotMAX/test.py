import numpy as np

import skimage.io

import matplotlib.pyplot as plt

# mkate_mask_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-04-30_Diploid_SCD_3\2020-04-30_ASY15_SCD_15nM\TIFFs\Position_9\Images\ASY15-1_15nM-09_s09_mKate_mask.npz"
# mkate_data_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-04-30_Diploid_SCD_3\2020-04-30_ASY15_SCD_15nM\TIFFs\Position_9\Images\ASY15-1_15nM-09_s09_mKate.tif"

# mkate_data = skimage.io.imread(mkate_data_path)
# mkate_mask = np.load(mkate_mask_path)['arr_0']

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(mkate_data.max(axis=0))
# ax[1].imshow(mkate_mask.max(axis=0))
# plt.show()

mkate_mask_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-04-30_Diploid_SCD_3\2020-04-30_ASY15_SCD_15nM\TIFFs\Position_10\Images\ASY15-1_15nM-10_s10_mKate_mask.npz"
mkate_data_path = r"G:\My Drive\1_MIA_Data\Anika\WTs\SCD\Diploids\2020-04-30_Diploid_SCD_3\2020-04-30_ASY15_SCD_15nM\TIFFs\Position_10\Images\ASY15-1_15nM-10_s10_mKate.tif"

mkate_data = skimage.io.imread(mkate_data_path)
mkate_mask = np.load(mkate_mask_path)['arr_0']

fig, ax = plt.subplots(1,2)
ax[0].imshow(mkate_data.max(axis=0))
ax[1].imshow(mkate_mask.max(axis=0))
plt.show()