import os

import numpy as np
import skimage.io

import matplotlib.pyplot as plt

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')
data_path = os.path.join(pwd_path, 'data')

images_path = os.path.join(data_path, 'SCD', '2022-12-14', 'TIFFs', 'Position_6', 'Images')

mKate_mask_path = os.path.join(images_path, 'YFT006-06_s06_mKate_mask.npz')
mKate_mask = np.load(mKate_mask_path)['arr_0']

mKate_path = os.path.join(images_path, 'YFT006-06_s06_mKate.tif')
mKate_data = skimage.io.imread(mKate_path)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(mKate_data.max(axis=0))
ax[1].imshow(mKate_mask.max(axis=0))
plt.show()
