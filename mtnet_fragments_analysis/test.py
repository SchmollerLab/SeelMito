import os

import pandas as pd
import numpy as np

import skimage.io

import matplotlib.pyplot as plt

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

data_path = r'G:\My Drive\1_MIA_Data\Anika'
petite_data_path = os.path.join(data_path, 'Mutants', 'Petite')

images_path = os.path.join(
    petite_data_path, '2020-10-09_ASY39-1_2', 'SCD_0nM', 'TIFFs', 
    'Position_8', 'Images'
)

for file in os.listdir(images_path):
    file_path = os.path.join(images_path, file)
    if file.endswith('mKate.tif'):
        mKate_data = skimage.io.imread(file_path)
    elif file.endswith('.npz') and file.find('mKate_mask')!=-1:
        mKate_mask = np.load(file_path)['arr_0']

fig, ax = plt.subplots(1, 2)

ax[0].imshow(mKate_data.max(axis=0))
ax[1].imshow(mKate_mask.max(axis=0))
plt.show()