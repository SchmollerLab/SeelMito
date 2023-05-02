import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats

SAVE = False

sns.set_theme()
sns.set(font_scale=1.3)

pwd_path = os.path.dirname(os.path.abspath(__file__))

tables_path = os.path.join(pwd_path, 'tables')

WT_filename = 'WT_stain_index.csv'
nc_filename = 'negative_control_stain_index.csv'

df_WT = pd.read_csv(os.path.join(tables_path, WT_filename))
df_WT['category'] = df_WT['medium']

df_nc = pd.read_csv(os.path.join(tables_path, nc_filename))
df_nc['category'] = 'Neg. control'

import pdb; pdb.set_trace()

cols = [
    'category', 'mKate_stain_index', 
    'mNeon_inside_cell_stain_index', 
    'mNeon_inside_mtNet_stain_index'
]
df = pd.concat([df_WT[cols], df_nc[cols]]).reset_index()

fig, ax = plt.subplots(1, 2, figsize=(10,6))
fig.subplots_adjust(
    left=0.08, bottom=0.08, right=0.96, top=0.95
)

palette = {
    'SCD': (229/255,56/255,33/255), 
    'SCGE': (64/255,169/255,224/255), 
    'Neg. control': 'g'
}

sns.boxplot(x='category', y='mKate_stain_index', data=df, ax=ax[0], palette=palette)
sns.boxplot(x='category', y='mNeon_inside_cell_stain_index', data=df, ax=ax[1], palette=palette)
# sns.boxplot(x='category', y='mNeon_inside_mtNet_stain_index', data=df, ax=ax[2])

ax[0].set_ylabel('mKate stain index')
ax[1].set_ylabel('mNeon stain index')
# ax[2].set_ylabel('mNeon stain index (inside mtNetwork)')
for axes in ax:
    axes.set_xlabel('')

if SAVE:
    filename = 'spotMAX_SNR_SI.svg'
    filepath = os.path.join(pwd_path, filename)
    fig.savefig(filepath)

plt.show()

