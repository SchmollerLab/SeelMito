import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

SAVE = True

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

cols = [
    'category', 'mKate_stain_index', 
    'mNeon_inside_cell_stain_index'
]

print(df_WT.columns)

df_WT = df_WT.set_index(['medium', 'ploidy', 'replicate', 'exp_folder', 'Position_n', 'Cell_ID'])[cols]
df_nc = df_nc.set_index(['exp_folder', 'Position_n', 'Cell_ID'])[cols]

print(df_WT[df_WT.isna().any(axis=1)])

df_WT = df_WT.dropna()
df_nc = df_nc.dropna()

df_WT.to_csv(os.path.join(tables_path, 'spotMAX_SNR_WT_stain_index.csv'))
df_nc.to_csv(os.path.join(tables_path, 'spotMAX_SNR_neg_control_stain_index.csv'))