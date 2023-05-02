import os

import pandas as pd

pwd_path = os.path.dirname(os.path.abspath(__file__))

tables_path = os.path.join(pwd_path, 'tables')

WT_filename = 'WT_stain_index.csv'
nc_filename = 'negative_control_stain_index.csv'

df_WT = pd.read_csv(os.path.join(tables_path, WT_filename))
df_nc = pd.read_csv(os.path.join(tables_path, nc_filename))

df_WT = df_WT[df_WT.mKate_stain_index > 0] 
df_WT = df_WT[df_WT.mNeon_inside_cell_stain_index > 0] 
df_WT = df_WT[df_WT.mNeon_inside_mtNet_stain_index > 0] 

df_nc = df_WT[df_WT.mKate_stain_index > 0] 
df_nc = df_WT[df_WT.mNeon_inside_cell_stain_index > 0] 
df_nc = df_WT[df_WT.mNeon_inside_mtNet_stain_index > 0] 

df_WT.to_csv(os.path.join(tables_path, WT_filename))
df_nc.to_csv(os.path.join(tables_path, nc_filename))