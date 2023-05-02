import os

import pandas as pd
import numpy as np

SAVE = False

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

df_WT = (
    pd.read_csv(os.path.join(tables_path, 'WT_mtNet_num_fragments.csv'))
    .set_index(['replicate_num', 'exp_folder', 'Position_n', 'Cell_ID'])
    .sort_index()
)
df_petite = (
    pd.read_csv(os.path.join(tables_path, 'petite_mtNet_num_fragments.csv'))
    .set_index(['replicate_num', 'exp_folder', 'Position_n', 'Cell_ID'])
    .sort_index()
)

df_WT_spotmax = (
    pd.read_csv(os.path.join(tables_path, 'spotMAX_Anika_WT_SCD_Haploid_TOT_data.csv'))
    .set_index(['replicate_num', 'horm_conc', 'Position_n', 'Moth_ID'])
    .sort_index()
)
df_WT_spotmax.index.names = ['replicate_num', 'exp_folder', 'Position_n', 'Cell_ID']
df_petite_spotmax = (
    pd.read_csv(os.path.join(tables_path, 'spotMAX_Anika_Petite_TOT_data.csv'))
    .set_index(['replicate_num', 'horm_conc', 'Position_n', 'Moth_ID'])
    .sort_index()
)
df_petite_spotmax.index.names = ['replicate_num', 'exp_folder', 'Position_n', 'Cell_ID']

# Get the intersection of the indexes --> when we had more than 3 cells in the 
# image we skipped the position when calculating number of fragments
# which means this Position is in the spotmax data but not in the fragnments dfs
idx_WT = df_WT_spotmax.index.intersection(df_WT.index)
idx_petite = df_petite_spotmax.index.intersection(df_petite.index)

# Keep only mother cells
df_WT = df_WT.loc[idx_WT]
df_petite = df_petite.loc[idx_petite]
df_WT_spotmax = df_WT_spotmax.loc[idx_WT]
df_petite_spotmax = df_petite_spotmax.loc[idx_petite]

df_WT['cell_vol_fl'] = df_WT_spotmax.loc[df_WT.index, 'cell_vol_fl']
df_petite['cell_vol_fl'] = df_petite_spotmax.loc[df_petite.index, 'cell_vol_fl']

print(df_WT.head(10))

df_WT.to_csv(os.path.join(tables_path, 'WT_mtNet_num_fragments_with_cell_vol.csv'))
df_petite.to_csv(os.path.join(tables_path, 'petite_mtNet_num_fragments_with_cell_vol.csv'))