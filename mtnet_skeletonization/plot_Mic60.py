import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from cellacdc import plot

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'spotmax_final_tables')

SCD_Mic60_df_path = os.path.join(tables_path, 'Mic60_del_SCD_spotMAX_data.csv')
SCGE_Mic60_df_path = os.path.join(tables_path, 'Mic60_del_SCGE_spotMAX_data.csv')

cols = [['cell_vol_fl', 'num_spots', 'ref_ch_vol_um3', 'mtnet_skeleton_length_voxels', 'medium', 'strain', 'genotype', 'original_filename']]

SCD_Mic60_df = (
    pd.read_csv(SCD_Mic60_df_path) 
    .query('mtnet_skeleton_length_voxels < 1000')
)

SCGE_Mic60_df = (
    pd.read_csv(SCGE_Mic60_df_path) 
    .query('mtnet_skeleton_length_voxels < 1000')
)

colors = sns.color_palette('Paired', n_colors=4)

fig, ax = plt.subplots()
ax = [ax]
ax_idx = 0

labels = []
scatter_kws={'s': 5}
x = 'mtnet_skeleton_length_voxels' # 'ref_ch_vol_um3' # 'mtnet_skeleton_length_voxels'
y =  'num_spots' 

all_bins = {
    'ref_ch_vol_um3': [np.arange(15,36,5), np.arange(25,54,7)],
    'mtnet_skeleton_length_voxels': [np.arange(100,351,50), np.arange(250,701,100)],
}

'''Mic60 SCD'''
bins = np.arange(15,36,5)

data = SCD_Mic60_df.query('genotype == "WT"')
labels.append('SCD WT')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[0], bins=all_bins[x][0], label='WT',
    scatter_kws=scatter_kws
)

# bins = np.arange(0,21,4)
data = SCD_Mic60_df.query('genotype == "mutant"')
labels.append('SCD Mic60-del')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[1], bins=all_bins[x][0], label='WT',
    scatter_kws=scatter_kws
)

'''Mic60 SCGE'''
bins = np.arange(25,54,7)
data = SCGE_Mic60_df.query('genotype == "WT"')
labels.append('SCGE WT')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[2], bins=all_bins[x][1], label='WT',
    scatter_kws=scatter_kws
)

# bins = np.arange(10,41,7)
data = SCGE_Mic60_df.query('genotype == "mutant"')
labels.append('SCGE Mic60-del')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[3], bins=all_bins[x][1], label='WT',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

legend_handles = []
for s, label in enumerate(labels):
    legend_handles.append(
        mpatches.Patch(color=colors[s], label=label)
    )
ax[ax_idx].legend(handles=legend_handles, loc='lower right')

'******************************************************************************'

ax[ax_idx].set_title('Mic60 experiment')

plt.show()