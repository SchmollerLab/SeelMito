import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from cellacdc import plot

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'spotmax_final_tables')

SCD_WT_df_haploid_path = os.path.join(tables_path, 'spotMAX_Anika_WT_SCD_Haploid_TOT_data.csv')
SCD_WT_df_diploid_path = os.path.join(tables_path, 'spotMAX_Anika_WT_SCD_Diploid_TOT_data.csv')

SCGE_WT_df_haploid_path = os.path.join(tables_path, 'spotMAX_Anika_WT_SCGE_Haploid_TOT_data.csv')
SCGE_WT_df_diploid_path = os.path.join(tables_path, 'spotMAX_Anika_WT_SCGE_Diploid_TOT_data.csv')

cols = [['cell_vol_fl', 'num_spots', 'ref_ch_vol_um3', 'mtnet_skeleton_length_voxels', 'medium', 'strain', 'genotype', 'original_filename']]

SCD_WT_hapl_df = (
    pd.read_csv(SCD_WT_df_haploid_path) 
    # .query('mtnet_skeleton_length_voxels < 1000')
)

SCD_WT_dipl_df = (
    pd.read_csv(SCD_WT_df_haploid_path) 
    # .query('mtnet_skeleton_length_voxels < 1000')
)

SCGE_WT_hapl_df = (
    pd.read_csv(SCGE_WT_df_haploid_path) 
    # .query('mtnet_skeleton_length_voxels < 1000')
)

SCGE_WT_dipl_df = (
    pd.read_csv(SCGE_WT_df_diploid_path) 
    # .query('mtnet_skeleton_length_voxels < 1000')
)

colors = sns.color_palette('Paired', n_colors=4)

fig, ax = plt.subplots()
ax = [ax]
ax_idx = 0

labels = []
scatter_kws={'s': 5}
x = 'ref_ch_vol_um3' # 'ref_ch_vol_um3' # 'mtnet_skeleton_length_voxels'
y =  'num_spots' 

all_bins = {
    'ref_ch_vol_um3': [np.arange(15,51,10), np.arange(25,126,25)],
    'mtnet_skeleton_length_voxels': [np.arange(100,401,50), np.arange(250,1751,250)],
}

'''Mic60 SCD'''
bins = np.arange(15,36,5)

data = SCD_WT_hapl_df # .query('genotype == "WT"')
labels.append('SCD WT Haploid')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[0], bins=all_bins[x][0], label='WT',
    scatter_kws=scatter_kws
)

# bins = np.arange(0,21,4)
data = SCD_WT_dipl_df # .query('genotype == "mutant"')
labels.append('SCD WT Diploid')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[1], bins=all_bins[x][0], label='WT',
    scatter_kws=scatter_kws
)

'''Mic60 SCGE'''
data = SCGE_WT_hapl_df # .query('genotype == "WT"')
labels.append('SCGE WT Haploid')

plot.binned_means_plot(
    x=x, y=y, data=data, ax=ax[ax_idx], 
    color=colors[2], bins=all_bins[x][1], label='WT',
    scatter_kws=scatter_kws
)

# bins = np.arange(10,41,7)
data = SCGE_WT_dipl_df # .query('genotype == "mutant"')
labels.append('SCGE WT Diploid')

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

ax[ax_idx].set_title('WT experiment (Fig. 2)')

plt.show()