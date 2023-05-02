import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from cellacdc import plot

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

df1_path = os.path.join(
    tables_path, 'repl_01_221111_Mic60_3_AllPos_p-_ellip_test_TOT_data.csv'
)
df2_path = os.path.join(
    tables_path, 'repl_02_221116_Mic60_3_AllPos_p-_ellip_test_TOT_data.csv'
)
df3_path = os.path.join(
    tables_path, 'repl_03_221117_Mic60_3_AllPos_p-_ellip_test_TOT_data.csv'
)

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)
df3 = pd.read_csv(df3_path)

df_SCGE = pd.concat([df1, df2, df3], keys=['2022-11-11','2022-11-16','2022-11-17'], names=['replicate'])

print(df_SCGE.sort_values('cell_vol_fl')[['Position_n', 'Moth_ID', 'cell_vol_fl']])

df_SCD_path = os.path.join(
    tables_path, 'Mic60_del_SCD_3_AllExp_p-_ellip_test_TOT_data.csv'
)
df_SCD = pd.read_csv(df_SCD_path)

for df in (df_SCGE, df_SCD):
    df['strain'] = df['original_filename'].str.split('-', expand=True).iloc[:, 0]

    print(df[['strain', 'cell_vol_fl',  'num_spots']].groupby('strain').describe())

    df['genotype'] = 'WT'
    df.loc[df.strain == 'YFT023', 'genotype'] = 'mutant'

    print(df[['genotype', 'strain']])

df_SCGE.to_csv(os.path.join(tables_path, 'Mic60_del_SCGE_spotMAX_data.csv'))
df_SCD.to_csv(os.path.join(tables_path, 'Mic60_del_SCD_spotMAX_data.csv'))

colors = sns.color_palette(n_colors=2)

"""SCGE plots"""
df = df_SCGE

fig, ax = plt.subplots(2,3, figsize=(18,10))
ax = ax.flatten()

fig.subplots_adjust(
    left=0.05, right=0.92, bottom=0.1, top=0.95
)

'------------------------------------------------------------------------------'
ax_idx = 0
bins = np.arange(50,200,25)
scatter_kws={'s': 5}
x = 'cell_vol_fl'
y = 'num_spots'

df_WT = df[df.strain=='YFT006']

print(f'Number of WT cells SCGE = {len(df_WT)}')

plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

df_mutant = df[df.strain=='YFT023']

print(f'Number of mutant cells SCGE = {len(df_mutant)}')

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

'------------------------------------------------------------------------------'
ax_idx = 1
x = 'cell_vol_fl'
y = 'ref_ch_vol_um3'


plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

'------------------------------------------------------------------------------'
ax_idx = 2
bins = np.arange(0,50,10)
x = 'ref_ch_vol_um3'
y = 'num_spots'

plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

legend_handles = []
for s, label in enumerate(df['genotype'].unique()):
    legend_handles.append(
        mpatches.Patch(color=colors[s], label=label)
    )
fig.legend(handles=legend_handles, loc='center right')

'******************************************************************************'

"""SCD plots"""
df = df_SCD

fig.subplots_adjust(
    left=0.05, right=0.92, bottom=0.1, top=0.95
)

'------------------------------------------------------------------------------'
ax_idx = 3
bins = np.arange(50,350,50)
scatter_kws={'s': 5}
x = 'cell_vol_fl'
y = 'num_spots'

df_WT = df[df.strain=='YFT006']

print(f'Number of WT cells SCD = {len(df_WT)}')

plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

df_mutant = df[df.strain=='YFT023']

print(f'Number of mutant cells SCD = {len(df_mutant)}')

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

'------------------------------------------------------------------------------'
ax_idx = 4
x = 'cell_vol_fl'
y = 'ref_ch_vol_um3'


plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

'------------------------------------------------------------------------------'
ax_idx = 5
bins = np.arange(0,50,10)
x = 'ref_ch_vol_um3'
y = 'num_spots'

plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

plot.binned_means_plot(
    x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
    color=colors[1], bins=bins, label='mutant',
    scatter_kws=scatter_kws
)
'------------------------------------------------------------------------------'

# legend_handles = []
# for s, label in enumerate(df['genotype'].unique()):
#     legend_handles.append(
#         mpatches.Patch(color=colors[s], label=label)
#     )
# fig.legend(handles=legend_handles, loc='center right')


plt.show()