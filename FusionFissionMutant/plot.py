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
    tables_path, '1_repl_3_AllPos_p-_ellip_test_TOT_data.csv'
)
df2_path = os.path.join(
    tables_path, '2_repl_3_AllPos_p-_ellip_test_TOT_data.csv'
)
df3_path = os.path.join(
    tables_path, '3_repl_3_AllPos_p-_ellip_test_TOT_data.csv'
)

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)
df3 = pd.read_csv(df3_path)

df = pd.concat([df1, df2, df3], keys=[1,2,3], names=['replicate_num'])

# Remove number of nucleoids 0 --> outliers
df = df[df['num_spots']>0]

df['strain'] = df['original_filename'].str.split('-', expand=True).iloc[:, 0]

print(df[['strain', 'cell_vol_fl',  'num_spots']].groupby('strain').describe())

df['genotype'] = 'WT'
df.loc[df.strain == 'yC344', 'genotype'] = 'mutant'

print(df[['genotype', 'strain']])

colors = sns.color_palette(n_colors=2)

fig, ax = plt.subplots(1,2, figsize=(18,10))
ax = ax.flatten()

fig.subplots_adjust(
    left=0.05, right=0.92, bottom=0.1, top=0.95
)

'------------------------------------------------------------------------------'
ax_idx = 0
bins = np.arange(25,200,35)
scatter_kws={'s': 5}
x = 'cell_vol_fl'
y = 'num_spots'

df_WT = df[df.strain=='yC280']

print(f'Number of WT cells = {len(df_WT)}')

plot.binned_means_plot(
    x=x, y=y, data=df_WT, ax=ax[ax_idx], 
    color=colors[0], bins=bins, label='WT',
    scatter_kws=scatter_kws
)

df_mutant = df[df.strain=='yC344']

print(f'Number of mutant cells = {len(df_mutant)}')

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


# '------------------------------------------------------------------------------'
# ax_idx = 2
# bins = np.arange(25,75,15)
# x = 'ref_ch_vol_um3'
# y = 'num_spots'

# plot.binned_means_plot(
#     x=x, y=y, data=df_WT, ax=ax[ax_idx], 
#     color=colors[0], bins=bins, label='WT',
#     scatter_kws=scatter_kws
# )

# plot.binned_means_plot(
#     x=x, y=y, data=df_mutant, ax=ax[ax_idx], 
#     color=colors[1], bins=bins, label='mutant',
#     scatter_kws=scatter_kws
# )
# '------------------------------------------------------------------------------'

legend_handles = []
for s, label in enumerate(df['genotype'].unique()):
    legend_handles.append(
        mpatches.Patch(color=colors[s], label=label)
    )
fig.legend(handles=legend_handles, loc='center right')

'******************************************************************************'

plt.show()