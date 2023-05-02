import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import matplotlib

matplotlib.rc('font', **{'size': 14})

SAVE = True
SHOW_FULL_RANGE = False

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

df_WT = pd.read_csv(os.path.join(tables_path, 'WT_mtNet_num_fragments_with_cell_vol.csv'))
df_petite = pd.read_csv(os.path.join(tables_path, 'petite_mtNet_num_fragments_with_cell_vol.csv'))

df_WT = df_WT[df_WT.num_fragments > 0]
df_petite = df_petite[df_petite.num_fragments > 0]

# Remove those 4 outliers where there was no mKate signal at all
df_petite = df_petite[df_petite.spotmax_threshold_value>15]

vol_ranges = [(0,100), (100, 200), (200,300)]

min_spotmax_threshold_WT = df_WT.spotmax_threshold_value.min()
max_spotmax_threshold_WT = df_WT.spotmax_threshold_value.max()
median_spotmax_threshold_WT = df_WT.spotmax_threshold_value.median()

min_spotmax_threshold_petite = df_petite.spotmax_threshold_value.min()
max_spotmax_threshold_petite = df_petite.spotmax_threshold_value.max()
median_spotmax_threshold_petite = df_petite.spotmax_threshold_value.median()

min_thresh_range = int(min(min_spotmax_threshold_WT, min_spotmax_threshold_petite))
max_thresh_range = int(max(max_spotmax_threshold_WT, max_spotmax_threshold_petite))

print(df_petite.sort_values('spotmax_threshold_value').head(100))

df_WT_petite = pd.concat([df_WT, df_petite], keys=('WT', 'Petite'), names=['strain']).reset_index()
sns.boxplot(x='strain', y='spotmax_threshold_value', data=df_WT_petite)
plt.show()

nrows = 2 if SHOW_FULL_RANGE else 1
height = 10 if SHOW_FULL_RANGE else 6
fig, ax = plt.subplots(nrows, 3, figsize=(18,height))
if nrows == 1:
    ax = ax[np.newaxis]

fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.98, top=0.93
)

colors = sns.color_palette(n_colors=2)

for a, (vol_min, vol_max) in enumerate(vol_ranges):
    df_WT_volrange = df_WT[(df_WT.cell_vol_fl >= vol_min) & (df_WT.cell_vol_fl < vol_max)]
    df_petite_volrange = df_petite[(df_petite.cell_vol_fl >= vol_min) & (df_petite.cell_vol_fl < vol_max)]

    df_summary_WT = df_WT_volrange[['num_fragments', 'threshold_value']].groupby('threshold_value').describe()
    df_summary_WT.columns = ['_'.join(col) for col in df_summary_WT.columns.values]

    # Add standard error
    df_summary_WT['num_fragments_sem'] = df_summary_WT['num_fragments_std']/np.sqrt(df_summary_WT['num_fragments_count'])
    df_summary_WT = df_summary_WT.reset_index()

    df_summary_petite = df_petite_volrange[['num_fragments', 'threshold_value']].groupby('threshold_value').describe()
    df_summary_petite.columns = ['_'.join(col) for col in df_summary_petite.columns.values]
    df_summary_petite = df_summary_petite.reset_index()

    # Add standard error
    df_summary_petite['num_fragments_sem'] = df_summary_petite['num_fragments_std']/np.sqrt(df_summary_petite['num_fragments_count'])

    ci_y_min_WT = df_summary_WT['num_fragments_25%'] # df_summary_WT.num_fragments_min # df_summary_WT.num_fragments_mean-1.96*df_summary_WT['num_fragments_sem']
    ci_y_max_WY = df_summary_WT['num_fragments_75%'] # df_summary_WT.num_fragments_max # df_summary_WT.num_fragments_mean+1.96*df_summary_WT['num_fragments_sem']

    ci_y_min_petite = df_summary_petite['num_fragments_25%']
    ci_y_max_petite = df_summary_petite['num_fragments_75%']

    for row in range(nrows):
        axes = ax[row, a]
        axes.plot(
            df_summary_WT.threshold_value, df_summary_WT['num_fragments_mean'], 
            label='WT', color=colors[0]
        )
        axes.plot(
            df_summary_petite.threshold_value, 
            df_summary_petite['num_fragments_mean'], 
            label='petite', color=colors[1]
        )

        axes.fill_between(
            df_summary_WT.threshold_value, ci_y_min_WT, ci_y_max_WY,
            alpha=0.3, color=colors[0]
        )

        axes.fill_between(
            df_summary_petite.threshold_value, ci_y_min_petite, ci_y_max_petite,
            alpha=0.3, color=colors[1]
        )

        axes.legend()

        axes.set_xlim((min_thresh_range-5, 260))
        axes.set_ylim((-5, 100))

        axes.set_xlabel('Threshold value')
        axes.set_ylabel('Number of mitochondrial network fragments')

        if row == 0:
            axes.set_title(f'Cell volume range = ({vol_min}, {vol_max}) fL')
            axes.xaxis.set_major_locator(MultipleLocator(25))
        else:
            axes.xaxis.set_major_locator(MultipleLocator(5))

        axes.axvspan(
            min_spotmax_threshold_WT, max_spotmax_threshold_WT, 
            alpha=0.2, color=colors[0]
        )
        axes.axvspan(
            min_spotmax_threshold_petite, max_spotmax_threshold_petite, 
            alpha=0.2, color=colors[1]
        )

        axes.axvline(
            median_spotmax_threshold_WT, 
            color=colors[0], linestyle='--'
        )

        axes.axvline(
            median_spotmax_threshold_petite, 
            color=colors[1], linestyle='--'
        )

        if row > 0 or not SHOW_FULL_RANGE:
            axes.set_xlim((min_thresh_range-5, max_thresh_range+5))
            axes.set_ylim((-5, 20))
            axes.set_xticks(np.arange(15, max_thresh_range+5, 5))

if SAVE:
    fig_path = os.path.join(pwd_path, 'mitochondria_num_fragments_petite_vs_WT_roubustness.png')
    fig.savefig(fig_path, transparent=True)

plt.show()