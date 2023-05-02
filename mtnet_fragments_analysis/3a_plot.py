import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import matplotlib

matplotlib.rc('font', **{'size': 14})

SAVE = False

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

df_WT = pd.read_csv(os.path.join(tables_path, 'WT_mtNet_num_fragments.csv'))
df_petite = pd.read_csv(os.path.join(tables_path, 'petite_mtNet_num_fragments.csv'))

df_summary_WT = df_WT[['num_fragments', 'threshold_value']].groupby('threshold_value').describe()
df_summary_WT.columns = ['_'.join(col) for col in df_summary_WT.columns.values]

# Add standard error
df_summary_WT['num_fragments_sem'] = df_summary_WT['num_fragments_std']/np.sqrt(df_summary_WT['num_fragments_count'])
df_summary_WT = df_summary_WT.reset_index()

df_summary_petite = df_petite[['num_fragments', 'threshold_value']].groupby('threshold_value').describe()
df_summary_petite.columns = ['_'.join(col) for col in df_summary_petite.columns.values]
df_summary_petite = df_summary_petite.reset_index()

# Add standard error
df_summary_petite['num_fragments_sem'] = df_summary_petite['num_fragments_std']/np.sqrt(df_summary_petite['num_fragments_count'])

ci_y_min_WT = df_summary_WT['num_fragments_25%'] # df_summary_WT.num_fragments_min # df_summary_WT.num_fragments_mean-1.96*df_summary_WT['num_fragments_sem']
ci_y_max_WY = df_summary_WT['num_fragments_75%'] # df_summary_WT.num_fragments_max # df_summary_WT.num_fragments_mean+1.96*df_summary_WT['num_fragments_sem']

ci_y_min_petite = df_summary_petite['num_fragments_25%']
ci_y_max_petite = df_summary_petite['num_fragments_75%']

fig, ax = plt.subplots(figsize=(12,10))

fig.subplots_adjust(
    left=0.1, bottom=0.1, right=0.98, top=0.95
)

ax.plot(df_summary_WT.threshold_value, df_summary_WT['num_fragments_mean'], label='WT')
ax.plot(df_summary_petite.threshold_value, df_summary_petite['num_fragments_mean'], label='petite')

ax.fill_between(
    df_summary_WT.threshold_value, ci_y_min_WT, ci_y_max_WY,
    alpha=0.3
)

ax.fill_between(
    df_summary_petite.threshold_value, ci_y_min_petite, ci_y_max_petite,
    alpha=0.3
)

ax.legend()

ax.set_xlim((10, 260))
ax.set_ylim((-5, 100))

ax.set_xlabel('Threshold value')
ax.set_ylabel('Number of mitochondrial network fragments')

ax.xaxis.set_major_locator(MultipleLocator(25))

if SAVE:
    fig_path = os.path.join(pwd_path, 'mitochondria_num_fragments_petite_vs_WT_roubustness.svg')
    fig.savefig(fig_path)

# ax.set_yscale('log')

plt.show()