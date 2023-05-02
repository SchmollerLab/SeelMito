import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rc('font', **{'size': 14})

from cellacdc import plot

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(pwd_path, 'tables')

media = ['SCD', 'SCGE']
ploidies = ['Diploid', 'Haploid']
p_vals = [0.001, 0.025, 0.05]

colors = {
    'SCGE_Diploid': (41,75,151),
    'SCGE_Haploid': (64,169,224),
    'SCD_Diploid': (122,23,19),
    'SCD_Haploid': (229,56,33)
}
markers = {
    'SCGE_Diploid': 'D',
    'SCGE_Haploid': 'o',
    'SCD_Diploid': 's',
    'SCD_Haploid': '^'
}

fig, ax = plt.subplots(1,3, figsize=(18,6), sharey=True, layout=None)

fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.98, top=0.93, wspace=0.05
)

for medium in media:
    for ploidy in ploidies:
        for col, p_val in enumerate(p_vals):
            if p_val == 0.001:
                pval_string = '_low_p_val_'
            elif p_val == 0.05:
                pval_string = '_high_p_val_'
            else:
                pval_string = '_default_p_val_'
            
            df_filename = f'WT_{medium}_{ploidy}{pval_string}3_AllExp_p-_ellip_test_TOT_data.csv'
            df_filepath = os.path.join(tables_path, df_filename)

            df = pd.read_csv(df_filepath)
            df = df[df.cell_vol_fl<=800]

            axes = ax[col]

            r, g, b = colors[f'{medium}_{ploidy}']
            c = [(r/255, g/255, b/255) for _ in range(len(df))]
            axes.scatter(
                df.cell_vol_fl, df.num_spots, 
                c=c, alpha=0.3, s=11, marker=markers[f'{medium}_{ploidy}']
            )

            plot.binned_means_plot(
                x='cell_vol_fl',
                y='num_spots',
                data=df,
                ax=axes, 
                scatter=False,
                bins_min_count=10,
                bins=np.arange(25,451,50),
                color=c[0]
            )
            axes.set_xlabel('Cell volume [fL]')
            if col == 0:
                axes.set_ylabel('Number of nucleoids')
            else:
                # axes.set_yticklabels([])
                axes.set_ylabel('')

            axes.set_title(f'p-value = {p_val}')

if SAVE:
    fig_path = os.path.join(pwd_path, 'low_normal_high_p_value_spotmax_robustness.png')
    fig.savefig(fig_path, transparent=True)

plt.show()