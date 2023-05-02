import os
import re
import json

import numpy as np
import pandas as pd

import scipy.stats

import re

from tqdm import tqdm

import utils
from utils import printl

import matplotlib.pyplot as plt
import seaborn as sns

from cellacdc import plot

import matplotlib

matplotlib.rc('font', **{'size': 14})


SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

spotmax_filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')

y_col_mitog = 'mitograph_volume_from_voxels' # 'mitograph_volume_from_length_um3', 'mitograph_volume_from_voxels'
x_col = 'ref_ch_vol_um3'

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

bins = {
    'SCGE_Diploid': np.arange(0,151,25),
    'SCGE_Haploid': np.arange(0,126,25),
    'SCD_Diploid': np.arange(25,101,25),
    'SCD_Haploid': np.arange(0,60,10)
}

fig, ax = plt.subplots(figsize=(10,8))
fig.subplots_adjust(
    left=0.08, bottom=0.08, right=0.96, top=0.95
)

for f, file in enumerate(utils.listdir(spotmax_filtered_tables_path)):
    spotmax_final_df_path = os.path.join(spotmax_filtered_tables_path, file)
    df = pd.read_csv(spotmax_final_df_path)
    key = re.findall(r'WT_(\w+)_TOT', file)[0]
    color = colors[key]
    marker = markers[key]
    r, g, b = color
    c = [(r/255, g/255, b/255) for _ in range(len(df))]

    ax.scatter(
        df[x_col], df[y_col_mitog], c=c, alpha=0.3, s=13, marker=marker
    )

    sns.regplot(x=x_col, y=y_col_mitog, data=df, ax=ax, scatter=False, color=c[0])

ax.set_xlabel('mtNetwork volume as computed in this paper [arb.u.]')
ax.set_ylabel('MitoGraph mtNetwork volume [fL]')
ax.set_xlim((0,250))

if SAVE:
    filename = 'spotMAX_vs_MitoGraph_SI.svg'
    filepath = os.path.join(pwd_path, filename)
    fig.savefig(filepath)

plt.show()

                

            

    


