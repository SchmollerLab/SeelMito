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

pwd_path = os.path.dirname(os.path.abspath(__file__))

spotmax_filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')

y_col_mitog = 'mitograph_volume_from_length_um3' # 'mitograph_volume_from_length_um3', 'mitograph_volume_from_voxels'
x_col = 'ref_ch_vol_um3'

for f, file in enumerate(utils.listdir(spotmax_filtered_tables_path)):
    spotmax_final_df_path = os.path.join(spotmax_filtered_tables_path, file)
    df = pd.read_csv(spotmax_final_df_path)
    key = re.findall(r'WT_(\w+)_TOT', file)[0]

    df = df.dropna()

    pearson = scipy.stats.pearsonr(df[x_col], df[y_col_mitog])
    print(f'{key} Pearson coeff = {pearson.statistic}')

                

            

    


