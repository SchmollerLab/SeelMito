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

df_SCGE.to_csv(os.path.join(tables_path, 'Mic60_del_SCGE_spotMAX_data.csv'))