import os
import re
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import utils
from utils import printl

import matplotlib.pyplot as plt
import seaborn as sns

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

sns.set_theme()

data_cols = [
    'replicate_num', 'horm_conc', 'Position_n', 'Moth_ID', 'RoiSet_filename', 
    'roi_name', 'mitograph_foldername', 'mitograph_filename', 'cell_vol_fl',
    'ref_ch_vol_um3', 'mitograph_volume_from_voxels'
    ]

def pointsClicked(s, points, event):
    point = points[0]
    idx = point.data()
    print('='*40)
    print(s.df.loc[idx, data_cols])
    print('*'*40)

backend = 'sns' # 'pg', 'sns'

SAVE = True

pwd_path = os.path.dirname(os.path.abspath(__file__))

spotmax_filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')

y_col_mitog = 'mitograph_volume_from_length_um3' # 'mitograph_volume_from_length_um3', 'mitograph_volume_from_voxels'

colors = sns.color_palette(n_colors=1)
if backend == 'pg':
    app = pg.mkQApp("Interactive scatter plots") 
    mw = QtWidgets.QMainWindow()
    mw.resize(800,800)
    view = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle(f'MitoGraph vs spotMAX')
else:
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
for f, file in enumerate(utils.listdir(spotmax_filtered_tables_path)):
    spotmax_final_df_path = os.path.join(spotmax_filtered_tables_path, file)
    df = pd.read_csv(spotmax_final_df_path)
    if backend == 'pg':
        xx_col = 'ref_ch_vol_um3'
        yy_col = y_col_mitog
        row, col = np.unravel_index(f, (2,2))
        w = view.addPlot(row=row, col=col)
        w.setLabel('bottom', xx_col)
        w.setLabel('left', yy_col)

        c = [int(round(v*255)) for v in colors[0]]
        c.append(150)
        s = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(c[:3]),
            brush=pg.mkBrush(c),
            hoverable=True,
            hoverPen=pg.mkPen('r', width=2),
            hoverBrush=pg.mkBrush((255,0,0,100))
        )
        s.df = df
        s.sigClicked.connect(pointsClicked)
        w.addItem(s)
        s.setData(
            df[xx_col], df[yy_col], data=df.index
        )
    else:
        sns.regplot(x='ref_ch_vol_um3', y=y_col_mitog, data=df, ax=ax[f])
        ax[f].set_title(file)

if backend == 'pg':
    pg.exec()
else:
    plt.show()

                

            

    


