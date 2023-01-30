import pandas as pd
import tkinter as tk
import tkinter.filedialog

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

path = tkinter.filedialog.askopenfilename()

store = pd.HDFStore(path, mode='r')

df = store['frame_0']

store.close()

print(df.columns)
print(df.index)

print(df.loc[2, ['z', 'y', 'x', 'is_spot_inside_ref_ch']])
