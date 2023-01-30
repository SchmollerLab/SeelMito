import os
import sys
import subprocess
import re
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from tkinter import ttk
from tkinter.messagebox import askyesno
from natsort import natsorted
from tqdm import tqdm

script_dirpath = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_dirpath))
sys.path.insert(0, src_path)

import apps, prompts, load

class listbox_selector:
    def __init__(self, label_txt, lb_items, win_title='Listbox'):
        root = tk.Tk()
        root.geometry('+800+400')
        root.title(win_title)
        root.lift()
        root.attributes("-topmost", True)
        tk.Label(root,
                 text=label_txt,
                 font=(None, 11)
                 ).grid(row=0, column=0, pady=(10, 5), padx=10)

        width = max([len(item) for item in lb_items])+5

        self.lb = tk.Listbox(root,
                             width=width)
        self.lb.grid(row=1, column=0, pady=(0,10))

        for i, item in enumerate(lb_items):
            self.lb.insert(i+1, item)

        ttk.Button(root, text='   Ok   ', command=self._close
                  ).grid(row=2, column=0, pady=(0,10))

        root.protocol("WM_DELETE_WINDOW", self._abort)
        self.root = root
        root.mainloop()

    def _close(self):
        self.lb_selection = self.lb.get(self.lb.curselection())
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

# expand dataframe beyond page width in the terminal
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', 300)
# pd.set_option('display.precision', 3)
# pd.set_option('display.expand_frame_repr', False)

# Select experiment path
src_listdir = os.listdir(src_path)
main_idx = [i for i, f in enumerate(src_listdir) if f.find('main_') !=- 1][0]
main_filename = src_listdir[main_idx]
NUM = re.findall('v(\d+).py', main_filename)[0]
vNUM = f'v{NUM}'
run_num = prompts.single_entry_messagebox(
    entry_label='Analysis run number: ', input_txt='1', toplevel=False
).entry_txt

h5_name = listbox_selector('Select .h5 file name to concatenate:',
                           ['0_Orig_data', '1_ellip_test_data',
                            '2_p-_test_data', '3_p-_ellip_test_data',
                            '4_spotFIT_data'],
                            win_title='Select file name').lb_selection

h5_filename = f'{run_num}_{h5_name}_{vNUM}.h5'

selected_path = prompts.folder_dialog(
    title='Select folder with multiple experiments, the TIFFs folder or '
    'a specific Position_n folder'
)

if not selected_path:
    exit('Execution aborted.')

selector = load.select_exp_folder()

(main_paths, prompts_pos_to_analyse, run_num, tot,
is_pos_path, is_TIFFs_path) = load.get_main_paths(selected_path, vNUM)

TIFFs_paths = main_paths

dfs = []
keys = []
for TIFFs_path in TIFFs_paths:
    exp_path = os.path.dirname(TIFFs_path)
    TIFFs_path = os.path.join(exp_path, 'TIFFs')
    pos_filenames = [
        p for p in os.listdir(TIFFs_path)
        if p.find('Position_')!=-1
        and os.path.isdir(os.path.join(TIFFs_path, p))
    ]
    # print(f'Loading experiment {os.path.dirname(TIFFs_path)}...')
    for pos in tqdm(pos_filenames, ncols=100):
        spotmax_out_path = os.path.join(TIFFs_path, pos, 'spotMAX_output')
        h5_path = os.path.join(spotmax_out_path, h5_filename)
        if os.path.exists(h5_path):
            try:
                df_h5 = pd.read_hdf(h5_path, key='frame_0')
                dfs.append(df_h5)
                keys.append(pos)
            except Exception as e:
                traceback.print_exc()
                pass

df = pd.concat(dfs, keys=keys, names=['Position_n'])

name = prompts.single_entry_messagebox(
    entry_label='Name to prepend to file name: ',
    input_txt='WT',
    toplevel=False
).entry_txt

save_to = prompts.folder_dialog(title=
    'Select where to save the .csv file'
)

csv_path = os.path.join(save_to, f'{name}_{h5_name}_allPos.csv')
df.to_csv(csv_path)

print(f'CSV file saved to "{csv_path}"')

print(df.shape)
