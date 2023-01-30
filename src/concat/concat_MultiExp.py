import os
import sys
import subprocess
import re
import traceback
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import askyesno
from natsort import natsorted

script_dirpath = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(script_dirpath)
sys.path.insert(0, src_path)

import load, prompts, apps

class beyond_listdir_pos:
    def __init__(self, folder_path, spotM_data_foldername):
        self.bp = apps.tk_breakpoint()
        self.folder_path = folder_path
        self.spotM_paths = []
        self.count_recursions = 0
        self.spotM_data_foldername = spotM_data_foldername
        self.walk_directories(folder_path)
        # self.listdir_recursion(folder_path)
        if not self.spotM_paths:
            raise FileNotFoundError(f'Path {folder_path} is not valid!')
        self.all_exp_info = self.count_analysed_pos()

    def walk_directories(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=True):
            # Avoid scanning TIFFs and CZIs folder
            dirs[:] = [d for d in dirs
                       if d not in ['TIFFs', 'Original_TIFFs', 'CZIs']]
            for dirname in dirs:
                path = f'{root}/{dirname}'
                listdir_folder = natsorted(os.listdir(path))
                if dirname == self.spotM_data_foldername:
                    self.spotM_paths.append(path)
                    break

    def listdir_recursion(self, folder_path):
        """DEPRECATED"""
        if os.path.isdir(folder_path):
            listdir_folder = natsorted(os.listdir(folder_path))
            contains_mitoQ_data = any([name==self.spotM_data_foldername
                                       for name in listdir_folder])
            rec_count_ok = self.count_recursions < 50
            if contains_mitoQ_data and rec_count_ok:
                self.spotM_paths.append(f'{folder_path}/'
                                        f'{self.spotM_data_foldername}')
            elif rec_count_ok:
                for name in listdir_folder:
                    subfolder_path = f'{folder_path}/{name}'
                    self.listdir_recursion(subfolder_path)
                self.count_recursions += 1
            else:
                raise RecursionError(
                      'Recursion went too deep and it was aborted'
                      'Check that the experiments contains the TIFFs folder')

    def get_rel_path(self, path):
        rel_path = ''
        parent_path = path
        count = 0
        while parent_path != self.folder_path or count==10:
            if count > 0:
                rel_path = f'{os.path.basename(parent_path)}/{rel_path}'
            parent_path = os.path.dirname(parent_path)
            count += 1
        rel_path = f'.../{rel_path}'
        return rel_path

    def count_analysed_pos(self):
        all_exp_info = []
        rel_paths = []
        for path in self.spotM_paths:
            rel_path = self.get_rel_path(path)
            foldername = os.path.basename(path)
            exp_info = f'{rel_path} (spotMAX data present!)'
            all_exp_info.append(exp_info)
            rel_paths.append(rel_path)
        self.rel_paths = rel_paths
        return all_exp_info

class single_entry_messagebox:
    def __init__(self, title='', entry_label='Entry 1', input_txt='',
                 toplevel=False, width=None):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.title(title)
        root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0, padx=8)
        w = len(input_txt)+10 if width is None else width
        e = tk.Entry(root, justify='center', width=w)
        e.grid(row=1, padx=16, pady=4)
        e.focus_force()
        e.insert(0, input_txt)
        tk.Button(root, command=self._quit, text='Ok!', width=10).grid(row=2,
                                                                      pady=4)
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.e = e
        root.mainloop()

    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        exit('Execution aborted by the user')

    def _quit(self, event=None):
        self.entry_txt = self.e.get()
        self._root.quit()
        self._root.destroy()

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)

# Select experiment path
src_listdir = os.listdir(src_path)
main_idx = [i for i, f in enumerate(src_listdir) if f.find('main_') !=- 1][0]
main_filename = src_listdir[main_idx]
NUM = re.findall('v(\d+).py', main_filename)[0]
vNUM = f'v{NUM}'
run_num = single_entry_messagebox(entry_label='Analysis run number: ',
                               input_txt='1', toplevel=False).entry_txt
spotM_data_foldername = f'spotMAX_{vNUM}_run-num{run_num}'
selected_path = prompts.folder_dialog(title = "Select folder containing valid experiments")
beyond_listdir_pos = beyond_listdir_pos(selected_path, spotM_data_foldername)
num_exp = len(beyond_listdir_pos.spotM_paths)
load.select_exp_folder().run_widget(beyond_listdir_pos.all_exp_info,
                         title='Concatenate experiments',
                         label_txt=f'There are {num_exp} analysed experiments:',
                         full_paths=beyond_listdir_pos.spotM_paths,
                         showinexplorer_button=True,
                         remaining_button=False)


# Iterate mitoQUANT data path and concatenate dataframes
ellips_test_df_moth_li = []
ellips_test_df_bud_li = []
ellips_test_df_tot_li = []
p_test_df_moth_li = []
p_test_df_bud_li = []
p_test_df_tot_li = []
p_ellips_test_df_moth_li = []
p_ellips_test_df_bud_li = []
p_ellips_test_df_tot_li = []



example_path = beyond_listdir_pos.rel_paths[0]
suggested_keys = ','.join(example_path.split('/')[1:-1])
entry_label = ('Insert names of the keys separated by comma\n'
   'The number of keys determine how many folder levels\n'
   'will be used to determine each experiment group\n\n'
   'Experiments selected have the following relative path:\n\n '
   f'{example_path}\n\n'
   'Keys names:')
keys_names = [key.strip() for key in single_entry_messagebox(
                                                     title='Keys names',
                                                     entry_label=entry_label,
                                                     input_txt=suggested_keys,
                                                     ).entry_txt.split(',')]

print(f'Keys names: {keys_names}')

keys = []
analysis_inputs_found = False
for spotM_path in beyond_listdir_pos.spotM_paths:
    exp_path = os.path.dirname(spotM_path)
    # Each key will be a tuple of n last folder levels where n is len(keys_names)
    # This results in a n levels MultiIndex for each AllPos dataframe
    exp_path = exp_path.replace(os.sep, '/')
    keys.append(tuple(exp_path.split('/')[-len(keys_names):]))
    idx = ['cell_cycle_stage', 'Moth_ID', 'Position_n', 'frame_i']
    idx_bud = ['cell_cycle_stage', 'Bud_ID', 'Position_n', 'frame_i']
    for filename in os.listdir(spotM_path):
        print(filename)
        csv_path = f'{spotM_path}\{filename}'
        if filename.find('1_AllPos_ellip_test_MOTH_data')!=-1:
            ellips_test_df_moth = pd.read_csv(csv_path, index_col=idx)
            ellips_test_df_moth_li.append(ellips_test_df_moth)
        elif filename.find('1_AllPos_ellip_test_BUD_data')!=-1:
            ellips_test_df_bud = pd.read_csv(csv_path, index_col=idx_bud)
            ellips_test_df_bud_li.append(ellips_test_df_bud)
        elif filename.find('1_AllPos_ellip_test_TOT_data')!=-1:
            ellips_test_df_tot = pd.read_csv(csv_path, index_col=idx)
            ellips_test_df_tot_li.append(ellips_test_df_tot)

        elif filename.find('2_AllPos_p-_test_MOTH_data.csv')!=-1:
            p_test_df_moth = pd.read_csv(csv_path, index_col=idx)
            p_test_df_moth_li.append(p_test_df_moth)
        elif filename.find('2_AllPos_p-_test_BUD_data')!=-1:
            p_test_df_bud = pd.read_csv(csv_path, index_col=idx_bud)
            p_test_df_bud_li.append(p_test_df_bud)
        elif filename.find('2_AllPos_p-_test_TOT_data')!=-1:
            p_test_df_tot = pd.read_csv(csv_path, index_col=idx)
            p_test_df_tot_li.append(p_test_df_tot)

        elif filename.find('3_AllPos_p-_ellip_test_MOTH_data')!=-1:
            p_ellips_test_df_moth = pd.read_csv(csv_path, index_col=idx)
            p_ellips_test_df_moth_li.append(p_ellips_test_df_moth)
        elif filename.find('3_AllPos_p-_ellip_test_BUD_data')!=-1:
            p_ellips_test_df_bud = pd.read_csv(csv_path, index_col=idx_bud)
            p_ellips_test_df_bud_li.append(p_ellips_test_df_bud)
        elif filename.find('3_AllPos_p-_ellip_test_TOT_data')!=-1:
            p_ellips_test_df_tot = pd.read_csv(csv_path, index_col=idx)
            p_ellips_test_df_tot_li.append(p_ellips_test_df_tot)
        elif filename.find('analysis_inputs.csv')!=-1:
            if not analysis_inputs_found:
                analysis_inputs_path = csv_path
                analysis_inputs_found = True

names = keys_names.copy()
names.extend(['Cell Cycle Stage', 'Moth_ID', 'Position_n', 'frame_i'])
if ellips_test_df_moth_li:
    ellips_test_df_moth = pd.concat(ellips_test_df_moth_li, keys=keys,
                                    names=names)
if ellips_test_df_bud_li:
    ellips_test_df_bud = pd.concat(ellips_test_df_bud_li, keys=keys,
                               names=names)
if ellips_test_df_tot_li:
    ellips_test_df_tot = pd.concat(ellips_test_df_tot_li, keys=keys,
                               names=names)
if p_test_df_moth_li:
    p_test_df_moth = pd.concat(p_test_df_moth_li, keys=keys,
                               names=names)
if p_test_df_bud_li:
    p_test_df_bud = pd.concat(p_test_df_bud_li, keys=keys,
                               names=names)
if p_test_df_tot_li:
    p_test_df_tot = pd.concat(p_test_df_tot_li, keys=keys,
                               names=names)
if p_ellips_test_df_moth_li:
    p_ellips_test_df_moth = pd.concat(p_ellips_test_df_moth_li, keys=keys,
                               names=names)
if p_ellips_test_df_bud_li:
    p_ellips_test_df_bud = pd.concat(p_ellips_test_df_bud_li, keys=keys,
                               names=names)
if p_ellips_test_df_tot_li:
    p_ellips_test_df_tot = pd.concat(p_ellips_test_df_tot_li, keys=keys,
                               names=names)

print(names)
print(p_ellips_test_df_tot.index.get_level_values(3).unique())
print(p_ellips_test_df_tot)

print('Saving...')
main_data_path = f'{selected_path}/AllExp_mitoQUANT_data_{vNUM}_run-num{run_num}'

name = single_entry_messagebox(entry_label='Name to append to file names: ',
                               input_txt='WT', width=40,
                               toplevel=False).entry_txt

ellips_test_df_moth_filename = f'{name}_1_AllExp_ellip_test_MOTH_data.csv'
ellips_test_df_bud_filename = f'{name}_1_AllExp_ellip_test_BUD_data.csv'
ellips_test_df_tot_filename = f'{name}_1_AllExp_ellip_test_TOT_data.csv'
p_test_df_moth_filename = f'{name}_2_AllExp_p-_test_MOTH_data.csv.csv'
p_test_df_bud_filename = f'{name}_2_AllExp_p-_test_BUD_data.csv'
p_test_df_tot_filename = f'{name}_2_AllExp_p-_test_TOT_data.csv'
p_ellips_test_df_moth_filename = f'{name}_3_AllExp_p-_ellip_test_MOTH_data.csv'
p_ellips_test_df_bud_filename = f'{name}_3_AllExp_p-_ellip_test_BUD_data.csv'
p_ellips_test_df_tot_filename = f'{name}_3_AllExp_p-_ellip_test_TOT_data.csv'

if not os.path.exists(main_data_path):
    os.mkdir(main_data_path)

if ellips_test_df_moth_li:
    ellips_test_df_moth.to_csv(f'{main_data_path}/{ellips_test_df_moth_filename}',
                           encoding='utf-8-sig')
if ellips_test_df_bud_li:
    ellips_test_df_bud.to_csv(f'{main_data_path}/{ellips_test_df_bud_filename}',
                           encoding='utf-8-sig')
if ellips_test_df_tot_li:
    ellips_test_df_tot.to_csv(f'{main_data_path}/{ellips_test_df_tot_filename}',
                           encoding='utf-8-sig')

if p_test_df_moth_li:
    p_test_df_moth.to_csv(f'{main_data_path}/{p_test_df_moth_filename}',
                           encoding='utf-8-sig')
if p_test_df_bud_li:
    p_test_df_bud.to_csv(f'{main_data_path}/{p_test_df_bud_filename}',
                           encoding='utf-8-sig')
if p_test_df_tot_li:
    p_test_df_tot.to_csv(f'{main_data_path}/{p_test_df_tot_filename}',
                           encoding='utf-8-sig')

if p_ellips_test_df_moth_li:
    p_ellips_test_df_moth.to_csv(f'{main_data_path}/{p_ellips_test_df_moth_filename}',
                           encoding='utf-8-sig')
if p_ellips_test_df_bud_li:
    p_ellips_test_df_bud.to_csv(f'{main_data_path}/{p_ellips_test_df_bud_filename}',
                           encoding='utf-8-sig')
if p_ellips_test_df_tot_li:
    p_ellips_test_df_tot.to_csv(f'{main_data_path}/{p_ellips_test_df_tot_filename}',
                           encoding='utf-8-sig')

if analysis_inputs_found:
    try:
        shutil.copy2(analysis_inputs_path, main_data_path)
    except:
        traceback.print_exc()
print(f'File saved to {main_data_path}!')

subprocess.Popen('explorer "{}"'.format(os.path.normpath(main_data_path)))
