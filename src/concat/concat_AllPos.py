import os
import sys
import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import askyesno
from natsort import natsorted
from tqdm import tqdm

script_dirpath = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(script_dirpath)
sys.path.insert(0, src_path)

import load, prompts, apps, core

class beyond_listdir_pos:
    def __init__(self, folder_path, spotMAX_data_foldername):
        self.bp = apps.tk_breakpoint()
        self.folder_path = folder_path
        self.TIFFs_paths = []
        self.count_recursions = 0
        self.spotMAX_data_foldername = spotMAX_data_foldername
        self.listdir_recursion(folder_path)
        if not self.TIFFs_paths:
            raise FileNotFoundError(f'Path {folder_path} is not valid!')
        self.all_exp_info = self.count_analysed_pos()

    def listdir_recursion(self, folder_path):
        if os.path.isdir(folder_path):
            listdir_folder = natsorted(os.listdir(folder_path))
            contains_pos_folders = any([name.find('Position_')!=-1
                                        for name in listdir_folder])
            if not contains_pos_folders:
                contains_TIFFs = any([name=='TIFFs'
                                      for name in listdir_folder])
                contains_mitoQ_data = any([name==self.spotMAX_data_foldername
                                           for name in listdir_folder])
                rec_count_ok = self.count_recursions < 15
                if contains_TIFFs and contains_mitoQ_data and rec_count_ok:
                    self.TIFFs_paths.append(f'{folder_path}/'
                                            f'{self.spotMAX_data_foldername}')
                elif contains_TIFFs and rec_count_ok:
                    self.TIFFs_paths.append(f'{folder_path}/TIFFs')
                elif rec_count_ok:
                    for name in listdir_folder:
                        subfolder_path = f'{folder_path}/{name}'
                        self.listdir_recursion(subfolder_path)
                    self.count_recursions += 1
                else:
                    raise RecursionError(
                          'Recursion went too deep and it was aborted '
                          'Check that the experiments contains the TIFFs folder')
            else:
                exp_path = os.path.dirname(os.path.dirname(folder_path))
                contains_mitoQ_data = any([name==self.spotMAX_data_foldername
                                           for name in listdir_folder])
                self.TIFFs_paths.append(exp_path)

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
        valid_TIFFs_path = []
        for path in self.TIFFs_paths:
            rel_path = self.get_rel_path(path)
            foldername = os.path.basename(path)
            if foldername == self.spotMAX_data_foldername:
                exp_info = f'{rel_path} (All Pos. DataFrames ALREADY generated)'
            else:
                exp_info = f'{rel_path} (DataFrames NOT present!)'
            all_exp_info.append(exp_info)
        return all_exp_info

def add_cca_info(pos_df, pos_cca_df):
    frames = pos_df.index.get_level_values(0)
    IDs = pos_df.index.get_level_values(1)
    if 'frame_i' in pos_cca_df.columns:
        pos_cca_df = pos_cca_df.reset_index().set_index(['frame_i', 'Cell_ID'])
        cc_stages = pos_cca_df['Cell cycle stage'].loc[(frame_i, IDs)]
        cc_nums = pos_cca_df['# of cycles'].loc[(frame_i, IDs)]
        relationships = pos_cca_df['Relationship'].loc[(frame_i, IDs)]
        relatives_IDs = pos_cca_df['Relative\'s ID'].loc[(frame_i, IDs)]
        OFs = pos_cca_df['OF'].loc[(frame_i, IDs)]
    else:
        cc_stages = pos_cca_df['Cell cycle stage'].loc[IDs]
        cc_nums = pos_cca_df['# of cycles'].loc[IDs]
        relationships = pos_cca_df['Relationship'].loc[IDs]
        relatives_IDs = pos_cca_df['Relative\'s ID'].loc[IDs]
        OFs = pos_cca_df['OF'].loc[IDs]
    pos_df['Cell Cycle Stage'] = cc_stages.to_list()
    pos_df['Cycle repetition #'] = cc_nums.to_list()
    pos_df['Relationship'] = relationships.to_list()
    pos_df['Relative\'s ID'] = relatives_IDs.to_list()
    pos_df['OF'] = OFs.to_list()
    return pos_df


def check_IDs_match(pos_df, cca_df, cca_df_path, pos_df_path):
    IDs_match = [ID in cca_df.index for ID in pos_df.index.get_level_values(1)]
    if all(IDs_match):
        return True
    else:
        print(cca_df.index)
        print(pos_df.index.get_level_values(1))
        print(cca_df)
        print(pos_df)
        err = (f'Cell cycle stage analysis at\n\n'
               f'{cca_df_path}\n\n has IDs that are not the same of the '
               'analysis file at:\n\n'
               f'{pos_df_path}\n\nRun cell cycle analysis again')
        tk.messagebox.showerror('IDs mismatch!', err)
        raise ValueError(err)


class replace_or_skip:
    def __init__(self):
        self.replace = False
        self.skip = False
        self.asked_once = False



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
selected_path = prompts.folder_dialog(title=
                                "Select folder containing valid experiments")
spotMAX_data_foldername = ''
if selected_path.find('TIFFs') != -1:
    selected_paths = [selected_path]
    TIFFs_path = selected_path
else:
    beyond_listdir_pos = beyond_listdir_pos(
        selected_path, spotMAX_data_foldername
    )
    selector = load.select_exp_folder()
    selector.run_widget(beyond_listdir_pos.all_exp_info,
                         title='Concatenate all Positions',
                         label_txt='Select experiment to generate DataFrames',
                         full_paths=beyond_listdir_pos.TIFFs_paths,
                         showinexplorer_button=True,
                         all_button=True)
    selected_paths = selector.paths
    TIFFs_path = beyond_listdir_pos.TIFFs_paths[0]

ls_TIFFs_path = os.listdir(TIFFs_path)

pos_foldernames = [p for p in ls_TIFFs_path
                   if p.find('Position_') != -1
                   and os.path.isdir(os.path.join(TIFFs_path, p))]
pos_path = os.path.join(TIFFs_path, pos_foldernames[0])
scan_run_num = prompts.scan_run_nums(vNUM)
run_nums = scan_run_num.scan(pos_path)
if len(run_nums) > 1:
    run_num = scan_run_num.prompt(run_nums,
                                  msg='Select run number to concatenate: ')
else:
    run_num = 1

spotMAX_data_foldername = f'spotMAX_{vNUM}_run-num{run_num}'

rs = replace_or_skip()

for selected_path in tqdm(selected_paths, ncols=100, unit='experiment'):
    foldername = os.path.basename(selected_path)
    if foldername == spotMAX_data_foldername:
        if not rs.asked_once:
            rs.replace = askyesno('FileExists', 'This experiment already '
                                              'contains spotMAX data!\n'
                                              'Do you want to replace them?'
                                              )
            rs.asked_once = True
        if rs.replace:
            TIFFs_path = f'{os.path.dirname(selected_path)}/TIFFs'
        else:
            rs.skip = True
    elif foldername == 'TIFFs':
        TIFFs_path = selected_path
        rs.skip = False

    if not rs.skip:
        AllPos_summary_df = core.spotMAX_concat_pos(TIFFs_path, vNUM=vNUM,
                                            run_num=run_num, do_save=True)

        print('')
        print(f'Loading all dataframes from {selected_path}...')
        # Iterate position folders and concatenate dataframes
        AllPos_summary_df.load_df_from_allpos(vNUM=vNUM, run_num=run_num)


        print('Generating big DataFrame...')

        (ellips_test_df_moth, ellips_test_df_bud,
        ellips_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.ellips_test_df_li)

        (p_test_df_moth, p_test_df_bud,
        p_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.p_test_df_li)

        (p_ellips_test_df_moth, p_ellips_test_df_bud,
        p_ellips_test_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                        AllPos_summary_df.p_ellips_test_df_li)

        if AllPos_summary_df.spotfit_df_li:
            (spotfit_df_moth, spotfit_df_bud,
            spotfit_df_tot) = AllPos_summary_df.generate_bud_moth_tot_dfs(
                                            AllPos_summary_df.spotfit_df_li)

        print('Saving all positions concantenated data...')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_moth,
                                         '1_AllPos_ellip_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_bud,
                                         '1_AllPos_ellip_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(ellips_test_df_tot,
                                         '1_AllPos_ellip_test_TOT_data.csv')

        AllPos_summary_df.save_AllPos_df(p_test_df_moth,
                                         '2_AllPos_p-_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(p_test_df_bud,
                                         '2_AllPos_p-_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(p_test_df_tot,
                                         '2_AllPos_p-_test_TOT_data.csv')

        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_moth,
                                         '3_AllPos_p-_ellip_test_MOTH_data.csv')
        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_bud,
                                         '3_AllPos_p-_ellip_test_BUD_data.csv')
        AllPos_summary_df.save_AllPos_df(p_ellips_test_df_tot,
                                         '3_AllPos_p-_ellip_test_TOT_data.csv')

        if AllPos_summary_df.spotfit_df_li:
            AllPos_summary_df.save_AllPos_df(spotfit_df_moth,
                                         '4_AllPos_spotfit_MOTH_data.csv')
            AllPos_summary_df.save_AllPos_df(spotfit_df_bud,
                                         '4_AllPos_spotfit_BUD_data.csv')
            AllPos_summary_df.save_AllPos_df(spotfit_df_tot,
                                         '4_AllPos_spotfit_TOT_data.csv')

        spotMAX_inputs_path = AllPos_summary_df.analysis_inputs_path
        AllPos_summary_df.save_ALLPos_analysis_inputs(spotMAX_inputs_path)

        print(f'Files save to {AllPos_summary_df.spotMAX_data_path}')
        print('')
