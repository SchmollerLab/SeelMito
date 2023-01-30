import os
import sys
import re
import shutil
import tkinter as tk
from tkinter import ttk
from natsort import natsorted
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(script_dir)
sys.path.append(main_dir)

from core import twobuttonsmessagebox
from load import select_exp_folder
from apps import tk_breakpoint
from prompts import folder_dialog, dual_entry_messagebox

class beyond_listdir_pos:
    def __init__(self, folder_path):
        self.bp = tk_breakpoint()
        self.folder_path = folder_path
        self.TIFFs_paths = []
        self.count_recursions = 0
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
                rec_count_ok = self.count_recursions < 15
                if contains_TIFFs and rec_count_ok:
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
            all_exp_info.append(rel_path)
        return all_exp_info

vNUM, run_num = dual_entry_messagebox(
    title='Analysis version and run number',
    entry_label="File version number to delete\n\n"
                "'All' = delete the entire spotMAX_output folder\n"
                " '0' = delete all files that don't end with 'vNUM.<ext>'\n"
                " 'v9' = delete all files that end with 'v9.<ext>'\n"
                "        (specific run or all runs if run number = int)\n",
    input_txt='v1',
    entry_label2="\nAnalysis run number: (int or 0)\n"
                 " '0' = delete all files that don't start with '{run_num}_'\n"
                 " '2' = delete all files from run number 2\n",
    input_txt2='0',
    toplevel=False
).entries_txt

selected_path = folder_dialog(title = "Select folder containing experiments")
selector = select_exp_folder()
is_pos_path = os.path.basename(selected_path).find('Position_') != -1
is_TIFFs_path = os.path.basename(selected_path).find('TIFFs') != -1
if not is_pos_path and not is_TIFFs_path:
    beyond_listdir_pos = beyond_listdir_pos(selected_path)
    selector.run_widget(beyond_listdir_pos.all_exp_info,
                         title='Concatenate all Positions',
                         label_txt='Select experiment to clean',
                         full_paths=beyond_listdir_pos.TIFFs_paths,
                         showinexplorer_button=True,
                         all_button=True)
    selected_paths = selector.paths
else:
    selected_paths = [selected_path]

for path in selected_paths:
    pos_foldernames = [pos for pos in os.listdir(path)
                           if os.path.isdir(os.path.join(path, pos))
                           and pos.find('Position_')!=-1]
    delete_all_li = []
    for pos in pos_foldernames:
        NucleoData_path = os.path.join(path, pos, 'NucleoData')
        spotMAX_path = os.path.join(path, pos, 'spotMAX_output')
        if os.path.exists(NucleoData_path):
            print(f'Renaming {NucleoData_path}')
            os.rename(NucleoData_path, spotMAX_path)

for path in selected_paths:
    print(f'Cleaning {path}')
    pos_foldernames = [pos for pos in os.listdir(path)
                           if os.path.isdir(os.path.join(path, pos))
                           and pos.find('Position_')!=-1]
    delete_all_li = []
    for pos in tqdm(pos_foldernames):
        spotMAX_path = os.path.join(path, pos, 'spotMAX_output')
        if os.path.exists(spotMAX_path):
            if vNUM == 'All':
                # Delete entire nucelodata folder
                delete_all_li.append(spotMAX_path)
            else:
                filenames = os.listdir(spotMAX_path)
                for file in filenames:
                    file_path = os.path.join(spotMAX_path, file)
                    if vNUM == '0':
                        # Delete files that don't have 'vNUM.' before extension
                        if not re.findall('_v(\d+)', file):
                            delete_all_li.append(file_path)
                    else:
                        if file.find(f'_{vNUM}.') != -1:
                            # Delete files that don't have any run number
                            # prepended
                            if run_num == '0':
                                delete_all_li.append(file_path)
                            else:
                                if file.endswith('.pdf'):
                                    if file.startswith(f'{run_num}_'):
                                        print('')
                                        print(file)
                                        import pdb; pdb.set_trace()
                                else:
                                    if file.find(f'{run_num}_mtNetQUANT_skel') != -1:
                                        print('')
                                        print(file)
                                        import pdb; pdb.set_trace()
                                    elif re.findall(f'{run_num}_(\d+)', file):
                                        print('')
                                        print(file)
                                        import pdb; pdb.set_trace()

    if delete_all_li:
        delete_print = '\n'.join(delete_all_li[:10])
        if len(delete_print) > 10:
            delete_print += '\n ....'
        msg = (f'You are going to delete the following {len(delete_all_li)} paths:\n\n'
              f'{delete_print}\n\n'
              'Are you sure you want to continue?')
        yes = twobuttonsmessagebox(
                  'Continue?', msg, 'Yes', 'Do not delete'
                  ).left_b_val
        if yes:
            print('Deleting folders/files..')
            for f in tqdm(delete_all_li):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

print('Done!')
