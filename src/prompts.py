import os
import re
import time
import traceback
import difflib
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from ast import literal_eval
from core import calc_resolution_limited_vol
from skimage.color import label2rgb, gray2rgb
from skimage.measure import regionprops
from skimage import img_as_float
import apps

def file_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.askopenfilename(**options)
    root.destroy()
    return path

def folder_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.Directory(**options).show()
    root.destroy()
    return path


class fourbuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self):
        self.prompt = True

    def run(self, title, message, button_1_text,
                 button_2_text, button_3_text, button_4_text,
                 path, geometry="+800+400",auto_close=False):
        self.do_save = False
        self.replace = False
        self.path = path
        root = tk.Tk()
        self.root = root
        root.lift()
        # # root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 10)).grid(row=0, column=0, columnspan=2,
                 pady=4, padx=4)

        do_save_b = tk.Button(root,
                      text=button_1_text,
                      command=self.do_save_cb,
                      width=10).grid(row=4,
                                      column=0,
                                      pady=4, padx=6)

        close = tk.Button(root,
                  text=button_2_text,
                  command=self.close,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=4, padx=6)
        repl = tk.Button(root,
                  text=button_3_text,
                  command=self.replace_cb,
                  width=10,).grid(row=5,
                                  column=0,
                                  pady=4, padx=6)

        expl = tk.Button(root,
                  text=button_4_text,
                  command=self.open_path_explorer,
                  width=15)

        expl.grid(row=5, column=1, pady=4, padx=6)
        expl.config(font=(None, 9, 'italic'))

        self.time_elapsed_sv = tk.StringVar()
        time_elapsed_label = tk.Label(root,
                         textvariable=self.time_elapsed_sv,
                         font=(None, 10)).grid(row=6, column=0,
                                               columnspan=3, padx=4, pady=4)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.bind('<Enter>', self.stop_timer)
        self.timer_t_final = time.time() + 10
        self.auto_close = True
        if auto_close:
            self.tk_timer()
        self.root.mainloop()

    def stop_timer(self, event):
        self.auto_close = False

    def tk_timer(self):
        if self.auto_close:
            seconds_elapsed = self.timer_t_final - time.time()
            seconds_elapsed = int(round(seconds_elapsed))
            if seconds_elapsed <= 0:
                print('Time elpased. Replacing files')
                self.replace_cb()
            self.time_elapsed_sv.set('Window will close automatically in: {} s'
                                                       .format(seconds_elapsed))
            self.root.after(1000, self.tk_timer)
        else:
            self.time_elapsed_sv.set('')

    def do_save_cb(self):
        self.do_save = True
        self.root.quit()
        self.root.destroy()

    def replace_cb(self):
        self.replace=True
        self.root.quit()
        self.root.destroy()

    def open_path_explorer(self):
        subprocess.Popen('explorer "{}"'.format(os.path.normpath(self.path)))

    def close(self):
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class scan_run_nums:
    def __init__(self, vNUM):
        self.vNUM = vNUM
        self.is_first_call = True

    def scan(self, pos_paths):
        self.spotmax_paths = []
        all_runs = []
        if isinstance(pos_paths, str):
            pos_paths = [pos_paths]
        for pos_path in pos_paths:
            spotmax_path = os.path.join(pos_path, 'spotMAX_output')
            # if not os.path.exists(spotmax_path):
            #     spotmax_path = os.path.join(pos_path, 'NucleoData')
            if os.path.exists(spotmax_path):
                self.spotmax_paths.append(spotmax_path)
                filenames = os.listdir(spotmax_path)
                run_nums = [re.findall('(\d+)_(\d)_', f)
                                     for f in filenames]
                run_nums = np.unique(
                           np.array(
                                [int(m[0][0]) for m in run_nums if m], int))
                run_nums = [r for r in run_nums for f in filenames
                            if f.startswith(f'{r}')]
                all_runs.extend(run_nums)
        return np.unique(np.array(all_runs, int))


    def prompt(self, run_nums, msg='Select run number to analyse :'):
        root = tk.Tk()
        root.lift()
        # # root.attributes("-topmost", True)
        root.title('Multiple runs detected')
        root.geometry("+800+400")
        # tk.Label(root,
        #          text='Select run number to analyse: ',
        #          font=(None, 11)).grid(row=0, column=0, columnspan=2,
        #                                                 pady=(10,0),
        #                                                 padx=10)
        tk.Label(root,
                 text=msg,
                 font=(None, 10),
                 justify='right').grid(row=1, column=0, pady=10, padx=10)

        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=3, column=1,
                                                  pady=(0,10), padx=10)

        show_b = tk.Button(root, text='Print analysis inputs', width=20,
                           command=self._print_analysis_inputs)
        show_b.grid(row=2, column=1, pady=(0,5), padx=10)
        show_b.config(font=(None, 9, 'italic'))

        run_num_Intvar = tk.IntVar()
        run_num_combob = ttk.Combobox(root, width=15, justify='center',
                                      textvariable=run_num_Intvar)
        run_num_combob.option_add('*TCombobox*Listbox.Justify', 'center')
        run_num_combob['values'] = list(run_nums)
        run_num_combob.grid(column=1, row=1, padx=10, pady=10)
        run_num_combob.current(0)

        root.protocol("WM_DELETE_WINDOW", self._abort)
        self.run_num_Intvar = run_num_Intvar
        self.root = root
        root.mainloop()
        return run_num_Intvar.get()

    def _print_analysis_inputs(self):
        run_num = self.run_num_Intvar.get()
        for spotmax_path in self.spotmax_paths:
            analysis_inputs_path = os.path.join(
                    spotmax_path,
                    f'{run_num}_{self.vNUM}_analysis_inputs.csv'
            )
            if os.path.exists(analysis_inputs_path):
                df_inputs = pd.read_csv(analysis_inputs_path)
                df_inputs['Description'] = df_inputs['Description'].str.replace(
                                                                '\u03bc', 'u')
                df_inputs.set_index('Description', inplace=True)
                print('================================')
                print(f'Analysis inputs for run number {run_num}:')
                print('')
                print(df_inputs)
                print('================================')
                break

    def _close(self):
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class num_frames_toQuant:
    def __init__(self):
        self.is_first_call = True

    def prompt(self, tot_frames, last_segm_i=None, last_tracked_i=None):
        root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        self.root.title('Number of frames to segment')
        root.geometry('+800+400')
        root.lift()
        # root.attributes("-topmost", True)
        # root.focus_force()
        tk.Label(root,
                 text="How many frames do you want to analyse?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        if last_segm_i is not None:
            txt = (f'(there is a total of {tot_frames} frames,\n'
                   f'last segmented frame is index {last_segm_i})')
        else:
            txt = f'(there is a total of {tot_frames} frames)'
        if last_tracked_i is not None:
            txt = f'{txt[:-1]}\nlast tracked frame is index {last_tracked_i})'
        tk.Label(root,
                 text=txt,
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start frame",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=tk.E,
                                               padx=4)
        tk.Label(root,
                 text="Number of frames to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                            textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(0))
        sv_sf.trace_add("write", self.set_all)
        self.start_frame = start_frame
        start_frame.grid(row=2, column=1, pady=8, sticky=tk.W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=tk.W)
        tk.Button(root,
                  text='All',
                  command=self.set_all,
                  width=8).grid(row=3,
                                 column=2,
                                 pady=4, padx=4)
        tk.Button(root,
                  text='OK',
                  command=self.ok,
                  width=12).grid(row=4,
                                 column=0,
                                 pady=8,
                                 columnspan=3)
        root.bind('<Return>', self.ok)
        start_frame.focus_force()
        start_frame.selection_range(0, tk.END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # root.after(1000, self.set_foreground_window)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            rightRange = self.tot_frames - startf
            self.num_frames.delete(0, tk.END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf
                self.num_frames.delete(0, tk.END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def ok(self, event=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf, stopf)
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

    def set_foreground_window(self):
        self.root.lift()
        # self.root.attributes("-topmost", True)
        self.root.focus_force()

class single_combobox_widget:
    def __init__(self):
        self.is_first_call = True


    def prompt(self, values, title='Select value', message=None):
        root = tk.Tk()
        root.lift()
        # root.attributes("-topmost", True)
        root.title(title)
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           pady=(10,0), padx=10)
            row += 1

        # tk.Label(root,
        #          text='Select value:',
        #          font=(None, 11)).grid(row=row, column=0, pady=(10,0),
        #                                                 padx=10)
        w = max([len(v) for v in values])+10
        _var = tk.StringVar()
        _combob = ttk.Combobox(
            root, width=w, justify='center', textvariable=_var
        )
        _combob.option_add('*TCombobox*Listbox.Justify', 'center')
        _combob['values'] = values
        _combob.grid(column=0, row=row, padx=10, pady=(10,0))
        _combob.current(0)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=row, column=0,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._abort)
        self._var = _var
        self.root = root
        root.mainloop()

    def _close(self):
        self.selected_val = self._var.get()
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class select_channel_name:
    def __init__(self, which_channel=None):
        self.is_first_call = True
        self.which_channel = which_channel
        self.last_sel_channel = self._load_last_selection()

    def get_available_channels(self, filenames):
        channel_names = []
        basename = filenames[0]
        for file in filenames:
            if file.find('.ini') != -1:
                continue
            sm = difflib.SequenceMatcher(None, file, basename)
            i, j, k = sm.find_longest_match(0, len(file), 0, len(basename))
            basename = file[i:i+k]
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if ext == '.tif':
                channel_name = filename.split(basename)[-1]
                channel_names.append(channel_name)
        return channel_names

    def _load_last_selection(self):
        last_sel_channel = None
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            temp_path = os.path.join(_path, 'temp')
            txt_path = os.path.join(temp_path, f'{ch}_last_sel.txt')
            if os.path.exists(txt_path):
                with open(txt_path) as txt:
                    last_sel_channel = txt.read()
        return last_sel_channel

    def _saved_last_selection(self, selection):
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            temp_path = os.path.join(_path, 'temp')
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            txt_path = os.path.join(temp_path, f'{ch}_last_sel.txt')
            with open(txt_path, 'w') as txt:
                txt.write(selection)


    def prompt(self, channel_names, message=None):
        root = tk.Tk()
        root.lift()
        # root.attributes("-topmost", True)
        root.title('Select channel name')
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           columnspan= 2, pady=(10,0),
                                                          padx=10)
            row += 1

        tk.Label(root,
                 text='Select channel name to analyse:',
                 font=(None, 11)).grid(row=row, column=0, pady=(10,0),
                                                        padx=10)

        ch_name_var = tk.StringVar()
        ch_name_combob = ttk.Combobox(root, width=20, justify='center',
                                      textvariable=ch_name_var)
        ch_name_combob.option_add('*TCombobox*Listbox.Justify', 'center')
        ch_name_combob['values'] = channel_names
        ch_name_combob.grid(column=1, row=row, padx=10, pady=(10,0))
        if self.last_sel_channel is not None:
            if self.last_sel_channel in channel_names:
                ch_name_combob.current(channel_names.index(self.last_sel_channel))
            else:
                ch_name_combob.current(0)
        else:
            ch_name_combob.current(0)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=row, column=0,
                                                  columnspan=2,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._abort)
        self.ch_name_var = ch_name_var
        self.root = root
        root.mainloop()

    def _close(self):
        self.channel_name = self.ch_name_var.get()
        self._saved_last_selection(self.channel_name)
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class spotFIT_inputs:
    def __init__(self):
        self.is_first_call = True

    def prompt(self, df_inputs, is_segm_3D):
        # Create and config tk window
        root = tk.Tk()
        root.lift()
        root.title('spotFIT analysis inputs')
        # root.attributes("-topmost", True)
        root.geometry("+800+400")
        self.root = root
        idx = -1

        # ZYX voxel size
        idx += 1
        zyx_vox_size_idx = 'ZYX voxel size (μm):'
        try:
            zyx_vox_size = literal_eval(df_inputs.at[zyx_vox_size_idx,
                                                             'Values'])
        except Exception as e:
            zyx_vox_size_idx = zyx_vox_size_idx.replace('μ', 'u')
            zyx_vox_size = literal_eval(df_inputs.at[zyx_vox_size_idx,
                                                             'Values'])
        zyx_vox_size = [round(s, 5) for s in zyx_vox_size]
        zyx_vox_size_label = tk.Label(root, text=zyx_vox_size_idx,
                                            font=(None, 10))
        zyx_vox_size_var = tk.StringVar()
        zyx_vox_size_label.grid(row=idx, sticky=tk.E, pady=(10,0), padx=(10,0))
        zyx_vox_size_entry = tk.Entry(root, justify='center', width=30)
        zyx_vox_size_entry.grid(row=idx, padx=(4, 10),
                                      pady=(10,0), column=1, columnspan=2)
        zyx_vox_size_entry.insert(0, f'{zyx_vox_size}')
        self.zyx_vox_size_entry = zyx_vox_size_entry

        # ZYX minimum spot volume (um)
        idx += 1
        df_row_i = 'ZYX minimum spot volume (um)'
        self.zyx_spot_min_vol_um = None
        if df_row_i in df_inputs.index:
            zyx_spot_min_vol_um = df_inputs.at[df_row_i, 'Values']
        else:
            zyx_spot_min_vol_um = '(1, 0.3, 0.3)'

        zyx_spot_vol_label = tk.Label(root, text=df_row_i,
                                            font=(None, 10))
        zyx_spot_vol_label.grid(row=idx, sticky=tk.E, pady=(10,0), padx=(10,0))
        zyx_spot_vol_entry = tk.Entry(root, justify='center', width=30)
        zyx_spot_vol_entry.grid(row=idx, padx=(4, 10),
                                      pady=(10,0), column=1, columnspan=2)
        zyx_spot_vol_entry.insert(0, zyx_spot_min_vol_um)
        self.zyx_spot_vol_entry = zyx_spot_vol_entry


        # Segmentation info ('2D' or '3D')
        df_row_i = 'Segmentation info (ignore if not present):'
        self.segm_info = None
        if df_row_i in df_inputs.index:
            segm_info = df_inputs.at[df_row_i, 'Values']
            self.segm_info = segm_info
        else:
            # Segmentation info ('2D' or '3D')
            idx += 1
            self.segm_info_var = tk.StringVar()
            self.segm_info_label_txt = 'Segmentation info:'
            self.segm_info_label = tk.Label(
                                             root,
                                             text=self.segm_info_label_txt,
                                             font=(None, 10))
            self.segm_info_label.grid(row=idx, sticky=tk.E, pady=(10,0),
                                      padx=(10,0))
            self.segm_info_3D_rb = ttk.Radiobutton(root,
                                                text='3D',
                                                variable=self.segm_info_var,
                                                value='3D')
            self.segm_info_3D_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                         column=1)

            self.segm_info_2D_rb = ttk.Radiobutton(root,
                                                text='2D',
                                                variable=self.segm_info_var,
                                                value='2D')
            self.segm_info_2D_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                      column=2)
            segm_info_var_val = '3D' if is_segm_3D else '2D'
            self.segm_info_var.set(segm_info_var_val)


        # Filter by ref_ch
        df_row_i = 'Filter spots by reference channel?'
        self.filter_by_ref_ch = None
        if df_row_i in df_inputs.index:
            var = df_inputs.at[df_row_i, 'Values']
            filter_by_ref_ch = self.isYes(var)
            var = df_inputs.at['Load a reference channel?', 'Values']
            load_ref_ch = self.isYes(var)
            self.filter_by_ref_ch = filter_by_ref_ch and load_ref_ch
        else:
            # Segmentation info ('2D' or '3D')
            idx += 1
            self.filter_by_ref_ch_var = tk.IntVar()
            self.filter_by_ref_ch_label_txt = 'Filter spots by ref. channel?'
            self.filter_by_ref_ch_label = tk.Label(
                                         root,
                                         text=self.filter_by_ref_ch_label_txt,
                                         font=(None, 10))
            self.filter_by_ref_ch_label.grid(row=idx, sticky=tk.E,
                                             pady=(10,0), padx=(10,0))
            self.filter_by_ref_ch_yes_rb = ttk.Radiobutton(root,
                                        text='Yes',
                                        variable=self.filter_by_ref_ch_var,
                                        value=1)
            self.filter_by_ref_ch_yes_rb.grid(row=idx, padx=(4, 10),
                                              pady=(10,0), column=1)

            self.filter_by_ref_ch_no_rb = ttk.Radiobutton(root,
                                        text='No',
                                        variable=self.filter_by_ref_ch_var,
                                        value=0)
            self.filter_by_ref_ch_no_rb.grid(row=idx, padx=(4, 10),
                                             pady=(10,0), column=2)
            self.filter_by_ref_ch_var.set(1)

        # Save radiobutton
        idx += 1
        save_rb_label  = tk.Label(root, text='Save?',
                                             font=(None, 10))
        save_rb_label.grid(row=idx, sticky=tk.E, pady=(10,0), padx=(10,0))
        self.save_rb_var = tk.IntVar()
        save_yes_rb = ttk.Radiobutton(root,
                                    text='Yes',
                                    variable=self.save_rb_var,
                                    value=0)
        save_yes_rb.grid(row=idx, padx=4, pady=4,
                                   column=1)
        save_no_rb = ttk.Radiobutton(root,
                                    text='No',
                                    variable=self.save_rb_var,
                                    value=1)
        save_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=2)
        self.save_rb_var.set(1)

        # Ok button
        idx += 1
        ok_b = tk.Button(root, command=self._close, text='Ok!', width=15)
        ok_b.grid(row=idx+1, pady=8, columnspan=3)

        root.bind('<Return>', self._close)
        root.protocol("WM_DELETE_WINDOW", self._abort)

        root.mainloop()

    def isYes(self, var):
        if isinstance(var, bool):
            isyes = var
        elif isinstance(var, str):
            print(var)
            isyes = var.lower()=='true' or var.lower()=='yes'
            print(isyes)
        else:
            isyes = None
            print('===========================================')
            warnings.warn(f'The variable {var} is not a boolean nor a string. The type is {type(var)}')
            print('===========================================')
        return isyes

    def _close(self):
        self.is_first_call = False
        self.zyx_vox_size = np.array(
                                literal_eval(self.zyx_vox_size_entry.get()))
        re_float = '([0-9]*[.]?[0-9]+)'
        s = self.zyx_spot_vol_entry.get()
        m = re.findall(f'{re_float}, {re_float}, {re_float}', s)
        self.zyx_spot_min_vol_um = [float(f) for f in m[0]]

        if self.segm_info is None:
            self.segm_info = self.segm_info_var.get()

        self.filter_by_ref_ch = self.filter_by_ref_ch_var.get() == 1
        self.do_save = self.save_rb_var.get() == 0
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user.')

class spotMAX_inputs_widget:
    def __init__(self, areArgsDefault):
        self.areArgsDefault = areArgsDefault
        self.show = True
        self.run_num = 1
        script_dirpath = os.path.dirname(os.path.realpath(__file__))
        last_status_csv_path = os.path.join(script_dirpath,
                                      'last_status_inputs_widget.csv')
        if os.path.exists(last_status_csv_path):
            self.last_status_df = pd.read_csv(last_status_csv_path
                                             ).set_index('Description')
        else:
            self.last_status_df = None

    def run(self, title='Analysis inputs', channel_name='mNeon',
                   zyx_voxel_size='', zyx_voxel_size_float=None,
                   z_resolution_limit='', numerical_aperture='',
                   em_wavelength='', gauss_sigma='',
                   toplevel=False, is_segm_3D=False):
        self.channel_name = channel_name
        self.input_zyx_voxel_size = zyx_voxel_size
        self.input_em_wavelength = em_wavelength
        self.input_NA = numerical_aperture
        entry6_label_txt = 'Reference channel threshold function:'
        self.entry_labels = ['ZYX voxel size (um):',
                             'Z resolution limit (um):',
                             'Numerical aperture:',
                             f'{channel_name} emission wavelength (nm):',
                             'Gaussian filter sigma:',
                             entry6_label_txt,
                             'Peak finder threshold function:',
                             'YX resolution multiplier:']

        self.zyx_voxel_size_float = zyx_voxel_size_float

        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.title(title)
        # root.attributes("-topmost", True)
        root.geometry("+800+50")
        self._root = root

        # Load reference channel radio button
        idx = 0
        self.load_ref_ch_var = tk.IntVar()
        self.load_ref_ch_label_txt = 'Load a reference channel?'
        self.load_ref_ch_label = tk.Label(
                                        root,
                                        text=self.load_ref_ch_label_txt,
                                        font=(None, 10))
        self.load_ref_ch_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.load_ref_ch_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.load_ref_ch_var,
                                            value=0,
                                            command=self.hide_ref_ch_inputs)
        self.load_ref_ch_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.load_ref_ch_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.load_ref_ch_var,
                                            value=1,
                                            command=self.hide_ref_ch_inputs)
        self.load_ref_ch_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)

        # Single object reference channel radiobutton
        idx += 1
        self.single_obj_ref_var = tk.IntVar()
        self.single_obj_ref_label_txt = 'Is ref. channel a single object per cell?'
        self.single_obj_ref_label = tk.Label(
                                        root,
                                        text=self.single_obj_ref_label_txt,
                                        font=(None, 10))
        self.single_obj_ref_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.single_obj_ref_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.single_obj_ref_var,
                                            value=0)
        self.single_obj_ref_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.single_obj_ref_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.single_obj_ref_var,
                                            value=1)
        self.single_obj_ref_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)
        self.single_obj_ref_var.set(1)

        # Filter spots by reference channel radio button
        idx += 1
        self.filter_by_ref_var = tk.IntVar()
        self.filter_by_ref_label_txt = 'Filter spots by reference channel?'
        self.filter_by_ref_label = tk.Label(
                                         root,
                                         text=self.filter_by_ref_label_txt,
                                         font=(None, 10))
        self.filter_by_ref_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.filter_by_ref_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.filter_by_ref_var,
                                            value=0)
        self.filter_by_ref_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.filter_by_ref_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.filter_by_ref_var,
                                            value=1)
        self.filter_by_ref_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)
        self.filter_by_ref_var.set(1)
        self.filter_by_ref_var.trace_add("write", self._update_gop_methods)

        # Filter z-boundaries radio button
        idx += 1
        self.filter_z_bound_var = tk.IntVar()
        self.filter_z_bound_label_txt = 'Filter spots too close to z-boundaries?'
        self.filter_z_bound_label = tk.Label(
                                         root,
                                         text=self.filter_z_bound_label_txt,
                                         font=(None, 10))
        self.filter_z_bound_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.filter_z_bound_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.filter_z_bound_var,
                                            value=0)
        self.filter_z_bound_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.filter_z_bound_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.filter_z_bound_var,
                                            value=1)
        self.filter_z_bound_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)


        # Sharpen image prior spot detection radio button
        idx += 1
        self.sharpen_var = tk.IntVar()
        self.sharpen_label_txt = 'Sharpen image prior spot detection?'
        self.sharpen_label = tk.Label(
                                         root,
                                         text=self.sharpen_label_txt,
                                         font=(None, 10))
        self.sharpen_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.sharpen_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.sharpen_var,
                                            value=0)
        self.sharpen_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.sharpen_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.sharpen_var,
                                            value=1)
        self.sharpen_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)
        self.sharpen_var.set(0)

        # Use local or global threshold for spot detection radio button
        idx += 1
        self.local_or_global_var = tk.StringVar()
        self.local_or_global_label_txt = 'Local or global threshold for spot detection?'
        self.local_or_global_label = tk.Label(
                                         root,
                                         text=self.local_or_global_label_txt,
                                         font=(None, 10))
        self.local_or_global_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.local_or_global_yes_rb = ttk.Radiobutton(root,
                                            text='Local',
                                            variable=self.local_or_global_var,
                                            value='Local')
        self.local_or_global_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )
        self.local_or_global_no_rb = ttk.Radiobutton(root,
                                            text='Global',
                                            variable=self.local_or_global_var,
                                            value='Global')
        self.local_or_global_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)
        self.local_or_global_var.set('Local')

        # Segmentation info ('2D' or '3D')
        idx += 1
        self.segm_info_var = tk.StringVar()
        self.segm_info_label_txt = 'Segmentation info (ignore if not present):'
        self.segm_info_label = tk.Label(
                                         root,
                                         text=self.segm_info_label_txt,
                                         font=(None, 10))
        self.segm_info_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.segm_info_3D_rb = ttk.Radiobutton(root,
                                            text='3D',
                                            variable=self.segm_info_var,
                                            value='3D')
        self.segm_info_3D_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                     column=1, sticky=tk.W )

        self.segm_info_2D_rb = ttk.Radiobutton(root,
                                            text='2D',
                                            variable=self.segm_info_var,
                                            value='2D')
        self.segm_info_2D_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=2, sticky=tk.W)
        segm_info_var_val = '3D' if is_segm_3D else '2D'
        self.segm_info_var.set(segm_info_var_val)

        # ZYX voxel size (um)
        idx += 1
        tk.Label(root, text='ZYX voxel size (um):', font=(None, 10)
                                      ).grid(row=idx, sticky=tk.E, pady=(10,0))
        zyx_vox_size_var = tk.StringVar(root)
        self.zyx_vox_size_var = zyx_vox_size_var
        zyx_vox_size_entry = tk.Entry(root, justify='center', width=30,
                                      textvariable=zyx_vox_size_var)
        zyx_vox_size_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                          column=1, columnspan=2)
        if zyx_voxel_size_float is None:
            zyx_vox_size_entry.insert(0, f'{[1.0, 1.0, 1.0]}')
        else:
            e1_txt = f'{[round(r, 5) for r in  zyx_voxel_size_float]}'
            zyx_vox_size_entry.insert(0, e1_txt)
        self.zyx_vox_size_entry = zyx_vox_size_entry
        zyx_vox_size_var.trace_add("write", self._update_resol_vol)
        zyx_vox_size_var.trace_add("write", self.spotsize_limits_pxl_to_um)
        zyx_vox_size_var.trace_add("write", self.spotsize_limits_um_to_pxl)

        # Entry 3
        idx += 1
        tk.Label(root, text='Numerical aperture:', font=(None, 10)
                                    ).grid(row=idx, sticky=tk.E, pady=(10,0))
        NA_var = tk.DoubleVar(root)
        if numerical_aperture != 'None':
            NA_var.set(float(numerical_aperture))
        self.NA_var = NA_var
        NA_entry = tk.Entry(root, justify='center', width=30,
                                  textvariable=NA_var)
        NA_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                      column=1, columnspan=2)
        self.NA_entry = NA_entry
        NA_var.trace_add("write", self._update_resol_vol)

        # Entry 4
        idx += 1
        tk.Label(root, text=f'{channel_name} emission wavelength (nm):',
                       font=(None, 10)).grid(row=idx, sticky=tk.E, pady=(10,0))
        em_wavel_var = tk.DoubleVar(root)
        self.em_wavel_var = em_wavel_var
        if em_wavelength != 'None':
            em_wavel_var.set(float(em_wavelength))
        em_wavel_entry = tk.Entry(root, justify='center', width=30,
                                  textvariable=em_wavel_var)
        em_wavel_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                            column=1, columnspan=2)
        self.em_wavel_entry = em_wavel_entry
        em_wavel_var.trace_add("write", self._update_resol_vol)


        # Entry 2
        idx += 1
        tk.Label(root, text='Z resolution limit (um):', font=(None, 10)
                                    ).grid(row=idx, sticky=tk.E, pady=(10,0))
        z_resolution_limit_var = tk.DoubleVar(root, 1.0)
        z_resolution_limit_entry = tk.Entry(root, justify='center', width=30,
                                  textvariable=z_resolution_limit_var)
        z_resolution_limit_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                                      column=1, columnspan=2)
        self.z_resolution_limit_entry = z_resolution_limit_entry
        z_resolution_limit_var.trace_add("write", self._update_resol_vol)

        # YX resolution multiplier
        idx += 1
        tk.Label(root, text='YX esolution multiplier:', font=(None, 10)
                                      ).grid(row=idx, sticky=tk.E, pady=(10,0))
        yx_multi_var = tk.DoubleVar(root, 1.0)
        yx_multiplier_entry = tk.Entry(root, justify='center', width=30,
                                       textvariable=yx_multi_var)
        yx_multiplier_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                          column=1, columnspan=2)
        yx_multi_var.trace_add("write", self._update_resol_vol)
        self.yx_multiplier_entry = yx_multiplier_entry

        # ZYX resolution limited volume
        idx += 1
        tk.Label(root,  text='Spot ZYX minimum volume:', font=(None, 10)
                                     ).grid(row=idx, rowspan=2,
                                            sticky=tk.E, pady=(10,0))
        yx_resolution_multiplier = float(self.yx_multiplier_entry.get())
        calc = (
            zyx_voxel_size_float is not None and
            numerical_aperture != 'None' and
            em_wavelength != 'None'
        )
        if calc:
            wavelen = float(self.em_wavel_entry.get())
            NA = float(self.NA_entry.get())
            zyx_vox_dim = literal_eval(self.zyx_vox_size_entry.get())
            z_resolution_limit = float(self.z_resolution_limit_entry.get())

            (zyx_resolution,
            zyx_resolution_pxl, _) = calc_resolution_limited_vol(wavelen, NA,
                                                        yx_resolution_multiplier,
                                                        zyx_vox_dim,
                                                        z_resolution_limit)
            z, y, x = zyx_resolution
            zyx_resolution_txt = f'({z:.3f}, {y:.3f}, {x:.3f}) um'
            z, y, x = zyx_resolution_pxl
            zyx_resolution_txt = f'({z:.3f}, {y:.3f}, {x:.3f}) pxl'
        else:
            zyx_resolution_txt = f'(ND, ND, ND) um'
            zyx_resolution_txt = f'(ND, ND, ND) pxl'

        zyx_resol_limit_vol_um = tk.Label(root,
                                        text=zyx_resolution_txt,
                                        font=(None, 10))
        zyx_resol_limit_vol_um.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=1, columnspan=2)

        idx += 1
        zyx_resol_limit_vol_vox = tk.Label(root,
                                        text=zyx_resolution_txt,
                                        font=(None, 10))
        zyx_resol_limit_vol_vox.grid(row=idx, padx=(4, 10), pady=(5,0),
                                     column=1, columnspan=2)

        self.zyx_resol_limit_vol_um = zyx_resol_limit_vol_um
        self.zyx_resol_limit_vol_vox = zyx_resol_limit_vol_vox



        # Entry 5
        idx += 1
        tk.Label(root, text='Gaussian filter sigma:', font=(None, 10)
                                    ).grid(row=idx, sticky=tk.E, pady=(10,0))
        gauss_sigma_entry = tk.Entry(root, justify='center', width=30)
        gauss_sigma_entry.grid(row=idx, padx=(4, 10), pady=(10,0),
                               column=1, columnspan=2)
        gauss_sigma_entry.insert(0, gauss_sigma)

        # Entry 6
        idx += 1
        thresh_methods = ['threshold_li', 'threshold_isodata',
                          'threshold_otsu', 'threshold_minimum',
                          'threshold_triangle', 'threshold_mean',
                          'threshold_yen'
        ]
        self.entry6_label = tk.Label(root, text=entry6_label_txt,
                                           font=(None, 10))
        self.entry6_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        ref_ch_thresh_func_var = tk.StringVar()
        ref_ch_thresh_func_cb = ttk.Combobox(root, width=25, justify='center',
                                         textvariable=ref_ch_thresh_func_var)
        ref_ch_thresh_func_cb.option_add('*TCombobox*Listbox.Justify', 'center')
        ref_ch_thresh_func_cb['values'] = thresh_methods
        ref_ch_thresh_func_cb.current(0)
        ref_ch_thresh_func_cb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                        column=1, columnspan=2)
        self.ref_ch_thresh_func_cb = ref_ch_thresh_func_cb
        self.thresh_methods = thresh_methods

        # Entry 7
        idx += 1
        tk.Label(root, text='Peak finder threshold function:', font=(None, 10)
                                    ).grid(row=idx, sticky=tk.E, pady=(10,0))
        peak_finder_thresh_func_var = tk.StringVar()
        peak_finder_tresh_func_cb = ttk.Combobox(root, width=25,
                                      justify='center',
                                      textvariable=peak_finder_thresh_func_var)
        peak_finder_tresh_func_cb.option_add('*TCombobox*Listbox.Justify',
                                                                 'center')
        peak_finder_tresh_func_cb['values'] = thresh_methods
        peak_finder_tresh_func_cb.current(0)
        peak_finder_tresh_func_cb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                       column=1, columnspan=2)
        self.peak_finder_tresh_func_cb = peak_finder_tresh_func_cb
        self.peak_finder_thresh_func_var = peak_finder_thresh_func_var

        # Entry 8
        idx += 1
        self.gop_cb_idx = idx
        gop_methods = ['t-test', 'effect size', 'effect size bootstrapping']
        self.gop_methods = gop_methods
        tk.Label(root, text='Filter good peaks method:',
                       font=(None, 10)).grid(row=idx, sticky=tk.E, pady=(10,0))
        self.entry_labels.append('Filter good peaks method:')
        e_gop_var = tk.StringVar()
        self.e_gop_var = e_gop_var
        gop_methods_cb = ttk.Combobox(root, width=25, justify='center',
                                            textvariable=e_gop_var)
        gop_methods_cb.option_add('*TCombobox*Listbox.Justify', 'center')
        gop_methods_cb['values'] = gop_methods

        gop_methods_cb.grid(row=idx, padx=(4, 10), pady=(10,0),
                            column=1, columnspan=2)
        self.gop_methods_cb = gop_methods_cb

        # Which effect size to use
        idx += 1
        effect_size_values = [
            'effsize_cohen_s', 'effsize_hedge_s', 'effsize_glass_s',
            'effsize_cliffs_s', 'effsize_cohen_pop', 'effsize_hedge_pop',
            'effsize_glass_pop'
        ]
        self.effect_size_values = effect_size_values
        self.effect_size_cb_label = tk.Label(root,
                           text='Effect size to use:',
                           font=(None, 10))
        self.effect_size_cb_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.entry_labels.append('Effect size used:')
        effect_size_cb_var = tk.StringVar()
        self.effect_size_cb_var = effect_size_cb_var
        effect_size_cb = ttk.Combobox(root, width=25, justify='center',
                                            textvariable=effect_size_cb_var)
        self.effect_size_cb = effect_size_cb
        effect_size_cb.option_add('*TCombobox*Listbox.Justify', 'center')
        effect_size_cb['values'] = effect_size_values
        effect_size_cb.current(2)
        effect_size_cb.grid(row=idx, padx=(4, 10), pady=(10,0),
                            column=1, columnspan=2)


        # Entry 9
        idx += 1
        gop_limit_label = tk.Label(root, text='p-value limit:', font=(None, 10))
        gop_limit_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.gop_limit_label = gop_limit_label
        gop_limit_e = tk.Entry(root, justify='center', width=30)
        gop_limit_e.grid(row=idx, padx=(4, 10), pady=(10,0),
                         column=1, columnspan=2)
        gop_limit_e.insert(0, '0.025')
        self.gop_limit_e = gop_limit_e

        e_gop_var.trace_add("write", self.gop_var_cb)
        gop_methods_cb.current(0)

        # Spotsize limits
        idx += 1
        spotsize_limit_label = tk.Label(
                                   root,
                                   text='Spot size limits (min, max):\n'
                                        '((0, 0) = do not filter)',
                                   justify='right',
                                   font=(None, 10))

        spotsize_limit_label.grid(row=idx, sticky=tk.E, pady=(10,0), rowspan=2)
        self.spotsize_limit_label = spotsize_limit_label

        # Spot size limits in um
        self.spotsize_limit_um_var = tk.StringVar()
        spotsize_limit_um_e = tk.Entry(root, justify='center', width=13,
                                    textvariable=self.spotsize_limit_um_var)
        spotsize_limit_um_e.grid(row=idx, padx=(4, 2), pady=(10,0),
                                 column=1, columnspan=2, sticky=tk.W)

        self.spotsize_limit_um_e = spotsize_limit_um_e

        tk.Label(root, text='um', font=(None, 10)
                 ).grid(row=idx, pady=(10,0),
                        column=2, sticky=tk.W)

        self.spotsize_limit_um_var.trace_add("write",
                                             self.spotsize_limits_um_to_pxl)

        optial_size_button = ttk.Button(root,
                                      text='Optimal',
                                      comman=self.set_optimal_size)
        optial_size_button.grid(row=idx, pady=(10,0), padx=(4,10),
                                column=2, rowspan=2, sticky=tk.E)

        # Spot size limits in pxl
        idx += 1
        self.spotsize_limit_pxl_var = tk.StringVar()
        spotsize_limit_pxl_e = tk.Entry(root, justify='center', width=13,
                                    textvariable=self.spotsize_limit_pxl_var)
        spotsize_limit_pxl_e.grid(row=idx, padx=(4, 2), pady=(10,0),
                              column=1, columnspan=2, sticky=tk.W)
        self.spotsize_limit_pxl_e = spotsize_limit_pxl_e

        tk.Label(root, text='pxl', font=(None, 10)
                 ).grid(row=idx, pady=(10,0),
                        column=2, sticky=tk.W)


        self.spotsize_limit_pxl_var.trace_add("write",
                                               self.spotsize_limits_pxl_to_um)
        self.spotsize_limit_pxl_var.set('1.0, 6.0')



        # Calc mtnetLEN radiobutton
        idx += 1
        self.calc_ref_ch_len_rb_var = tk.IntVar()
        self.calc_ref_network_length_label_txt = (
                                       'Calculate ref. channel network length?')
        self.calc_ref_ch_len_label = tk.Label(root,
                                text=self.calc_ref_network_length_label_txt,
                                font=(None, 10))
        self.calc_ref_ch_len_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.calc_ref_ch_len_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.calc_ref_ch_len_rb_var,
                                            value=0)
        self.calc_ref_ch_len_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=1)
        self.calc_ref_ch_len_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.calc_ref_ch_len_rb_var,
                                            value=1)
        self.calc_ref_ch_len_rb_var.set(1)
        self.calc_ref_ch_len_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=2)

        # do_spotSIZE radiobutton
        idx += 1
        self.compute_spots_size_label_txt = 'Compute spots size?'
        tk.Label(root, text=self.compute_spots_size_label_txt,
                 font=(None, 10)).grid(row=idx, sticky=tk.E, pady=(10,0))
        self.do_spotSIZE_rb_var = tk.IntVar()
        do_spotSIZE_yes_rb = ttk.Radiobutton(root,
                                            text='Yes',
                                            variable=self.do_spotSIZE_rb_var,
                                            value=0,
                                            command=self.hide_gauss_rb)
        do_spotSIZE_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=1)
        do_spotSIZE_no_rb = ttk.Radiobutton(root,
                                            text='No',
                                            variable=self.do_spotSIZE_rb_var,
                                            value=1,
                                            command=self.hide_gauss_rb)
        do_spotSIZE_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=2)

        # do_gaussian_fit radiobutton
        idx += 1
        self.do_gauss_fit_label_txt = 'Fit 3D Gaussians?'
        gauss_fit_rb_label  = tk.Label(root, text=self.do_gauss_fit_label_txt,
                                             font=(None, 10))
        gauss_fit_rb_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.do_gaussian_fit_rb_var = tk.IntVar()
        do_gaussian_fit_yes_rb = ttk.Radiobutton(root,
                                        text='Yes',
                                        variable=self.do_gaussian_fit_rb_var,
                                        value=0)
        do_gaussian_fit_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                    column=1)
        do_gaussian_fit_no_rb = ttk.Radiobutton(root,
                                        text='No',
                                        variable=self.do_gaussian_fit_rb_var,
                                        value=1)
        do_gaussian_fit_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=2)
        self.do_gaussian_fit_yes_rb = do_gaussian_fit_yes_rb
        self.do_gaussian_fit_no_rb = do_gaussian_fit_no_rb
        self.gauss_fit_rb_label = gauss_fit_rb_label
        self.do_gaussian_fit_rb_var.set(1)

        # Save radiobutton
        idx += 1
        row11_label = 'Save?'
        save_rb_label = tk.Label(root, text=row11_label,
                                 font=(None, 10))
        save_rb_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        self.save_rb_var = tk.IntVar()
        save_yes_rb = ttk.Radiobutton(root,
                                    text='Yes',
                                    variable=self.save_rb_var,
                                    value=0)
        save_yes_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                         column=1)
        save_no_rb = ttk.Radiobutton(root,
                                    text='No',
                                    variable=self.save_rb_var,
                                    value=1)
        save_no_rb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                   column=2)
        self.save_yes_rb = save_yes_rb
        self.save_no_rb = save_no_rb
        self.save_rb_label = save_rb_label
        self.save_row = idx
        if self.areArgsDefault:
            self.save_rb_var.set(0)
        else:
            self.save_rb_var.set(1)

        # Ok button
        idx += 1
        ok_b = tk.Button(root, command=self._quit, text='Ok!', width=15)
        ok_b.grid(row=idx+1, pady=8, column=1, columnspan=2)

        # Load inputs button
        load_b = tk.Button(root, command=self._load_inputs,
                           text='Load analysis inputs', width=20)
        load_b.grid(row=idx+1, pady=8, column=0)

        zyx_vox_size_entry.focus_force()
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root = root
        self.gauss_sigma_entry = gauss_sigma_entry
        self.ref_ch_thresh_func_var = ref_ch_thresh_func_var
        # self.e9 = e9

        self.do_spotSIZE_rb_var.set(1)
        self.hide_gauss_rb()

        self.last_row = idx

        self.filter_z_bound_var.set(1)
        self.load_ref_ch_var.set(1)
        self.hide_ref_ch_inputs()

        try:
            self.set_last_status()
        except Exception as e:
            traceback.print_exc()
            print('IGNORE error.')

        root.mainloop()

    def set_optimal_size(self):
        zyx_resol_limit_vol_vox = self.zyx_resol_limit_vol_vox['text']
        _f = '([0-9]*[.]?[0-9]+)'
        m = re.findall(f'{_f}', zyx_resol_limit_vol_vox)
        opt_z, opt_y, opt_x = [float(f)*0.68/0.997 for f in m]
        opt_limits = f'{opt_y:.2f}, {5*np.round(opt_y, 2)}'
        self.spotsize_limit_pxl_var.set(opt_limits)

    def spotsize_limits_pxl_to_um(self, name=None, index=None, mode=None):
        floatRegex = '[+-]?([0-9]*[.]?[0-9]+)'
        s = self.zyx_vox_size_entry.get()
        fa = re.findall(f'{floatRegex},\s*{floatRegex},\s*{floatRegex}', s)
        if fa:
            _, ys, xs = fa[0]
            ys, xs = float(ys), float(xs)
            s = self.spotsize_limit_pxl_var.get()
            fa = re.findall(f'{floatRegex},\s*{floatRegex},\s*{floatRegex}', s)
            if fa:
                _, a, b = fa[0]
                a, b = float(a), float(b)
                a *= ys
                b *= ys
                spotsize_limit_um_txt = f'{a:.3f}, {b:.3f}'
            else:
                ys = 0
                spotsize_limit_um_txt = 'invalid entry'
        else:
            ys = 0
            spotsize_limit_um_txt = 'invalid entry'
        if ys > 0:
            cbname = self.spotsize_limit_um_var.trace_info()[0][1]
            self.spotsize_limit_um_var.trace_remove('write', cbname)
            self.spotsize_limit_um_var.set(spotsize_limit_um_txt)
            self.spotsize_limit_um_var.trace_add('write',
                                                 self.spotsize_limits_um_to_pxl)

    def spotsize_limits_um_to_pxl(self, name=None, index=None, mode=None):
        floatRegex = '[+-]?([0-9]*[.]?[0-9]+)'
        s = self.zyx_vox_size_entry.get()
        fa = re.findall(f'{floatRegex},\s*{floatRegex},\s*{floatRegex}', s)
        if fa:
            _, ys, xs = fa[0]
            ys, xs = float(ys), float(xs)
            s = self.spotsize_limit_um_var.get()
            fa = re.findall(f'{floatRegex},\s*{floatRegex},\s*{floatRegex}', s)
            if fa:
                _, a, b = fa[0]
                a, b = float(a), float(b)
                a *= ys
                b *= ys
                spotsize_limit_um_txt = f'{a:.3f}, {b:.3f}'
            else:
                ys = 0
                spotsize_limit_um_txt = 'invalid entry'
        else:
            ys = 0
            spotsize_limit_um_txt = 'invalid entry'
        if ys > 0:
            cbname = self.spotsize_limit_pxl_var.trace_info()[0][1]
            self.spotsize_limit_pxl_var.trace_remove('write', cbname)
            self.spotsize_limit_pxl_var.set(spotsize_limit_pxl_txt)
            self.spotsize_limit_pxl_var.trace_add('write',
                                            self.spotsize_limits_pxl_to_um)

    def _load_inputs(self):
        path = file_dialog(title='Select analysis inputs .csv file',
                           filetypes=(('CSV', '.csv'), ('All files', '*')))
        self.last_status_df = pd.read_csv(path).set_index('Description')
        self.set_last_status()

    def _update_gop_methods(self, name=None, index=None, mode=None):
        if self.filter_by_ref_var.get() == 0:
            gop_methods = ['t-test', 'effect size',
                           'effect size bootstrapping']
            self.gop_methods = gop_methods
        else:
            gop_methods = ['t-test', 'effect size',
                           'effect size bootstrapping',
                           'peak_to_background ratio']
            self.gop_methods = gop_methods
        self.gop_methods_cb['values'] = gop_methods


    def _update_resol_vol(self, name=None, index=None, mode=None):
        try:
            yx_resolution_multiplier = float(self.yx_multiplier_entry.get())
            wavelen = float(self.em_wavel_entry.get())
            NA = float(self.NA_entry.get())
            zyx_vox_dim = literal_eval(self.zyx_vox_size_entry.get())
            z_resolution_limit = float(self.z_resolution_limit_entry.get())
            (zyx_resolution,
            zyx_resolution_pxl, _) = calc_resolution_limited_vol(wavelen, NA,
                                                    yx_resolution_multiplier,
                                                    zyx_vox_dim,
                                                    z_resolution_limit)
            z, y, x = zyx_resolution
            zyx_resolution_txt = f'({z:.3f}, {y:.3f}, {x:.3f}) um'
            self.zyx_resol_limit_vol_um.config(text=zyx_resolution_txt)

            z, y, x = zyx_resolution_pxl
            zyx_resolution_txt = f'({z:.3f}, {y:.3f}, {x:.3f}) pxl'
            self.zyx_resol_limit_vol_vox.config(text=zyx_resolution_txt)
        except Exception as e:
            pass

    def isYes(self, var):
        if isinstance(var, bool):
            isyes = var
        elif isinstance(var, str):
            isyes = var.lower()=='true' or var.lower()=='yes'
        else:
            isyes = None
            print('===========================================')
            warnings.warn(f'The variable {var} is not a boolean nor a string. The type is {type(var)}')
            print('===========================================')
        return isyes

    def set_last_status(self):
        if self.last_status_df is not None:
            v = 'Values'

            try:
                # Load reference channel
                load_ref_ch = (self.last_status_df
                                   .at['Load a reference channel?', v])
                print(load_ref_ch, type(load_ref_ch))
                load_ref_ch_val = 0 if self.isYes(load_ref_ch) else 1
                self.load_ref_ch_var.set(load_ref_ch_val)
                self.hide_ref_ch_inputs()
            except Exception as e:
                traceback.print_exc()
                pass

            try:
                # Filter by reference channel
                filter_by_ref_ch = (self.last_status_df
                                     .at['Filter spots by reference channel?', v])
                filter_by_ref_val = 0 if self.isYes(filter_by_ref_ch) else 1
                self.filter_by_ref_var.set(filter_by_ref_val)
            except Exception as e:
                pass

            try:
                # Filter by reference channel
                single_obj_ref = (self.last_status_df
                            .at['Is ref. channel a single object per cell?', v])
                single_obj_ref_val = 0 if self.isYes(single_obj_ref) else 1
                self.single_obj_ref_var.set(single_obj_ref_val)
            except Exception as e:
                pass

            try:
                # Filter z-boundaries
                filter_z_bound = (self.last_status_df
                                .at['Filter spots too close to z-boundaries?', v])
                filter_z_bound_val = 0 if self.isYes(filter_z_bound) else 1
                self.filter_z_bound_var.set(filter_z_bound_val)
            except Exception as e:
                pass

            try:
                # Sharpen?
                sharpen = (self.last_status_df
                                .at['Sharpen image prior spot detection?', v])
                sharpen_val = 0 if self.isYes(sharpen) else 1
                self.sharpen_var.set(sharpen_val)
            except Exception as e:
                pass

            try:
                # Local or global?
                local_or_global = (self.last_status_df
                            .at['Local or global threshold for spot detection?', v])
                self.local_or_global_var.set(local_or_global)
            except Exception as e:
                pass

            try:
                if self.input_zyx_voxel_size == 'None':
                    # ZYX voxel size (um):
                    zyx_voxel_size = (self.last_status_df
                                .at['ZYX voxel size (um):', v])
                    self.zyx_vox_size_var.set(zyx_voxel_size)
            except Exception as e:
                self.zyx_vox_size_var.set(f'{[1.0, 1.0, 1.0]}')

            try:
                if self.input_NA == 'None':
                    # Numerical aperture:
                    NA = (self.last_status_df
                                .at['Numerical aperture:', v])
                    self.NA_var.set(float(NA))
            except Exception as e:
                self.NA_var.set(1.3)

            try:
                if self.input_em_wavelength == 'None':
                    # Emission wavelength (nm)
                    em_wavel = (self.last_status_df
                         .at[f'{self.channel_name} emission wavelength (nm):', v])
                    self.em_wavel_var.set(float(em_wavel))
            except Exception as e:
                self.em_wavel_var.set(509.0)


            try:
                # Compute spots size?
                compute_spot_size = (self.last_status_df
                                     .at['Compute spots size?', v])
                compute_spot_size_val = 0 if self.isYes(compute_spot_size) else 1
                self.do_spotSIZE_rb_var.set(compute_spot_size_val)
            except Exception as e:
                pass

            try:
                # Compute reference channel network length?
                calc_ref_network_length = (self.last_status_df
                                  .at['Calculate ref. channel network length?', v])
                calc_ref_network_length_val = 0 if self.isYes(calc_ref_network_length) else 1
                self.calc_ref_ch_len_rb_var.set(calc_ref_network_length_val)
            except Exception as e:
                pass

            try:
                # Compute gaussian fits?
                gauss_fit = (self.last_status_df
                                     .at['Fit 3D Gaussians?', v])
                gauss_fit_val = 0 if self.isYes(gauss_fit) else 1
                self.do_gaussian_fit_rb_var.set(gauss_fit_val)

                self.hide_gauss_rb()
            except Exception as e:
                pass

            try:
                # Z resolution limit
                z_resolution_limit = (self.last_status_df
                                          .at['Z resolution limit (um):', v])
                self.z_resolution_limit_entry.delete(0, tk.END)
                self.z_resolution_limit_entry.insert(0, z_resolution_limit)
            except Exception as e:
                pass

            try:
                # Gaussian filter sigma
                gauss_sigma = (self.last_status_df
                                          .at['Gaussian filter sigma:', v])
                self.gauss_sigma_entry.delete(0, tk.END)
                self.gauss_sigma_entry.insert(0, gauss_sigma)
            except Exception as e:
                pass

            try:
                # Peak finder threshold function
                peak_finder_thresh_func = (self.last_status_df
                                         .at['Peak finder threshold function:', v])
                idx = self.thresh_methods.index(peak_finder_thresh_func)
                self.peak_finder_tresh_func_cb.current(idx)
                self.peak_finder_thresh_func_var.set(peak_finder_thresh_func)
            except Exception as e:
                pass

            try:
                # Reference channel thresholding function
                ref_ch_tresh_func = (self.last_status_df
                                    .at['Reference channel threshold function:', v])
                idx = self.thresh_methods.index(ref_ch_tresh_func)
                self.ref_ch_thresh_func_cb.current(idx)
                self.ref_ch_thresh_func_var.set(ref_ch_tresh_func)
            except Exception as e:
                pass

            try:
                # Filter good peaks method
                gop_method = (self.last_status_df
                                    .at['Filter good peaks method:', v])
                idx = self.gop_methods.index(gop_method)
                self.gop_methods_cb.current(idx)
                self.e_gop_var.set(gop_method)
            except Exception as e:
                pass

            try:
                # Which effect size to use
                which_effsize = (self.last_status_df
                                     .at['Effect size used:', v])
                idx = self.effect_size_values.index(which_effsize)
                self.effect_size_cb.current(idx)
                self.effect_size_cb_var.set(which_effsize)
            except Exception as e:
                pass

            try:
                # Gaussian filter sigma
                df = self.last_status_df
                i_idx = [i for i, idx in enumerate(df.index.str.find('limit:'))
                                      if idx != -1][0]
                gop_limit = self.last_status_df.iloc[i_idx][v]
                self.gop_limit_e.delete(0, tk.END)
                self.gop_limit_e.insert(0, gop_limit)
            except Exception as e:
                pass

            try:
                # YX resolution multiplier
                yx_multi = (self.last_status_df
                                          .at['YX resolution multiplier:', v])
                self.yx_multiplier_entry.delete(0, tk.END)
                self.yx_multiplier_entry.insert(0, yx_multi)
            except Exception as e:
                pass

            try:
                # spotsize_limit_pxl
                spotsize_limit_pxl = (self.last_status_df
                                          .at['Spotsize limits (pxl)', v])
                self.spotsize_limit_pxl_var.set(spotsize_limit_pxl)
            except Exception as e:
                pass


    def hide_ref_ch_inputs(self):
        if self.load_ref_ch_var.get() == 1:
            self.ref_ch_thresh_func_cb.grid_remove()
            self.entry6_label.grid_remove()
            self.calc_ref_ch_len_label.grid_remove()
            self.calc_ref_ch_len_yes_rb.grid_remove()
            self.calc_ref_ch_len_no_rb.grid_remove()
            self.filter_by_ref_label.grid_remove()
            self.filter_by_ref_yes_rb.grid_remove()
            self.filter_by_ref_no_rb.grid_remove()
            self.single_obj_ref_yes_rb.grid_remove()
            self.single_obj_ref_no_rb.grid_remove()
            self.single_obj_ref_label.grid_remove()
            self.filter_by_ref_var.set(1)
            self.calc_ref_ch_len_rb_var.set(1)
        elif self.load_ref_ch_var.get() == 0:
            self.ref_ch_thresh_func_cb.grid()
            self.entry6_label.grid()
            self.calc_ref_ch_len_label.grid()
            self.calc_ref_ch_len_yes_rb.grid()
            self.calc_ref_ch_len_no_rb.grid()
            self.filter_by_ref_label.grid()
            self.filter_by_ref_yes_rb.grid()
            self.filter_by_ref_no_rb.grid()
            self.single_obj_ref_yes_rb.grid()
            self.single_obj_ref_no_rb.grid()
            self.single_obj_ref_label.grid()

    def gop_var_cb(self, name=None, index=None, mode=None):
        if self.e_gop_var.get().find('effect') != -1:
            self.gop_limit_label.config(text='Effect size limit:')
            self.effect_size_cb.grid()
            self.effect_size_cb_label.grid()
            self.gop_limit_e.delete(0, tk.END)
            self.gop_limit_e.insert(0, '0.8')
        elif self.e_gop_var.get().find('t-test') != -1:
            self.gop_limit_label.config(text='p-value limit:')
            self.effect_size_cb.grid_remove()
            self.effect_size_cb_label.grid_remove()
            self.gop_limit_e.delete(0, tk.END)
            self.gop_limit_e.insert(0, '0.025')
        elif self.e_gop_var.get().find('peak_to_background') != -1:
            self.gop_limit_label.config(text='Peak/background limit:')
            self.effect_size_cb.grid_remove()
            self.effect_size_cb_label.grid_remove()
            self.gop_limit_e.delete(0, tk.END)
            self.gop_limit_e.insert(0, '1.5')

    def hide_gauss_rb(self):
        if self.do_spotSIZE_rb_var.get() == 1:
            self.do_gaussian_fit_no_rb.grid_remove()
            self.do_gaussian_fit_yes_rb.grid_remove()
            self.gauss_fit_rb_label.grid_remove()
            self.do_gaussian_fit_rb_var.set(1)
        elif self.do_spotSIZE_rb_var.get() == 0:
            self.do_gaussian_fit_no_rb.grid()
            self.do_gaussian_fit_yes_rb.grid()
            self.gauss_fit_rb_label.grid()


    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        exit('Execution aborted by the user')

    def _quit(self, event=None):
        valid_limits = (self.spotsize_limit_um_var.get() != 'invalid entry'
                    and self.spotsize_limit_pxl_var.get() != 'invalid entry')
        msg = ''
        if valid_limits:
            s = self.spotsize_limit_pxl_var.get()
            m = re.findall('([0-9]+([.][0-9]*)?|[.][0-9]+)', s)
            (a, _), (b, _) = m
            a = float(a)
            b = float(b)
            if a <= b:
                do_not_close = False
                self.spotsize_limits_pxl = (a, b)
            else:
                do_not_close = True
                msg = 'Your min value is greater than max value!'
        else:
            do_not_close = True

        if do_not_close:
            tk.messagebox.showerror('Invalid spot size limits',
                'INVALID spot size limits!\n\nValid limits are (0, 0) '
                'if you do not want to filter spots by size or (min, max) '
                f'where min < max is required.\n\n{msg}')
            return
        self.zyx_vox_size = self.zyx_vox_size_entry.get()
        self.yx_resolution_multiplier = self.yx_multiplier_entry.get()
        self.z_resol_limit = self.z_resolution_limit_entry.get()
        self.NA = self.NA_entry.get()
        self.em_wavel = self.em_wavel_entry.get()
        self.gauss_sigma = self.gauss_sigma_entry.get()
        self.local_max_thresh_func = self.peak_finder_thresh_func_var.get()
        self.ref_ch_thresh_func = self.ref_ch_thresh_func_var.get()
        self.gop_limit_txt = self.gop_limit_e.get()
        self.gop_how = self.e_gop_var.get()
        self.which_effsize = self.effect_size_cb_var.get()
        self.calc_ref_ch_len = self.calc_ref_ch_len_rb_var.get() == 0
        self.do_spotSIZE = self.do_spotSIZE_rb_var.get() == 0
        self.do_gaussian_fit = self.do_gaussian_fit_rb_var.get() == 0
        self.load_ref_ch = self.load_ref_ch_var.get() == 0
        self.is_ref_single_obj = self.single_obj_ref_var.get() == 0
        self.filter_by_ref_ch = self.filter_by_ref_var.get() == 0
        self.filter_z_bound = self.filter_z_bound_var.get() == 0
        self.save = self.save_rb_var.get() == 0
        self.make_sharper = self.sharpen_var.get() == 0
        self.local_or_global_thresh = self.local_or_global_var.get()
        self.segm_info = self.segm_info_var.get()
        self.zyx_resol_limit_vol_um = self.zyx_resol_limit_vol_um.cget("text")
        self.zyx_resol_limit_vol_vox = self.zyx_resol_limit_vol_vox.cget("text")
        zyx_vox_size_txt = self.zyx_vox_size
        self.entry_txts = [
                   zyx_vox_size_txt, self.z_resol_limit, self.NA,
                   self.em_wavel, self.gauss_sigma, self.ref_ch_thresh_func,
                   self.local_max_thresh_func, self.yx_resolution_multiplier,
                   self.gop_how, self.which_effsize,
                   self.gop_limit_txt, self.load_ref_ch,
                   self.filter_by_ref_ch, self.filter_z_bound,
                   self.make_sharper, self.local_or_global_thresh,
                   self.segm_info, self.do_spotSIZE, self.is_ref_single_obj,
                   self.calc_ref_ch_len, self.do_gaussian_fit,
                   self.zyx_resol_limit_vol_um, self.zyx_resol_limit_vol_vox,
                   self.spotsize_limit_um_var.get(),
                   self.spotsize_limit_pxl_var.get()]
        self.entry_labels.append(self.gop_limit_label.cget("text"))
        self.entry_labels.append(self.load_ref_ch_label_txt)
        self.entry_labels.append(self.filter_by_ref_label_txt)
        self.entry_labels.append(self.filter_z_bound_label_txt)
        self.entry_labels.append(self.sharpen_label_txt)
        self.entry_labels.append(self.local_or_global_label_txt)
        self.entry_labels.append(self.segm_info_label_txt)
        self.entry_labels.append(self.compute_spots_size_label_txt)
        self.entry_labels.append(self.single_obj_ref_label_txt)
        self.entry_labels.append(self.calc_ref_network_length_label_txt)
        self.entry_labels.append(self.do_gauss_fit_label_txt)
        self.entry_labels.append('ZYX minimum spot volume (um)')
        self.entry_labels.append('ZYX minimum spot volume (vox)')
        self.entry_labels.append('Spotsize limits (um)')
        self.entry_labels.append('Spotsize limits (pxl)')
        # self.entry9_txt = self.e9.get()
        self._root.quit()
        self._root.destroy()

    def save_analysis_inputs(self, data_path, vNUM, df_inputs):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        csv_path = f'{data_path}/{self.run_num}_{vNUM}_analysis_inputs.csv'
        df_inputs.to_csv(csv_path, encoding='utf-8-sig')
        return csv_path

    def load_prev_inputs(self, data_path, vNUM):
        filenames = os.listdir(data_path)
        string_to_find = f'{vNUM}_analysis_inputs.csv'
        self.analysis_inputs_nums = []
        self.analysis_inputs_vals_li = []
        for filename in filenames:
            if filename.find(string_to_find) != -1:
                csv_path = f'{data_path}/{filename}'
                inputs_vals = pd.read_csv(csv_path, index_col='Description'
                                         ).sort_index().astype('string')
                match = re.findall(f'(\d+)_{string_to_find}', filename)
                run_num = int(match[0])
                self.analysis_inputs_nums.append(run_num)
                self.analysis_inputs_vals_li.append(inputs_vals)

    def check_modules_already_performed(self):
        pass
        # CHECK which modules were already performed
        # widfets with possitbility to choose either repeat (more questions to come),
        # or keep data from previous run

    def count_number_of_output_files(self, data_path, run_num):
        filenames = os.listdir(data_path)
        count = 0
        for f in filenames:
            if f.startswith(f'{run_num}_'):
                count += 1
        return count > 10


    def check_if_same_of_prev_run(self, data_path, vNUM, df_inputs):
        self.load_prev_inputs(data_path, vNUM)
        iter = zip(self.analysis_inputs_nums, self.analysis_inputs_vals_li)
        if self.analysis_inputs_nums:
            self.run_num = max(self.analysis_inputs_nums)+1
        else:
            self.run_num = 1
            return True, False
        is_new_run = []
        for num, df in iter:
            if df_inputs.equals(df):
                same_run_found = self.count_number_of_output_files(
                                                            data_path, num)
                if same_run_found:
                    continue_how = fourbuttonsmessagebox()
                    continue_how.run('Previous run found',
                        'The analysis with the same input parameters\n'
                        f'was already performed at run number {num}\n\n'
                        'What should I do?\n',
                        'Do not save', 'Create new files', 'Replace',
                        'Show in Explorer', data_path)
                    do_save = not continue_how.do_save
                    replace = continue_how.replace
                    if replace:
                        self.run_num = num
                    return do_save, replace
            else:
                is_new_run.append(True)
        if all(is_new_run):
            continue_how = fourbuttonsmessagebox()
            n = len(self.analysis_inputs_nums)
            r = self.run_num
            num = max(self.analysis_inputs_nums)
            continue_how.run('Previous run found',
                f'There are already {n} analysis runs saved\n'
                'with DIFFERENT input parameters.\n\n'
                'If you choose to create new files they will have\n'
                f'run number "{r}" prepended.\n'
                f'Otherwise, run number {num} will be overwritten.\n\n'
                'What should I do?',
                'Do not save', 'Create new files', 'Replace',
                'Show in Explorer'
                , data_path)
            do_save = not continue_how.do_save
            replace = continue_how.replace
            if replace:
                self.run_num = num
            return do_save, replace
        return True, False

class analyse_subset_IDswidget:
    def __init__(self):
        self.testing_mode_ON = True

    def run(self, lab_3D, rp, intensity_imgs=[]):
        root = tk.Tk()
        root.lift()
        root.title('Analyse labels subset')
        # root.attributes("-topmost", True)
        root.geometry("+800+200")
        self.root = root

        self.intensity_imgs = intensity_imgs

        self.lab_3D = lab_3D
        self.rp = rp
        self.IDs = [obj.label for obj in rp]

        idx = -1

        # Labels selection mode
        idx += 1
        IDs_selection_mode_vals = [
            'all labels', 'random selection', 'manual selection',
        ]
        self.IDs_selection_mode_vals = IDs_selection_mode_vals
        self.IDs_selection_mode_label = tk.Label(root,
                           text='Labels selection mode:',
                           font=(None, 10))
        self.IDs_selection_mode_label.grid(row=idx, sticky=tk.E, pady=(10,0))
        IDs_selection_mode_var = tk.StringVar()
        self.IDs_selection_mode_var = IDs_selection_mode_var
        IDs_selection_mode_cb = ttk.Combobox(root,
                                     width=25,
                                     justify='center',
                                     textvariable=IDs_selection_mode_var)
        self.IDs_selection_mode_cb = IDs_selection_mode_cb
        IDs_selection_mode_cb.option_add('*TCombobox*Listbox.Justify', 'center')
        IDs_selection_mode_cb['values'] = IDs_selection_mode_vals
        IDs_selection_mode_cb.grid(row=idx, padx=(4, 10), pady=(10,0),
                                  column=1, columnspan=2)

        # select Labels entry
        idx += 1
        self.select_IDs_label = tk.Label(root,
                           text='Labels to analyse:',
                           font=(None, 10))
        self.select_IDs_label.grid(row=idx, sticky=tk.E, pady=(10,0),
                                   padx=(10,0))
        select_IDs_var = tk.StringVar()
        self.select_IDs_var = select_IDs_var
        self.select_IDs_entry = tk.Entry(root, justify='center', width=30,
                                         textvariable=self.select_IDs_var)
        if self.IDs:
            self.select_IDs_entry.insert(0, f'{self.IDs[0]},')
            self.current_select_IDs_val = f'{self.IDs[0]},'
        self.select_IDs_entry.grid(row=idx, padx=(10, 10), pady=(10,0),
                                   column=1, columnspan=2)

        # Ok button
        idx += 1
        ok_b = tk.Button(root, command=self._quit, text='Ok!', width=20)
        ok_b.grid(row=idx+1, pady=(16,0), column=1, columnspan=2)

        # Deactivate testing mode button
        off_b = tk.Button(root, command=self._off,
                          text='Deactivate testing mode',
                          width=20)
        off_b.grid(row=idx+1, pady=(16,0), column=0, padx=(10,0))

        # Ok button
        idx += 1
        show_lab_b = tk.Button(root, command=self._show_lab,
                               text='Visualize segmentation',
                               width=20)
        show_lab_b.grid(row=idx+1, pady=(6,16), column=1, columnspan=2)

        # Deactivate testing mode button
        abort_b = tk.Button(root, command=self.on_closing,
                          text='Abort',
                          width=20)
        abort_b.grid(row=idx+1, pady=(6,16), column=0, padx=(10,0))

        # Var events
        IDs_selection_mode_var.trace_add("write", self._selection_mode_callback)
        IDs_selection_mode_cb.current(2)

        self.select_IDs_var.trace_add("write", self._check_selection)

        root.bind('<Return>', self._quit)
        # root.protocol("WM_DELETE_WINDOW", self.on_closing)

        root.mainloop()

    def _show_lab(self):
        self.lab = self.lab_3D.max(axis=0)
        IDs = [obj.label for obj in regionprops(self.lab)]
        self.lab_RGB = label2rgb(self.lab, bg_label=0, bg_color=[0.1,0.1,0.1])
        for i, im in enumerate(self.intensity_imgs):
            img_i =  img_as_float(im.max(axis=0))
            img_i /= img_i.max()
            self.intensity_imgs[i] = img_i
        fig = apps.imshow_tk(self.lab_RGB,
                             additional_imgs=self.intensity_imgs,
                             run=False)
        self.fig = fig
        # Draw ID on object
        for obj in self.rp:
            _, y, x = obj.centroid
            fig.ax[0].text(x, y, f'{obj.label}', c='k', ha='center',
                           va='center', fontsize=12, fontweight='semibold')
        # Draw contours on intensity images
        for i in range(len(self.intensity_imgs)):
            contours = apps.auto_select_slice().find_contours(self.lab,
                                                          IDs, group=True)
            axi = fig.ax[i+1]
            for cont in contours:
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                axi.plot(x, y, c='r', alpha=0.5, lw=1, ls='--')
        for a in fig.ax:
            a.axis('off')
        self.fig.fig.suptitle('Right-click on a label (on any of the images) '
        'to highlight it.\nPress "esc" to exit from highlight mode.')
        fig.sub_win.canvas.mpl_connect('button_press_event', self._mouse_down)
        fig.sub_win.canvas.mpl_connect('key_press_event', self._key_down)
        fig.sub_win.canvas.draw()

    def _key_down(self, event):
        if event.key == 'escape':
            ax = self.fig.ax
            additional_aximgs = self.fig.additional_aximgs
            intensity_imgs = self.intensity_imgs
            for im, ax_img in zip(intensity_imgs, additional_aximgs):
                ax_img.set_data(im)
            self.fig.ax0img.set_data(self.lab_RGB)
            self.fig.sub_win.canvas.draw_idle()



    def _mouse_down(self, event):
        ax = self.fig.ax
        additional_aximgs = self.fig.additional_aximgs
        intensity_imgs = self.intensity_imgs
        lab = self.lab
        additional_ax_event = any([event.inaxes==a for a in ax])
        if event.inaxes == ax[0] and event.button==3:
            y, x = int(event.ydata), int(event.xdata)
            ID = lab[y, x]
            if ID != 0:
                for im, ax_img in zip(intensity_imgs, additional_aximgs):
                    img_focus = im.copy()
                    img_focus[lab!=ID] *= 0.2
                    ax_img.set_data(img_focus)
                    bg_mask = (lab!=ID) & (lab!=0)
                    lab_foregr = self.lab_RGB.copy()
                    lab_foregr[bg_mask] *= 0.2
                    self.fig.ax0img.set_data(lab_foregr)
            elif ID == 0:
                for im, ax_img in zip(intensity_imgs, additional_aximgs):
                    ax_img.set_data(im)
            self.fig.sub_win.canvas.draw_idle()
        elif additional_ax_event and event.button==3:
            y, x = int(event.ydata), int(event.xdata)
            ID = lab[y, x]
            if ID != 0:
                bg_mask = (lab!=ID) & (lab!=0)
                lab_foregr = self.lab_RGB.copy()
                lab_foregr[bg_mask] *= 0.1
                self.fig.ax0img.set_data(lab_foregr)
            elif ID == 0:
                self.fig.ax0img.set_data(self.lab_RGB)
            self.fig.sub_win.canvas.draw_idle()

    def _check_selection(self, name=None, index=None, mode=None):
        selection_mode = self.IDs_selection_mode_var.get()
        if selection_mode == 'manual selection':
            # Try to convert to integer any new number inserted in the entry
            s1 = self.current_select_IDs_val
            s2 = self.select_IDs_var.get()
            str_diff = difflib.ndiff(s1, s2)
            new_txt = [li for li in str_diff if li[0] != ' ']
            if '+ ,' in new_txt:
                self.current_select_IDs_val = s2
                num = ''.join([s[2:] for s in new_txt[:-1]])
                try:
                    ID = int(num)
                    if ID not in self.IDs:
                        tk.messagebox.showerror('Not a valid entry',
                            f'Label {ID} is not present in segmentation mask')
                except Exception as e:
                    tk.messagebox.showerror('Not valid entry',
                                    'Only integers separated by comma allowed')
            if any(['-' in s for s in new_txt]):
                self.current_select_IDs_val = s2
            if ',' not in s2:
                self.current_select_IDs_val = ''
        elif selection_mode == 'random selection':
            if self.select_IDs_var.get():
                try:
                    int(self.select_IDs_var.get())
                except Exception as e:
                    tk.messagebox.showerror('Not a valid entry',
                                    'Only ONE single integer allowed')
                    self.select_IDs_var.set('')

    def _selection_mode_callback(self, name=None, index=None, mode=None):
        selection_mode = self.IDs_selection_mode_var.get()
        if selection_mode == 'all labels':
            self.select_IDs_label.grid_remove()
            self.select_IDs_entry.grid_remove()
        elif selection_mode == 'random selection':
            self.select_IDs_label.grid()
            self.select_IDs_entry.grid()
            self.select_IDs_label.config(text='Number of labels to analyse:')
            self.select_IDs_var.set('1')
        elif selection_mode == 'manual selection':
            self.select_IDs_label.grid()
            self.select_IDs_entry.grid()
            self.select_IDs_label.config(text='Labels to analyse:')
            if self.IDs:
                self.select_IDs_var.set(f'{self.IDs[0]},')

    def _off(self):
        self.testing_mode_ON = False
        self._quit()

    def _quit(self, event=None):
        self.selection_mode = self.IDs_selection_mode_var.get()
        self.selection_entry_txt = self.select_IDs_var.get()
        if self.selection_mode == 'manual selection':
            s = self.selection_entry_txt
            r = re.compile('(\d+),?\s*')
            m = r.match(s)
            if m is not None and m.span()[1] == len(s):
                self.selection_entry_txt = r.findall(s)[0]
                check_again = False
                close = True
            else:
                check_again = True
                close = False
            if check_again:
                li = self.selection_entry_txt.split(',')
                try:
                    [int(id) for id in li]
                    close = True
                except Exception as e:
                    tk.messagebox.showerror('Not a valid entry',
                        'You chose to manually select which labels to analyse.\n'
                        'Only integers separated by a comma are allowed')
                    close = False
        elif self.selection_mode == 'random selection':
            try:
                int(self.selection_entry_txt)
                close = True
            except Exception as e:
                tk.messagebox.showerror('Not a valid entry',
                    'You chose to analyse randomly selected labels.\n'
                    'Only a SINGLE integer is allowed for the number of '
                    'labels to analyse entry')
                close = False
        else:
            close = True
        if close:
            try:
                self.fig.sub_win.root.quit()
                self.fig.sub_win.root.destroy()
            except Exception as e:
                pass
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        exit('Execution aborted by the user')
        # try:
        #     self.fig.quit()
        #     self.fig.destroy()
        # except Exception as e:
        #     pass
        # self.root.quit()
        # self.root.destroy()


def check_img_shape_vs_metadata(img_shape, num_frames, SizeT, SizeZ):
    msg = ''
    if num_frames > 1:
        data_T, data_Z = img_shape[:2]
        ndim_msg = 'Data is expected to be 4D with TZYX order'
    else:
        data_T, data_Z = 1, img_shape[0]
        expected_data_ndim = 3
        ndim_msg = 'Data is expected to be 3D with ZYX order'
    if data_T != SizeT:
        msg = (f'{ndim_msg}.\nData shape is {img_shape} '
        f'(i.e. {data_T} frames), but the metadata of the '
        f'.tif file says that there should be {SizeT} frames.\n\n'
        f'Process cannot continue.')
        tk.messagebox.showerror('Shape mismatch!', msg)
    if data_Z != SizeZ:
        msg = (f'{ndim_msg}.\nData shape is {img_shape} '
        f'(i.e. {data_Z} z-slices), but the metadata of the '
        f'.tif file says that there should be {SizeZ} z-slices.\n\n'
        f'Process cannot continue.')
        tk.messagebox.showerror('Shape mismatch!', msg)
    return msg.replace('\n', ' ')

class single_entry_messagebox:
    def __init__(self, title='Entry', entry_label='Entry 1', input_txt='',
                       toplevel=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.title(title)
        # root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0, padx=8)
        w = len(input_txt)+10
        w = w if w>40 else 40
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

class dual_entry_messagebox:
    def __init__(self, title='Entry',
                       entry_label='Entry 1', input_txt='',
                       entry_label2='Entry 2', input_txt2='',
                       toplevel=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.title(title)
        # root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0, padx=8)
        w = len(input_txt)+10
        w = w if w>40 else 40
        e = tk.Entry(root, justify='center', width=w)
        e.grid(row=1, padx=16, pady=4)
        e.focus_force()
        e.insert(0, input_txt)

        tk.Label(root, text=entry_label2, font=(None, 10)).grid(row=2, padx=8)
        entry2 = tk.Entry(root, justify='center', width=w)
        entry2.grid(row=3, padx=16, pady=4)
        entry2.focus_force()
        entry2.insert(0, input_txt2)

        tk.Button(root, command=self._quit, text='Ok!', width=10).grid(row=4,
                                                                      pady=4)
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.e = e
        self.entry2 = entry2
        root.mainloop()

    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        exit('Execution aborted by the user')

    def _quit(self, event=None):
        self.entries_txt = (self.e.get(), self.entry2.get())
        self._root.quit()
        self._root.destroy()

def askyesno(title='tk', message='Yes or no?', toplevel=False):
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    yes = tk.messagebox.askyesno(title, message, master=root)
    if not toplevel:
        root.quit()
        root.destroy()
    return yes



if __name__ == '__main__':
    num_frames_prompt = num_frames_toQuant()
    num_frames_prompt.prompt(100, last_segm_i=100, last_tracked_i=85)
