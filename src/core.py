import numpy as np
import sys
import traceback
import time
import os
import shutil
import subprocess
import re
import tempfile
from sys import exit
import scipy.ndimage as nd
from scipy import stats
from scipy.special import erf
from scipy.optimize import least_squares
from ast import literal_eval
import cv2
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import tkinter as tk
from tkinter import N, S, E, W, END, ttk
import pandas as pd
import math
from math import sqrt
from itertools import compress
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from skimage import io, exposure
import skimage.util
import skimage.segmentation
from skimage.draw import circle, circle_perimeter
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.feature import register_translation
from skimage.transform import rotate
from skimage.morphology import skeletonize
from skimage.exposure import match_histograms
from skimage.filters import (sobel, unsharp_mask, threshold_li, gaussian,
                     try_all_threshold, threshold_isodata, threshold_multiotsu)
# from joblib import Parallel, delayed
from tifffile import TiffFile
from datetime import datetime, timedelta
from natsort import natsorted
import scipy.ndimage as ndi
from scipy import stats
from scipy.optimize import leastsq, curve_fit
from scipy.special import erf
from datetime import datetime
from tqdm import tqdm
from numba import jit, njit, prange
import apps #, unet

# src_path = os.path.dirname(os.path.realpath(__file__))
# gmail_path = os.path.join(src_path, 'gmail')
# sys.path.insert(0, gmail_path)

try:
    from gmail import gmail
except:
    pass

def gmail_send(to, subject, message_text):
    service = gmail.access_gmail()
    message = gmail.create_message('elpado6872@gmail.com', to, subject,
                                    message_text)
    gmail.send_message(service, 'elpado6872@gmail.com', message)

class num_frames_toQuant_tk:
    def __init__(self, tot_frames):
        root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        root.geometry('+800+400')
        root.attributes("-topmost", True)
        tk.Label(root,
                 text="How many frames do you want to QUANT?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        tk.Label(root,
                 text="(There are a total of {} segmented frames).".format(tot_frames),
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start frame",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=tk.E, padx=4)
        tk.Label(root,
                 text="# of frames to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        self.time_elapsed_sv = tk.StringVar()
        time_elapsed_label = tk.Label(root,
                         textvariable=self.time_elapsed_sv,
                         font=(None, 10)).grid(row=5, column=0,
                                               columnspan=3, padx=4, pady=4)
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
                                 columnspan=1, sticky=tk.E)
        tk.Button(root,
                  text='OK for all positions',
                  command=self.ok_all).grid(row=4,
                                            column=1,
                                            pady=8,
                                            columnspan=2)
        root.bind('<Return>', self.ok)
        root.focus_force()
        start_frame.focus_force()
        start_frame.selection_range(0, tk.END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.bind('<Enter>', self.stop_timer)
        self.timer_t_final = time.time() + 10
        self.auto_close = True
        self.ok_for_all = False
        self.tk_timer()
        root.mainloop()

    def stop_timer(self, event):
        self.auto_close = False

    def ok_all(self, event=None):
        self.ok_for_all = True
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf, stopf)
            self.root.quit()
            self.root.destroy()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            if startf >= self.tot_frames:
                self.start_frame.delete(0, tk.END)
                self.start_frame.insert(0, '{}'.format(self.tot_frames-1))
                startf = self.tot_frames-1
                self.time_elapsed_sv.set('NOTE: Frame count starts at 0.\n'
                'This means that last frame index is {}'.format(self.tot_frames-1))
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

    def tk_timer(self):
        if self.auto_close:
            seconds_elapsed = self.timer_t_final - time.time()
            seconds_elapsed = int(round(seconds_elapsed))
            if seconds_elapsed <= 0:
                print('Time elpased. Analysing all frames')
                self.ok()
            self.time_elapsed_sv.set('Window will close automatically in: {} s'
                                                       .format(seconds_elapsed))
            self.root.after(1000, self.tk_timer)
        else:
            self.time_elapsed_sv.set('')

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class num_pos_toQuant_tk:
    def __init__(self, tot_frames):
        root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        root.geometry('+800+400')
        root.attributes("-topmost", True)
        tk.Label(root,
                 text="How many positions do you want to QUANT?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        tk.Label(root,
                 text="(There are a total of {} positions).".format(tot_frames),
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start position",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=E, padx=4)
        tk.Label(root,
                 text="# of positions to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                            textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(1))
        sv_sf.trace_add("write", self.set_all)
        self.start_frame = start_frame
        start_frame.grid(row=2, column=1, pady=8, sticky=W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=W)
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
        start_frame.selection_range(0, END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            if startf > self.tot_frames:
                self.start_frame.delete(0, END)
                self.start_frame.insert(0, '{}'.format(self.tot_frames))
                startf = self.tot_frames
            rightRange = self.tot_frames - startf + 1
            self.num_frames.delete(0, END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf + 1
                self.num_frames.delete(0, END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def ok(self, event=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf-1, stopf-1)
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

def calc_resolution_limited_vol(wavelen, NA, yx_resolution_multi,
                                zyx_vox_dim, z_resolution_limit):
    airy_radius = (1.22 * wavelen)/(2*NA)
    sphere_radius = airy_radius*1E-3 #convert nm to µm
    yx_resolution = sphere_radius*yx_resolution_multi
    zyx_resolution = np.asarray([z_resolution_limit,
                                 yx_resolution, yx_resolution])
    zyx_resolution_pxl = zyx_resolution/np.asarray(zyx_vox_dim)
    return zyx_resolution, zyx_resolution_pxl, airy_radius

class align_frames:
    def __init__(self, frames, path=None, save_aligned=False, register=True,
                 saved_shifts=None):
        self.register = register
        if register:
            self.frames, all_shifts = self.align_by_reg_trans(frames)
            self.shifts = all_shifts
        else:
            self.shifts = saved_shifts
            self.frames = self.align_by_saved_shifts(frames, saved_shifts)
        if save_aligned:
            if path is not None:
                self.save_aligned_npy(path, self.frames)
            else:
                raise TypeError('Path argument of align_frames class '
                                'cannot be None')

    def align_by_reg_trans(self, frames):
        frames_aligned = np.copy(frames)
        registered_shifts = np.zeros((len(data),2), int)
        for frame_i, frame_V in enumerate(data):
            if frame_i != 0:  # skip first frame
                curr_frame_img = frame_V[slice]
                prev_frame_img = frames_aligned[frame_i-1, slice] #previously aligned frame, slice
                shifts = register_translation(prev_frame_img, curr_frame_img)[0]
                shifts = shifts.astype(int)
                aligned_frame_V = np.copy(frame_V)
                aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(1,2))
                frames_aligned[frame_i] = aligned_frame_V
                registered_shifts[frame_i] = shifts
        return frames_aligned, registered_shifts

    def align_by_saved_shifts(self, frames, saved_shifts):
        frames_aligned = np.zeros_like(frames)
        for i, frame in enumerate(frames):
            shifts = saved_shifts[i]
            # print(frame.shape)
            frame_aligned = np.roll(frame, tuple(shifts), axis=(1,2))
            frames_aligned[i] = frame_aligned
        return frames_aligned

    def save_aligned_npy(self, path, frames_aligned):
        position_n_path = os.path.dirname(path)
        filename = os.path.basename(path)
        filename_noEXT, filename_extension = os.path.splitext(filename)
        aligned_npy_path = position_n_path+'/'+filename_noEXT + '_aligned.npy'
        print('Saving aligned frames to {}'.format(aligned_npy_path))
        np.save(aligned_npy_path, frames_aligned, allow_pickle=False)


class analyze_skeleton:
    # similar idea to https://imagej.net/AnalyzeSkeleton
    def __init__(self, skel, zyx_dim):
        if skel.dtype != bool:
            raise TypeError('Skeleton image array is not \'boolean\' data type.')
        self.bp = apps.tk_breakpoint()
        self.coords_ld = [] # list of dict {'Fragment_idx','Segment_idx','coords_fs'}
        lab_skel, num_fragm = label(skel, return_num=True)
        self.num_fragm = num_fragm
        self.lab_skel = lab_skel

    def get_tot_length(self):
        tot_length = self.df['Length (µm)'].sum()
        return tot_length

    def analyse(self, skel, zyx_dim):
        lab_skel, num_fragm = self.lab_skel, self.num_fragm
        rp_skel = regionprops(lab_skel)
        fragment_IDs = [0]*num_fragm
        df_f = [0]*num_fragm
        all_ord_coords = []
        all_junctions = []
        all_endpoints = []
        for f, fragm_rp in enumerate(rp_skel):
            fragment_IDs[f] = fragm_rp.label
            thresh_f_noj = np.zeros_like(skel)
            coords = fragm_rp.coords
            thresh_f_noj[coords[:,0], coords[:,1], coords[:,2]] = True
            thresh_f_noj, _, junctions, _ = self.classify(fragm_rp, skel,
                                                      junc_bbox=True,
                                                      skel_nojunc=thresh_f_noj)
            all_junctions.extend(junctions)
            lab_f_noj = label(thresh_f_noj)
            rp_lab_f_noj = regionprops(lab_f_noj)
            for s, s_rp in enumerate(rp_lab_f_noj):
                thresh_fs = np.zeros_like(skel)
                coords = s_rp.coords
                thresh_fs[coords[:,0], coords[:,1], coords[:,2]] = True
                _, endpoints, _, slabs = self.classify(s_rp, thresh_fs)
                all_endpoints.extend(endpoints)
                ord_coords = self.order_coords(endpoints, thresh_fs)
                all_ord_coords.extend(ord_coords) # list of (z,y,x) coords
                fs_dict = {'F_idx': f, 'S_idx': s, 'ord_coords_fs': ord_coords}
                self.coords_ld.append(fs_dict)
            # df_f[f] = df_s
        coords_ld = self.assign_junc(all_junctions, all_endpoints, self.coords_ld)
        self.coords_ld = coords_ld
        self.df, self.df_zyx = self.create_df(coords_ld, zyx_dim)

    def create_df(self, coords_ld, zyx_dim):
        num_fs = len(coords_ld)
        fragment_IDs = [0] * num_fs
        segment_IDs = [0] * num_fs
        length_um = [0] * num_fs
        start_x = [0] * num_fs
        start_y = [0] * num_fs
        start_z = [0] * num_fs
        end_x = [0] * num_fs
        end_y = [0] * num_fs
        end_z = [0] * num_fs
        df_coords_li, keys = [0] * num_fs, [0] * num_fs
        for i, d in enumerate(coords_ld):
            ord_coords_fs = d['ord_coords_fs']
            coords_2D = np.asarray(ord_coords_fs)
            fragm_ID = d['F_idx']+1
            segm_ID = d['S_idx']+1
            keys[i] = (fragm_ID, segm_ID)
            df_fs_coords = pd.DataFrame({'z': coords_2D[:,0],
                                         'y': coords_2D[:,1],
                                         'x': coords_2D[:,2]})
            df_coords_li[i] = df_fs_coords
            fragment_IDs[i] = d['F_idx']+1
            segment_IDs[i] = d['S_idx']+1
            length_um[i] = self.calc_length(d['ord_coords_fs'], zyx_dim)
            start_x[i] = ord_coords_fs[0][2]
            start_y[i] = ord_coords_fs[0][1]
            start_z[i] = ord_coords_fs[0][0]
            end_x[i] = ord_coords_fs[-1][2]
            end_y[i] = ord_coords_fs[-1][1]
            end_z[i] = ord_coords_fs[-1][0]
        df = pd.DataFrame({'Fragment_ID': fragment_IDs,
                            'Segment_ID': segment_IDs,
                            'Length (µm)': length_um,
                            'Start_x': start_x,
                            'Start_y': start_y,
                            'Start_z': start_z,
                            'End_x': end_x,
                            'End_y': end_y,
                            'End_z': end_z})
        df_zyx = pd.concat(df_coords_li, keys=keys)
        df_zyx.index.set_names(['Fragm_ID', 'Segm_ID', 'zyx_idx'], inplace=True)
        return df, df_zyx

    def empty_df_zyx(self):
        df_zyx = pd.DataFrame({'Fragm_ID': [0], 'Segm_ID': [0], 'zyx_idx': [0],
                               'z': [0], 'y': [0], 'x': [0]})
        df_zyx.set_index(['Fragm_ID', 'Segm_ID', 'zyx_idx'], inplace=True)
        return df_zyx


    def calc_length(self, coords_lt, zyx_dim):
        length_um = 0
        prev_coords = coords_lt[0]
        for coords in coords_lt:
            dist = np.linalg.norm((np.asarray(prev_coords)-
                                   np.asarray(coords))*zyx_dim)
            length_um += dist
            prev_coords = coords
        return length_um

    def classify(self, rp, skel, junc_bbox=False, skel_nojunc=None):
        endpoints = []
        junctions = []
        slabs = []
        for z, y, x in rp.coords:
            neigh = skel[z-1:z+2, y-1:y+2, x-1:x+2]
            num_neigh = neigh.sum() - 1
            if num_neigh < 2:
                endpoints.append((z,y,x))
            elif num_neigh > 2:
                if junc_bbox:
                    junc_cube = skel[z-1:z+2, y-1:y+2, x-1:x+2]
                    zzjc, yyjc, xxjc = np.nonzero(junc_cube)
                    junctions.extend([(zjc+z-1, yjc+y-1, xjc+x-1)
                                    for zjc, yjc, xjc in zip(zzjc, yyjc, xxjc)])
                else:
                    junctions.append((z,y,x))
                if skel_nojunc is not None:
                    skel_nojunc[z-1:z+2, y-1:y+2, x-1:x+2] = False
            else:
                slabs.append((z,y,x))
        return skel_nojunc, endpoints, junctions, slabs

    def order_coords(self, endpoints, thresh, ordered_skel=None):
        if len(endpoints) == 1:
            coords = endpoints
        elif len(endpoints) == 2:
            start = endpoints[0]
            end = endpoints[1]
            point = start
            coords = [start]
            if ordered_skel is not None:
                ordered_skel[start] = True
            while point != end:
                z, y, x = point
                neigh = thresh[z-1:z+2, y-1:y+2, x-1:x+2]
                zzn, yyn, xxn = np.nonzero(neigh)
                for zn, yn, xn in zip(zzn, yyn, xxn):
                    temp_p = (zn+z-1, yn+y-1, xn+x-1)
                    if temp_p not in coords:
                        coords.append(temp_p)
                        point = temp_p
                        if ordered_skel is not None:
                            ordered_skel[z,y,x] = True
            # coords.append(end)
            if ordered_skel is not None:
                ordered_skel[end] = True
        else:
            coords = endpoints
            print('WARNING: segment with more than two endpoints found!'
                  'It will not contribute to overall length calculation!')
        return coords

    def assign_junc(self, all_junctions, all_endpoints, coords_ld):
        coords_ld_copy = deepcopy(coords_ld)
        ee = np.asarray(all_endpoints)
        for j, junc in enumerate(all_junctions):
            J = np.asarray(junc)
            dist_junc_all_end = np.linalg.norm(np.subtract(J, ee), axis=1)
            nearest_endp = all_endpoints[dist_junc_all_end.argmin()]
            for i, d in enumerate(coords_ld):
                li_coords = d['ord_coords_fs']
                if nearest_endp in li_coords:
                    idx = li_coords.index(nearest_endp)
                    if idx == 0:
                        coords_ld_copy[i]['ord_coords_fs'].insert(0, junc)
                    elif idx == len(li_coords)-1:
                        coords_ld_copy[i]['ord_coords_fs'].append(junc)
                    else:
                        print('WARNING: Neither a start nor an end point were '
                            'closet to junction at index {} It will not be '
                            'added to the list of coords.'.format(j))
                    # self.bp.pausehere()
                    break
        return coords_ld_copy

# Function to apply segmentation file as a mask
def apply_segmentation_mask(segm_npy, V, cell_regionprops, bbox_or_black=True):
    cell_shape = V[0][cell_regionprops.slice].shape #get the bbox shape of the current cell (2D)
    V_mask = np.zeros((V.shape[0], cell_shape[0], cell_shape[1]), int)
    background = np.zeros((cell_shape[0], cell_shape[1]), int)
    print('New shape after applying segmentation mask: {}'.format(V_mask.shape))
    for i, image in enumerate(V):
        if bbox_or_black:
            V_mask[i] = image[cell_regionprops.slice]
        else:
            image_sliced = image[cell_regionprops.slice]
            segm_npy_sliced = segm_npy[cell_regionprops.slice]
            background[segm_npy_sliced == cell_regionprops.label] = image_sliced[segm_npy_sliced == cell_regionprops.label]
            V_mask[i] = background
    return V_mask


# Normalize array to (0,1)
def norm(a):
    return (a-np.min(a))/np.ptp(a)


# This function takes a 2D array of coordinates and returns a 1D array of (z,y,x)
# or (y,x) tuples. If the input array is a 2D array of 3 or 2 columns
# then transpose is required
def obj_coords2Dto1Dtuples(obj_2Dcoords,V3D,order='zyx'):
    # check if the input obj_2Dcoords is a tuple (such as np.where() output)
    # or a 2D array such as coords attribute of regionprops
    if type(obj_2Dcoords) is tuple:
        array2D = False
    else:
        array2D = True
        #check if transpose is required (number of rows is 3 or 2 depending on V3D)
        if (obj_2Dcoords.shape[1] == 3 and V3D) or (obj_2Dcoords.shape[1] == 2 and not V3D):
            obj_2Dcoords = np.transpose(obj_2Dcoords)

    object_size = obj_2Dcoords[0].size
    if V3D:
        if order == 'zyx':
            Z, Y, X = obj_2Dcoords[0], obj_2Dcoords[1], obj_2Dcoords[2]
        elif order == 'xyz':
            X, Y, Z = obj_2Dcoords[0], obj_2Dcoords[1], obj_2Dcoords[2]
        D = 3
        obj_a = np.transpose(np.stack((Z,Y,X)))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0, 0)
        obj_tuplecoords = np.full((object_size), coord, dtype=object)
    else:
        if order == 'yx':
            Y, X = obj_2Dcoords[0], obj_2Dcoords[1]
        elif order == 'xy':
            X, Y = obj_2Dcoords[0], obj_2Dcoords[1]
        D = 2
        obj_a = np.transpose(np.stack((Y,X)))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0)
        obj_tuplecoords = np.full((object_size), coord, dtype=object)
    for c in range(object_size):
         obj_tuplecoords[c] = tuple(obj_a[c])
    return obj_tuplecoords


# This function converts a 1D array of tuples coordinates (z,y,x) or (y,x)
# into a 2D array with 3 or 2 columns [z][y][x]
def obj_1Dtuplesto2Dcoords(obj_tuplecoords_ith):
    V3D = len(obj_tuplecoords_ith[0]) == 3
    if V3D:
        obj_2Dcoords = np.zeros((len(obj_tuplecoords_ith),3))
        row = 0
        for coords in obj_tuplecoords_ith:
            obj_2Dcoords[row,0] = coords[0]
            obj_2Dcoords[row,1] = coords[1]
            obj_2Dcoords[row,2] = coords[2]
            row += 1
    else:
        obj_2Dcoords = np.zeros((len(obj_tuplecoords_ith),2))
        row = 0
        for coords in obj_tuplecoords_ith:
            obj_2Dcoords[row,0] = coords[0]
            obj_2Dcoords[row,1] = coords[1]
            row += 1
    return obj_2Dcoords


# Initialize euclidean distances between neighbours
def init_euclid_distance(zyx_vox_dim):
    D = len(zyx_vox_dim)
    if D == 3:
        eucl_dist_coords_3D_unit = np.array([[0,0,1],\
                                             [0,1,0],\
                                             [0,1,1],\
                                             [1,0,0],
                                             [1,0,1],\
                                             [1,1,0],\
                                             [1,1,1]])
        eucl_dist_coords_3D_resol = eucl_dist_coords_3D_unit*zyx_vox_dim
        eucl_dist_table = np.linalg.norm(eucl_dist_coords_3D_resol, axis=1)
    elif D == 2:
        eucl_dist_coords_2D_unit = np.array([[0,1],\
                                             [1,0],\
                                             [1,1]])
        eucl_dist_coords_2D_resol = eucl_dist_coords_2D_unit*zyx_vox_dim
        eucl_dist_table = np.linalg.norm(eucl_dist_coords_2D_resol, axis=1)
    return eucl_dist_table


# From the initialized euclidean distances index the specific eucliden distance
# between source and target
def lookup_eucl_dist(eucl_dist_table, source, target):
    unit = np.abs(np.asarray(source)-np.asarray(target))
    unit_idx = np.sum(unit*[4,2,1])-1 #index of the euclidean distance in the lookup table eucl_dist_table
    eucl_dist_source_target = eucl_dist_table[unit_idx]
    return eucl_dist_source_target


# Function that given the range of each slice returns the indexes of all neighboring elements. Range is a list of 2 or 3 elements made of tuples. e.g. [(Ys,Ye),(Xs,Xe)].
# Output indexes is a list of tuples with each tuple containing the coordinates of the neighboring elements
def all_neigh_indexes(ranges,N,D):
    if D == 2:
        Ys, Ye, Xs, Xe = ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]
        y, x = Ys+N, Xs+N
        l_slice = len(range(Xs,Xe))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0)
        indexes = np.full((D*2, l_slice), coord, dtype=object) #initialize a 2D array where each row is a slice and each element of the row is a tuple of (y,x) coordinates
        for row in range(D*2):
            if row == 0:
                count = 0 #count is the index of the elements of each slice
                for xi in range(Xs,Xe):
                    indexes[row,count] = (y-N,xi)
                    count += 1
            elif row == 1:
                count = 0 #count is the index of the elements of each slice
                for xi in range(Xs,Xe):
                    indexes[row,count] = (y+N, xi)
                    count += 1
            elif row == 2:
                count = 0 #count is the index of the elements of each slice
                for yi in range(Ys,Ye):
                    indexes[row,count] = (yi, x-N)
                    count += 1
            else:
                count = 0 #count is the index of the elements of each slice
                for yi in range(Ys,Ye):
                    indexes[row,count] = (yi, x+N)
                    count += 1
    if D == 3:
        Zs, Ze, Ys, Ye, Xs, Xe = ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1], ranges[2][0], ranges[2][1]
        z, y, x = Zs+N, Ys+N, Xs+N
        size_slice = len(range(Xs,Xe)) #each slice is a 2D square matrix with side = len(range(Xs,Xe))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0, 0)
        indexes = np.full((D*2, size_slice, size_slice), coord, dtype=object) #initialize a 3D array where each page is a slice composed of a 2D square matrix where elements are tuples of (z,y,x) coordinates
        for slice in range(D*2):
            if slice == 0:
                row = 0
                for yi in range(Ys,Ye):
                    count = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                    for xi in range(Xs,Xe):
                        indexes[slice, row, count] = (z-N,yi,xi)
                        count += 1
                    row += 1
            elif slice == 1:
                row = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                for yi in range(Ys,Ye):
                    count = 0 #count is the index of the elements of each slice
                    for xi in range(Xs,Xe):
                        indexes[slice, row, count] = (z+N,yi,xi)
                        count += 1
                    row += 1
            elif slice == 2:
                row = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                for zi in range(Zs,Ze):
                    count = 0 #count is the index of the elements of each slice
                    for xi in range(Xs,Xe):
                        indexes[slice, row, count] = (zi,y-N,xi)
                        count += 1
                    row += 1
            elif slice == 3:
                row = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                for zi in range(Zs,Ze):
                    count = 0 #count is the index of the elements of each slice
                    for xi in range(Xs,Xe):
                        indexes[slice, row, count] = (zi,y+N,xi)
                        count += 1
                    row += 1
            elif slice == 4:
                row = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                for yi in range(Ys,Ye):
                    count = 0 #count is the index of the elements of each slice
                    for zi in range(Zs,Ze):
                        indexes[slice, row, count] = (zi,yi,x-N)
                        count += 1
                    row += 1
            else:
                row = 0 #since each slice is a 2D array it has two counters for rows and columns: row, count
                for yi in range(Ys,Ye):
                    count = 0 #count is the index of the elements of each slice
                    for zi in range(Zs,Ze):
                        indexes[slice, row, count] = (zi,yi,x+N)
                        count += 1
                    row += 1
    return np.unique(indexes)


# Function that returns every neighboring pixel/voxel of the element (all=True)
# or every neighbour with the same label as the element (all=False).
# The output is a 1D array or list with coordinates coords = (z,y,x).
# N is the current number of 'rings' around the pixel/voxel. N=1 return 8-connectivity neighbours for 2D array and 26-connectivity neighbours for 3D arrays
# The neighboring elements are obtained by slicing the original array at the "borders" of the previous ring
def all_neigh(a,coords,N,all=True,edges_length=False,eucl_dist_table = None):
    D = len(coords)
    coords = tuple(coords)
    neigh_size = (2*N+1)**D-(2*N-1)**D #total number of neighbouring elements at ring number N
    label = a[coords]
    if D == 2:
        zyx_vox_dim = (zyx_vox_dim[1], zyx_vox_dim[2])
        y, x = coords
        if x-N<0 or y-N<0 or x+N+1>a.shape[1] or y+N+1>a.shape[0]:
            N = N-1
            stop = True #Since we reached the absolute borders of the image we stop the process
        else:
            N = N
            stop = False
        Xs, Xe = x-N, x+N+1 #start and end coords of the slice
        Ys, Ye = y-N, y+N+1
        if edges_length:
            neigh = np.zeros(neigh_size) #since neigh will return euclidean distances it needs to have floating elements data type
        else:
            neigh = np.zeros(neigh_size).astype(np.uint32)
        indexes = all_neigh_indexes([(Ys,Ye),(Xs,Xe)],N,D)
        indexes_label = [] #initialize empty list that will be populated with tuple coords of only the neighbours that has the same value of the center element (if all=False)
        neigh_label = [] #initialize empty list that will be populated with only the neighbours values equal to the center element's value (if all=False)
        i_coords = 0
        for neigh_coords in indexes:
            if all and not edges_length:
                neigh[i_coords] = a[neigh_coords] #all is True and edges_length length is False --> return neigh as values and all neighbours coords
                i_coords += 1
            elif all and edges_length:
                dist = lookup_eucl_dist(eucl_dist_table, coords, neigh_coords)
                neigh[i_coords] = dist #all is True and edges_length length is True --> return neigh as euclidean distances and all neighbours coords
                i_coords += 1
            elif not all and not edges_length:
                if a[neigh_coords] == label:
                    indexes_label.append(neigh_coords)
                    neigh_label.append(a[neigh_coords]) #all is False and edges_length length is False --> return neigh as values and only labelled neighbours coords
            elif not all and edges_length:
                if a[neigh_coords] == label:
                    indexes_label.append(neigh_coords) #all is False and edges_length length is True --> return neigh as euclidean distances and only labelled neighbours coords
                    dist = lookup_eucl_dist(eucl_dist_table, coords, neigh_coords)
                    neigh_label.append(dist)
        if not all:
            indexes = indexes_label
            neigh = neigh_label
    elif D == 3:
        z, y, x = coords
        if x-N<0 or y-N<0 or z-N<0 or x+N+1>a.shape[2] or y+N+1>a.shape[1] or z+N+1>a.shape[0]:
            N = N-1
            stop = True #Since we reached the absolute borders of the image we stop the process
        else:
            N = N
            stop = False
        Xs, Xe = x-N, x+N+1
        Ys, Ye = y-N, y+N+1
        Zs, Ze = z-N, z+N+1
        if edges_length:
            neigh = np.zeros(neigh_size) #since neigh will return euclidean distances it needs to have floating elements data type
        else:
            neigh = np.zeros(neigh_size).astype(np.uint32)
        indexes = all_neigh_indexes([(Zs,Ze),(Ys,Ye),(Xs,Xe)],N,D)
        indexes_label = [] #initialize empty list that will be populated with tuple coords of only the neighbours that has the same value of the center element (if all=False)
        neigh_label = [] #initialize empty list that will be populated with only the neighbours values equal to the center element's value (if all=False)
        i_coords = 0
        for neigh_coords in indexes:
            if all and not edges_length:
                neigh[i_coords] = a[neigh_coords] #all is True and edges_length length is False --> return neigh as values and all neighbours coords
                i_coords += 1
            elif all and edges_length:
                dist = lookup_eucl_dist(eucl_dist_table, coords, neigh_coords)
                neigh[i_coords] = dist #all is True and edges_length length is True --> return neigh as euclidean distances and all neighbours coords
                i_coords += 1
            elif not all and not edges_length:
                if a[neigh_coords] == label:
                    indexes_label.append(neigh_coords)
                    neigh_label.append(a[neigh_coords]) #all is False and edges_length length is False --> return neigh as values and only labelled neighbours coords
            elif not all and edges_length:
                if a[neigh_coords] == label:
                    indexes_label.append(neigh_coords) #all is False and edges_length length is True --> return neigh as euclidean distances and only labelled neighbours coords
                    dist = lookup_eucl_dist(eucl_dist_table, coords, neigh_coords)
                    neigh_label.append(dist)
        if not all:
            indexes = indexes_label
            neigh = neigh_label
    return neigh, indexes, a, stop #indexes is a 1D array (or list if All=False) of tuples with each tuple containing the coordinates of the neighboring elements


#Function that returns the coordinates of the max intensity of each object as a 2D array of 2 or 3 columns
def obj_max_intensity_2Dcoords(obj_max_intensity, obj_intensity_images, obj_bbox):
    V3D = len(obj_bbox[0]) == 6
    if V3D:
        obj_max_intensity_2Dcoords = np.zeros((obj_max_intensity.size,3))
        row = 0
        for max_intensity, intensity_image, bbox in zip(obj_max_intensity, obj_intensity_images, obj_bbox):
            zyx_max = np.where(intensity_image == max_intensity)
            obj_max_intensity_2Dcoords[row,0] = zyx_max[0][0] + bbox[0] #add z origin of the intensity image to the max intensity z coordinate within intensity_image
            obj_max_intensity_2Dcoords[row,1] = zyx_max[1][0] + bbox[1]
            obj_max_intensity_2Dcoords[row,2] = zyx_max[2][0] + bbox[2]
            row += 1
    else:
        obj_max_intensity_2Dcoords = np.zeros((obj_max_intensity.size,2))
        row = 0
        for max_intensity, intensity_image, bbox in zip(obj_max_intensity, obj_intensity_images, obj_bbox):
            zyx_max = np.where(intensity_image == max_intensity)
            obj_max_intensity_2Dcoords[row,0] = zyx_max[0][0] + bbox[0] #add y origin of the intensity image to the max intensity z coordinate within intensity_image
            obj_max_intensity_2Dcoords[row,1] = zyx_max[1][0] + bbox[1]
            row += 1
    return obj_max_intensity_2Dcoords


#From the center coordinates and the radius create a 3D surface plot of a sphere
def sphere3Dsurface(ax, radius, center_coords, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center_coords[2]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center_coords[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_coords[0]

    # Plot the surface
    ax.plot_surface(x, y, z, color=[172/255,254/255,43/255])


#Plot label_V as 3D scatter Plot
def plot_3D_sphere(local_max_radii, local_max_2Dcoords, win_title='3D spherical nucleoids surface plot'):
    plt.style.use('dark_background')
    fig = plt.figure(num = win_title)
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().patch.set_facecolor((0.3, 0.3, 0.3, 0.5))
    ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    #RGB_labels = label2rgb(obj_labels)

    for radius,center_coords in zip(local_max_radii, local_max_2Dcoords):
        sphere3Dsurface(ax, radius, center_coords)

    ax.set_xlabel('X Pixels')
    ax.set_ylabel('Y Pixels')
    ax.set_zlabel('Z Pixels')


#Plot 3D array as a 3D scatter splot
def plot_3D_scatter(array3D, win_title='3D scatter mtNetwork', single_color = True, labels = []):
    fig = plt.figure(num = win_title)
    ax = fig.add_subplot(111, projection='3d')
    if single_color:
        zyx_coords = np.where(array3D>0)
        x = zyx_coords[2]
        y = zyx_coords[1]
        z = zyx_coords[0]
        ax.scatter(x, y, z, c='r', marker='s')
    else:
        rgbs = label2rgb(labels)
        for id, rgb in zip(labels, rgbs):
            zyx_coords = np.where(array3D == id)
            x = zyx_coords[2]
            y = zyx_coords[1]
            z = zyx_coords[0]
            ax.scatter(x, y, z, c=[rgb], marker='s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


#Plot a list [[[(z1,y1,x1),(z2,y2,x2)...],...],...] as a 3D line plot
def plot_3D_line(skeleton_Astar, win_title='3D skeleton line mtNetwork', single_color = True, labels = []):
    plt.style.use('dark_background')
    fig = plt.figure(num = win_title)
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().patch.set_facecolor((0.3, 0.3, 0.3, 0.5))
    ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    if single_color:
        for obj_skel in skeleton_Astar:
            for segment in obj_skel:
                zyx_2D = obj_1Dtuplesto2Dcoords(segment)
                x = zyx_2D[:,2]
                y = zyx_2D[:,1]
                z = zyx_2D[:,0]
                ax.plot(x, y, z, c='r', linewidth=4)
    else:
        rgbs = label2rgb(labels) #1D array of RGB values for each label
        for id, rgb, obj_skel in zip(labels, rgbs, skeleton_Astar):
            for segment in obj_skel:
                zyx_2D = obj_1Dtuplesto2Dcoords(segment,True,order='zyx')
                x = zyx_2D[:,2]
                y = zyx_2D[:,1]
                z = zyx_2D[:,0]
                ax.plot(x, y, z, c=rgb, linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic
    dfn = x.size-1 #define degrees of freedom numerator
    dfd = y.size-1 #define degrees of freedom denominator
    p = 1-stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic
    return f, p

def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two

def effect_size(s1, s2, adjusting_sample=None,
                        adjust_s1=False, adjust_s2=False,
                        pop1=None, pop2=None, bootstrap=False):
    if bootstrap:
        # Sample is a 2D array with n_bootstraps rows and sample length columns
        # Output of bootstrap is a 1D array of effect sizes. After we
        # take the 95% to get the upper 95% CI
        if adjust_s1 and adjusting_sample is not None:
            # NOTE: [:,None] is needed for proper brodcasting when multiplying
            # 2D array (sample) with its 1D probability (s_prob)
            s1 *= adjusting_sample[:,None]
            # mu1 = np.quantile(s1, 0.75, axis=1)
            mu1 = np.mean(s1, axis=1)
            # s1_prob = s1/np.sum(s1, axis=1)[:,None]
            # mu1 = np.sum(s1*s1_prob, axis=1)
        else:
            mu1 = np.mean(s1, axis=1)

        if adjust_s2 and adjusting_sample is not None:
            s2 *= adjusting_sample[:,None]
            # mu2 = np.quantile(s2, 0.75, axis=1)
            mu2 = np.mean(s2, axis=1)
            # s2_prob = s2/np.sum(s2, axis=axis)[:,None]
            # mu2 = np.sum(s2*s2_prob, axis=1)
        else:
            mu2 = np.mean(s2, axis=1)

        # NOTE: bootstrapping is performed only on the sample (i.e. pop is 1D)
        std1 = np.std(s1, axis=1) if pop1 is None else np.std(pop1)
        std2 = np.std(s2, axis=1) if pop2 is None else np.std(pop2)

        n1 = s1.shape[1] # Sample length is the number of columns
        n2 = s2.shape[1] # Sample length is the number of columns

        # Cohen
        pooled_std = np.sqrt(((n1-1)*(std1**2)+(n2-1)*(std2**2))
                             /(n1+n2-2))
        cohen_efs = (mu1-mu2)/pooled_std
        # Hedge
        corr_f = 1 - (3/((4*(n1-n2))-9))
        hedge_efs = corr_f * cohen_efs
        # Glass
        glass_efs = (mu1-mu2)/std2
        # Cliffs
        # see https://github.com/neilernst/cliffsDelta/blob/master/cliffsDelta.py
        # and http://www.maths.dur.ac.uk/~dma0je/PG/TrainingWeek10/RSC-2010.pdf
        cliffs_efs = np.zeros_like(glass_efs)
        for i, (_s1, _s2) in enumerate(zip(s1, s2)):
            cliffs_d, _ = cliffsDelta(_s1, _s2)
            cliffs_efs[i] = cliffs_d
        return cohen_efs, hedge_efs, glass_efs, cliffs_efs
    else:
        # s1_prob = s1/np.sum(s1)
        if adjust_s1 and adjusting_sample is not None:
            s1 *= adjusting_sample
            # mu1 = np.quantile(s1, 0.75)
            mu1 = np.mean(s1)
        else:
            mu1 = np.mean(s1)

        # s2_prob = s2/np.sum(s1)
        if adjust_s2 and adjusting_sample is not None:
            s2 *= adjusting_sample
            # mu2 = np.quantile(s2, 0.75)
            mu2 = np.mean(s2)
        else:
            mu2 = np.mean(s2)

        std1 = np.std(s1) if pop1 is None else np.std(pop1)
        std2 = np.std(s2) if pop2 is None else np.std(pop2)

        n1 = len(s1) if pop1 is None else len(pop1)
        n2 = len(s2) if pop2 is None else len(pop2)

        # Cohen
        pooled_std = np.sqrt(((n1-1)*(std1**2)+(n2-1)*(std2**2))
                             /(n1+n2-2))
        cohen_efs = (mu1-mu2)/pooled_std
        # Hedge
        corr_f = 1 - (3/((4*(n1-n2))-9))
        hedge_efs = corr_f * cohen_efs
        # Glass
        glass_efs = (mu1-mu2)/std2
        cliffs_efs = np.zeros_like(glass_efs)
        return cohen_efs, hedge_efs, glass_efs, cliffs_efs


def _spot_detection(V_local_spots, local_maxima_thresh, zyx_resolution_pxl,
                   filter_by_ref_ch, ref_mask, slice_3D, local_obj_mask,
                   slice_bbox_lower, V_spots, global_segm_lab_3D, rp_segm_3D,
                   labels=None):
    z, y, x = np.round(zyx_resolution_pxl*2).astype(int)
    footprint = np.ones((z, y, x))

    backgr_val = _calc_global_backgr_val(V_spots, global_segm_lab_3D, rp_segm_3D)
    V_detect = V_local_spots# - backgr_val


    local_max_coords_ID = peak_local_max(
                             V_local_spots,
                             threshold_abs=local_maxima_thresh,
                             footprint=np.ones((z, y, x)),
                             labels=labels)

    if filter_by_ref_ch:
        # Filter peaks found inside reference channel
        local_ref_mask = ref_mask[slice_3D]
        local_ref_mask_ID = np.logical_and(local_ref_mask,
                                           local_obj_mask)
        local_ref_bool = local_ref_mask_ID[
                                   tuple(local_max_coords_ID.T)]
        local_max_coords_ID = local_max_coords_ID[
                                                 local_ref_bool]
    else:
        # Filter peaks inside segemnted objects (e.g. cell)
        in_cells_spots_mask = local_obj_mask[
                                   tuple(local_max_coords_ID.T)]
        local_max_coords_ID = local_max_coords_ID[
                                            in_cells_spots_mask]

    # Transform coordinates from local to global
    local_max_coords_ID += slice_bbox_lower
    return local_max_coords_ID

def spot_detection_global(rp_segm_3D, V_spots, local_maxima_thresh_func,
                          zyx_resolution_pxl, ref_mask, global_segm_lab_3D,
                          filter_by_ref_ch=False, make_sharper=False,
                          inspect=False, inspect_deep=False, bp=None):
    (V_spots_collage, V_local_heights,
    slice_bboxs_lower, local_obj_masks,
    collage_idxs, slices_3D, backgr_vals) = _make_collage(rp_segm_3D, V_spots,
                                             global_segm_lab_3D)

    if make_sharper:
        V_spots_collage_blurred = gaussian(V_spots_collage,
                                           sigma=zyx_resolution_pxl)
        V_spots_collage_sharp = V_spots_collage - V_spots_collage_blurred
        if inspect:
            img = V_spots_collage.max(axis=0).T
            shown_img = apps._crop_collage_two_lines(img,
                                             collage_idxs=collage_idxs)
            apps.imshow_tk(shown_img.T,
                           additional_imgs=[V_spots_collage_sharp.max(axis=0)],
                           titles=['Original non-filtered image',
                                   'Sharpened image'])
            if bp is not None:
                bp.pausehere()
        V_spots_collage = V_spots_collage_sharp


    # Test all threshold for local maxima threshold
    if inspect:
        matplotlib.use('TkAgg')
        img = V_spots_collage.max(axis=0).T
        (fig, ax,
        is_cropped) = apps.my_try_all_threshold(img,
                                 collage_idxs=img.shape[1])
        suptitle = ('Automatic thresholding to outline areas '
             'where spots will be searched.\n')
        if is_cropped:
            suptitle += (
                f'\n (Image has been cropped for visualization '
                'purposes and it could show less segmented '
                'objects.)')
        fig.suptitle(suptitle)
        plt.show()
        matplotlib.use('Agg')
        if bp is not None:
            bp.pausehere()

    local_maxima_thresh = local_maxima_thresh_func(V_spots_collage.max(axis=0))

    inputs = zip(V_local_heights, local_obj_masks, slice_bboxs_lower, slices_3D)
    d, _, w = V_spots_collage.shape
    y0 = 0
    local_max_coords = []
    if make_sharper:
        V_spots_sharp = np.zeros_like(V_spots)
    else:
        V_spots_sharp = None

    for h, local_obj_mask, slice_bbox_lower, slice_3D in inputs:
        V_local_spots = V_spots_collage[:, y0:y0+h]

        # Sharpen and insert sharper local image into global
        if make_sharper:
            V_local_spots_blurred = gaussian(V_local_spots,
                                             sigma=zyx_resolution_pxl)
            V_local_spots_sharp = V_local_spots - V_local_spots_blurred
            m = local_obj_mask
            V_spots_sharp[slice_3D][m] = V_local_spots_sharp[m]

        local_max_coords_ID = _spot_detection(
                                V_local_spots, local_maxima_thresh,
                                zyx_resolution_pxl, filter_by_ref_ch,
                                ref_mask, slice_3D, local_obj_mask,
                                slice_bbox_lower, V_spots, global_segm_lab_3D,
                                rp_segm_3D
        )

        if inspect_deep:
            apps.imshow_tk(V_local_spots.max(axis=0),
                dots_coords=local_max_coords_ID-slice_bbox_lower,
                additional_imgs=[local_obj_mask.max(axis=0)],
                x_idx=2)
            if bp is not None:
                bp.pausehere()

        local_max_coords.append(local_max_coords_ID)

        y0 += h

    local_max_coords = np.concatenate(local_max_coords, axis=0)
    return local_max_coords, V_spots_sharp

def _calc_global_backgr_val(V_spots, global_segm_lab_3D, rp_segm_3D):
    # Get all non-object intensity values for each sliced (local) object
    backgr_vals = []
    global_segm_mask_3D = global_segm_lab_3D>0
    for obj in rp_segm_3D:
        obj_mask_sliced = global_segm_mask_3D[obj.slice]
        zz, yy, xx = np.nonzero(~obj_mask_sliced)
        backgr_vals.extend(V_spots[zz, yy, xx])
    return np.median(backgr_vals)

def _make_collage(rp_segm_3D, V_spots, global_segm_lab_3D):
    V_local_shapes = {'depth': [], 'height': [], 'width': []}
    bbox_centers = []
    IDs = []
    for obj_rp in rp_segm_3D:
        depth, height, width = obj_rp.image.shape
        min_page, min_row, min_col, max_page, max_row, max_col = obj_rp.bbox
        zc = int(round(min_page+(max_page-min_page)/2))
        yc = int(round(min_row+(max_row-min_row)/2))
        xc = int(round(min_col+(max_col-min_col)/2))
        bbox_centers.append((zc, yc, xc))
        IDs.append(obj_rp.label)
        V_local_shapes['depth'].append(depth)
        V_local_shapes['height'].append(height)
        V_local_shapes['width'].append(width)

    width_collage = max(V_local_shapes['width'])
    height_collage = sum(V_local_shapes['height'])
    depth_collage = max(V_local_shapes['depth'])

    V_collage = np.zeros([depth_collage, height_collage, width_collage])
    slice_bboxs_lower = []
    local_obj_masks = []
    collage_idxs = []
    slices_3D = []
    backgr_vals = []
    w = width_collage
    d = depth_collage
    Z, Y, X = V_spots.shape
    y0 = 0
    for (zc, yc, xc), h, ID in zip(bbox_centers, V_local_shapes['height'], IDs):
        half_w = int(width_collage/2)
        half_h = int(h/2)
        half_d = int(depth_collage/2)
        l = xc-half_w if xc-half_w >= 0 else 0
        b = yc-half_h if yc-half_h >= 0 else 0
        f = zc-half_d if zc-half_d >= 0 else 0
        r = l+w
        if r > X:
            x_roll = r-X
            l -= x_roll
            r = X
        t = b+h
        if t > Y:
            y_roll = t-Y
            b -= y_roll
            t = Y
        k = f+d
        if k > Z:
            z_roll = k-Z
            f -= z_roll
            k = Z
        l = l if l >= 0 else 0
        b = b if b >= 0 else 0
        f = f if f >= 0 else 0


        slice_3D = slice(f,f+d), slice(b,b+h),slice(l,l+w)
        local_obj_mask = global_segm_lab_3D[slice_3D].copy()
        V_spots_local = V_spots[slice_3D]
        # backgr_vals.append(np.median(V_spots_local[~local_obj_mask]))

        local_obj_mask[local_obj_mask != ID] = 0
        local_obj_mask = local_obj_mask.astype(bool)
        try:
            V_collage[:, y0:y0+h] = V_spots_local
        except:
            traceback.print_exc()
            import pdb; pdb.set_trace()
        slice_bboxs_lower.append((f, b, l))
        local_obj_masks.append(local_obj_mask)
        collage_idxs.append(y0+h)
        slices_3D.append(slice_3D)

        y0 += h

    local_heights = V_local_shapes['height']
    return (V_collage, local_heights, slice_bboxs_lower, local_obj_masks,
            collage_idxs, slices_3D, backgr_vals)

def spot_detection_local(V_spots, rp_segm_3D, segm_npy_3D, cca_df,
                         zyx_resolution_pxl, local_maxima_thresh_func, ref_mask,
                         zyx_vox_dim, zyx_resolution,
                         V_local_spots_PC=None,  local_mask_PC_3D=None,
                         filter_by_ref_ch=False,  make_spots_sharper=False,
                         inspect_deep=False, gop_how='effect size',
                         gop_limit=0.2, V_spots_sharp=None, bp=None,
                         experimenting=False):
    local_max_coords = []
    for obj_rp in rp_segm_3D:
        ID = obj_rp.label
        # Skip buds
        if cca_df is not None:
            if cca_df.at[ID, 'Relationship'] == 'bud':
                continue

        (V_local_spots, slice_3D,
        local_obj_mask, V_global_spots,
        global_obj_mask, backgr_vals,
        obj_coords, V_local_plus_PC) = _preprocessing_spots(
                              V_spots, cca_df,
                              segm_npy_3D, ID,
                              filter_by_ref_ch=filter_by_ref_ch,
                              make_sharper=make_spots_sharper,
                              zyx_resolution_pxl=zyx_resolution_pxl,
                              inspect=inspect_deep,
                              include_postitive_control=False,
                              V_local_spots_PC=V_local_spots_PC,
                              local_mask_PC_3D=local_mask_PC_3D,
                              zyx_vox_dim=zyx_vox_dim,
                              zyx_resolution=zyx_resolution,
                              how=gop_how,
                              gop_limit=gop_limit, bp=bp,
                              local_max_thresh_func=local_maxima_thresh_func)

        labels = None
        if experimenting:
            try:
                # Load model
                model = unet.unet(base_n=16,
                                  input_size=(None, None, 1))
                src_path = os.path.dirname(os.path.realpath(__file__))
                main_path = os.path.dirname(src_path)
                weights_path = os.path.join(main_path, 'model', 'unet_spots.hdf5')
                model.load_weights(weights_path)
                # Crop image 256x256 at centre bbox
                obj_rp = regionprops(label(global_obj_mask.max(axis=0)))[0]
                y_min, x_min, y_max, x_max = obj_rp.bbox
                y_c = int(y_min+(y_max-y_min)/2)
                x_c = int(x_min+(x_max-x_min)/2)
                Z, _, _ = V_spots.shape
                slice_3D = (slice(0, Z),
                            slice(y_c-64, y_c+64),
                            slice(x_c-64, x_c+64))
                local_obj_mask = global_obj_mask[slice_3D]
                V_local_spots = V_spots[slice_3D]
                im = V_local_spots.max(axis=0)
                im = im/im.max()
                im = np.abs((im-gaussian(im, sigma=zyx_resolution_pxl[-1])))
                im /= im.max()
                im = gaussian(im, 0.75)
                im /= im.max()
                results = model.predict(im[np.newaxis,:,:,np.newaxis],
                                        batch_size=1)
                res = results[0,:,:,0]
                labels = label(np.array([res>0.9]*Z))
                apps.imshow_tk(im, additional_imgs=[labels])
            except:
                traceback.print_exc()
                (V_local_spots, slice_3D,
                local_obj_mask, V_global_spots,
                global_obj_mask, backgr_vals,
                obj_coords, V_local_plus_PC) = _preprocessing_spots(
                              V_spots, cca_df,
                              segm_npy_3D, ID,
                              filter_by_ref_ch=filter_by_ref_ch,
                              make_sharper=make_spots_sharper,
                              zyx_resolution_pxl=zyx_resolution_pxl,
                              inspect=inspect_deep,
                              include_postitive_control=True,
                              V_local_spots_PC=V_local_spots_PC,
                              local_mask_PC_3D=local_mask_PC_3D,
                              zyx_vox_dim=zyx_vox_dim,
                              zyx_resolution=zyx_resolution,
                              how=gop_how,
                              gop_limit=gop_limit, bp=bp,
                              local_max_thresh_func=local_maxima_thresh_func)

        slice_bbox_lower = np.array([s.start for s in slice_3D])

        # Insert sharper local image into global
        if make_spots_sharper:
            m = global_obj_mask
            V_spots_sharp[m] = V_global_spots[m]

        if V_local_plus_PC is not None:
            V_spots_for_thresh = V_local_plus_PC
        else:
            V_spots_for_thresh = V_local_spots
        try:
            local_maxima_thresh = local_maxima_thresh_func(
                                     V_spots_for_thresh.max(axis=0))
            if experimenting:
                local_maxima_thresh = 0
        except:
            traceback.print_exc()
            local_maxima_thresh = 0
            print('Trying with local maxima threshold = 0.0')

        local_max_coords_ID = _spot_detection(
                            V_local_spots, local_maxima_thresh,
                            zyx_resolution_pxl, filter_by_ref_ch,
                            ref_mask, slice_3D, local_obj_mask,
                            slice_bbox_lower, V_spots, segm_npy_3D,
                            rp_segm_3D, labels=labels
        )
        local_max_coords.append(local_max_coords_ID)
    local_max_coords = np.concatenate(local_max_coords, axis=0)
    return local_max_coords, V_spots_sharp

def dummy_cc_stage_df(all_cells_ids):
    cc_stage = ['G1' for ID in all_cells_ids]
    num_cycles = [-1]*len(all_cells_ids)
    relationship = ['mother' for ID in all_cells_ids]
    related_to = [0]*len(all_cells_ids)
    OF = np.zeros(len(all_cells_ids), bool)
    df = pd.DataFrame({
                        'Cell cycle stage': cc_stage,
                        '# of cycles': num_cycles,
                        'Relative\'s ID': related_to,
                        'Relationship': relationship,
                        'OF': OF},
                        index=all_cells_ids)
    df.index.name = 'Cell_ID'
    return df

def preprocessing_ref(V, cca_df, segm_npy_3D, ID,
                  zyx_vox_dim=None, zyx_resolution=None,
                  noisy_bkgr=True, zyx_resolution_pxl=None,
                  inspect=False, ridge_operator=None, bp=None,
                  local_max_thresh_func=None):

    # Build a mask of moth-bud if the cell is in 'S'
    cc_stage = cca_df.at[ID, 'Cell cycle stage']
    if cc_stage == 'S':
        bud_ID = cca_df.at[ID, 'Relative\'s ID']
        if bud_ID > 0:
            global_segm_mask_3D_ID = np.logical_or(segm_npy_3D==ID,
                                                   segm_npy_3D==bud_ID)
        else:
            print('')
            print('------------------------------------------------------------')
            raise IndexError(f'Cell ID {ID} is in "S" phase but the relative ID is {bud_ID}')
            print('------------------------------------------------------------')
            print('')
    else:
        global_segm_mask_3D_ID = segm_npy_3D==ID
        bud_ID = ID

    # Get obj props in 3D
    obj_rp = regionprops(global_segm_mask_3D_ID.astype(np.uint8))[0]
    local_obj_mask = obj_rp.image
    slice_3D = obj_rp.slice
    obj_cooords = obj_rp.coords

    # Generate local intensity image
    raw_V_local = V[slice_3D]
    global_V = None
    global_obj_mask = None

    backgr_mask = ~local_obj_mask
    backgr_vals = raw_V_local[backgr_mask]
    if backgr_vals.size>0:
        backgr_mean = backgr_vals.mean()
        backgr_std = backgr_vals.std()
    else:
        raise NotImplementedError('Not having a segmentation mask is still WIP.')

    # Generate noisy background (automatic thresholding works better)
    if noisy_bkgr:
        a = np.square(backgr_mean/backgr_std)
        b = np.square(backgr_std)/backgr_mean
        V_local = np.random.gamma(a, b, size=raw_V_local.shape)
    else:
        V_local = np.zeros_like(raw_V_local)



    # Place intensity image masked by segmentation onto noisy background
    V_local[local_obj_mask] = raw_V_local[local_obj_mask]

    # A gaussian filter improves segmentation accuracy
    V_local = gaussian(V_local, sigma=0.75)

    if ridge_operator is not None:
        _, y, x = zyx_resolution_pxl
        filtered_local_V = ridge_operator(raw_V_local, black_ridges=False)
                           #sigmas=np.linspace(0.5, y, num=5))

        if inspect:
            apps.imshow_tk(raw_V_local.max(axis=0),
                           additional_imgs=[filtered_local_V.max(axis=0)],
                           titles=['Original non-filtered image',
                                   'Sharpened image'])
            if bp is not None:
                bp.pausehere()
        raw_V_local = filtered_local_V

    return V_local, slice_3D, local_obj_mask

def _preprocessing_spots(V, cca_df, segm_npy_3D, ID,
                  zyx_vox_dim=None, zyx_resolution=None,
                  noisy_bkgr=True, ref_mask=None,
                  filter_by_ref_ch=False,
                  make_sharper=False, zyx_resolution_pxl=None,
                  inspect=False, include_postitive_control=False,
                  V_local_spots_PC=None, local_mask_PC_3D=None,
                  how='effect size', gop_limit=None, bp=None,
                  local_max_thresh_func=None):

    # Build a mask of moth-bud if the cell is in 'S'
    if cca_df is not None:
        cc_stage = cca_df.at[ID, 'Cell cycle stage']
        if cc_stage == 'S':
            bud_ID = cca_df.at[ID, 'Relative\'s ID']
            global_segm_mask_3D_ID = np.logical_or(segm_npy_3D==ID,
                                                   segm_npy_3D==bud_ID)
        else:
            global_segm_mask_3D_ID = segm_npy_3D==ID
    else:
        global_segm_mask_3D_ID = segm_npy_3D==ID
        bud_ID = ID


    # Get obj props in 3D
    obj_rp = regionprops(global_segm_mask_3D_ID.astype(np.uint8))[0]
    local_obj_mask = obj_rp.image
    slice_3D = obj_rp.slice
    obj_cooords = obj_rp.coords

    # Generate local intensity image
    raw_V_local = V[slice_3D]
    global_V = None
    global_obj_mask = None
    V_positive = None
    V_local_plus_PC_shape = None
    z_L, y_L, x_L = raw_V_local.shape

    # Get background values
    if ref_mask is not None and filter_by_ref_ch:
        local_ref_mask = ref_mask[slice_3D]
        backgr_mask = np.logical_and(local_obj_mask, ~local_ref_mask)
    else:
        backgr_mask = ~local_obj_mask

    if make_sharper:
        global_V = np.zeros_like(V)
        blurred_local_V = gaussian(raw_V_local, sigma=zyx_resolution_pxl)
        filtered_local_V = raw_V_local-blurred_local_V
        if inspect:
            apps.imshow_tk(raw_V_local.max(axis=0),
                           additional_imgs=[filtered_local_V.max(axis=0)],
                           titles=['Original non-filtered image',
                                   'Sharpened image'])
            if bp is not None:
                bp.pausehere()

        global_V[slice_3D] = filtered_local_V
        global_obj_mask = global_segm_mask_3D_ID
        V_local = filtered_local_V
    else:
        V_local = raw_V_local

    backgr_vals = np.abs(V_local[backgr_mask])
    raw_backgr_vals = raw_V_local[backgr_mask]
    if backgr_vals.size>0:
        backgr_mean = backgr_vals.mean()
        backgr_std = backgr_vals.std()
        backgr_var = np.var(backgr_vals)
        raw_backgr_mean = raw_backgr_vals.mean()
        raw_backgr_std = raw_backgr_vals.std()
    else:
        raise NotImplementedError('Not having a segmentation mask is still WIP.')

    if V_local_spots_PC is not None:
        # Load positive control from positive control analysis
        z_PC, y_PC, x_PC = V_local_spots_PC.shape
        V_local_plus_PC_shape = (max([z_L, z_PC]), y_L+y_PC, max([x_L, x_PC]))

    elif include_postitive_control:
        # generate a positive control
        V_local_spots_PC = _generate_positive_control(
                                            backgr_mean, backgr_std,
                                            raw_backgr_mean, raw_backgr_std,
                                            zyx_vox_dim, zyx_resolution,
                                            raw_V_local.shape,
                                            zyx_resolution_pxl,
                                            gop_limit, how)
        z_PC, y_PC, x_PC = V_local_spots_PC.shape
        V_local_plus_PC_shape = (max([z_L, z_PC]), y_L+y_PC, max([x_L, x_PC]))

    # Generate noisy background (automatic thresholding works better)
    if noisy_bkgr:
        a = np.square(backgr_mean/backgr_std)
        b = np.square(backgr_std)/backgr_mean
        synthetic_backgr = gaussian(
                              np.random.gamma(a, b, size=raw_V_local.shape),
                              sigma=1)
        V_local[backgr_mask] = synthetic_backgr[backgr_mask]
    else:
        V_local[backgr_mask] = 0

    if local_mask_PC_3D is not None:
        m = local_mask_PC_3D
        V_local_plus_PC = gaussian(
                              np.random.gamma(a, b, size=V_local_plus_PC_shape),
                              sigma=1)
        V_local_plus_PC[:z_L,:y_L,:x_L] = V_local
        V_local_plus_PC[:z_PC,y_L:y_L+y_PC,:x_PC] = V_local_spots_PC
    elif include_postitive_control:
        V_local_plus_PC = np.concatenate((V_local, V_local_spots_PC), axis=1)
    else:
        V_local_plus_PC = V_local

    # matplotlib.use('TkAgg')
    # img = V_local_plus_PC.max(axis=0)
    # fig, ax, _ = apps.my_try_all_threshold(img)
    # plt.show()
    # matplotlib.use('Agg')

    return (V_local, slice_3D, local_obj_mask, global_V,
            global_obj_mask, backgr_vals, obj_cooords, V_local_plus_PC)

def _generate_positive_control(backgr_mean, backgr_std,
                               raw_backgr_mean, raw_backgr_std,
                               zyx_vox_dim, zyx_resolution, V_shape,
                               zyx_resolution_pxl, gop_limit, how):

    a = np.square(backgr_mean/backgr_std)
    b = np.square(backgr_std)/backgr_mean
    V_positive = np.random.gamma(a, b, size=V_shape)

    Z, Y, X = V_shape

    # Determine peak_to_bkgr_ratio given how and gop limit
    sph = spheroid(V_positive)
    semiax_len = sph.calc_semiax_len(0, zyx_vox_dim, zyx_resolution)
    local_spot_mask = sph.get_local_spot_mask(semiax_len)
    local_spot_gauss = np.zeros(local_spot_mask.shape)

    z, y, x = np.nonzero(local_spot_mask)
    z0, y0, x0 = [(c-1)/2 for c in local_spot_mask.shape]
    sx, sz = semiax_len
    sy = sx

    gauss_x = np.exp(-((x-x0)**2)/(2*(sx**2)))
    gauss_y = np.exp(-((y-y0)**2)/(2*(sy**2)))
    gauss_z = np.exp(-((z-z0)**2)/(2*(sz**2)))

    if how == 'effect size':
        # Determine the peak_to_bkgr_ratio on unfiltered image and apply
        # to filtered image
        spot_mean = gop_limit*raw_backgr_std + raw_backgr_mean
        _gauss3D = gauss_x*gauss_y*gauss_z
        _gauss3D /= _gauss3D.mean()
        _gauss3D *= spot_mean
        peak_to_bkgr_ratio = _gauss3D.max()/raw_backgr_mean
        spot_A = peak_to_bkgr_ratio*backgr_mean
    elif how == 'peak_to_background ratio':
        spot_A = peak_to_bkgr_ratio*backgr_mean

    _gauss3D = gauss_x*gauss_y*gauss_z*spot_A
    local_spot_gauss[z, y, x] = _gauss3D

    _, yr, xr = zyx_resolution_pxl

    num_spots = int((Y*X)/(yr*xr*4)*0.3)

    sph_size_steps = np.random.randint(0, 2, size=num_spots)

    zc_li = np.random.randint(0, Z, size=num_spots)
    yc_li = np.random.randint(0, Y, size=num_spots)
    xc_li = np.random.randint(0, X, size=num_spots)

    inputs = zip(sph_size_steps, zc_li, yc_li, xc_li)

    for i, z0, y0, x0 in inputs:
        zyx_c = [z0, y0, x0]
        (V_positive, _,
        slice_G_to_L, slice_crop) = sph.index_local_into_global_mask(
                                     V_positive, local_spot_gauss,
                                     zyx_c, semiax_len, Z, Y, X, do_sum=True,
                                     return_slice=True
        )
    V_positive = gaussian(V_positive, sigma=1)
    return V_positive



class metrics_spots:
    def set_attributes(self, spot_mask_template, semiax_len, segm_npy_3D,
            df_spots_ch_norm, df_ref_ch_norm, V_spots, V_ref_ch, ref_ch_mask,
            sph, orig_data, filter_by_ref_ch, spots_mask, calc_effsize,
            do_bootstrap, V_spots_sharp, is_segm_3D):
        self.spot_mask_template = spot_mask_template
        self.spot_edt_template = ndi.distance_transform_edt(spot_mask_template)
        self.spot_edt_template /= self.spot_edt_template.max()
        self.semiax_len = semiax_len
        self.segm_npy_3D = segm_npy_3D
        self.is_segm_3D = is_segm_3D
        self.df_spots_ch_norm = df_spots_ch_norm
        self.df_ref_ch_norm = df_ref_ch_norm
        self.V_spots = V_spots
        self.V_spots_sharp = V_spots_sharp
        self.V_ref_ch = V_ref_ch
        self.spots_mask = spots_mask
        self.ref_ch_mask = ref_ch_mask
        self.sph = sph
        self.orig_data = orig_data
        self.filter_by_ref_ch = filter_by_ref_ch
        self.calc_effsize = calc_effsize
        self.do_bootstrap = do_bootstrap

    def _calc_metrics_spot(self, peak_coords):
        spot_mask_template = self.spot_mask_template
        spot_edt_template = self.spot_edt_template
        semiax_len = self.semiax_len
        segm_npy_3D = self.segm_npy_3D
        is_segm_3D = self.is_segm_3D
        df_spots_ch_norm = self.df_spots_ch_norm
        df_ref_ch_norm = self.df_ref_ch_norm
        V_spots = self.V_spots
        V_spots_sharp = self.V_spots_sharp
        V_ref_ch = self.V_ref_ch
        ref_ch_mask = self.ref_ch_mask
        sph = self.sph
        orig_data = self.orig_data
        filter_by_ref_ch = self.filter_by_ref_ch
        spots_mask = self.spots_mask
        calc_effsize = self.calc_effsize
        do_bootstrap = self.do_bootstrap

        # Initialize variables that will assume some value only in some cases
        is_peak_inside_ref_ch_mask = True

        cohen_efs_s, hedge_efs_s, glass_efs_s, cliffs_efs_s = 0, 0, 0, 0
        cohen_efs_pop, hedge_efs_pop, glass_efs_pop = 0, 0, 0

        (cohen_efs_s_bs_95p, hedge_efs_s_95p,
        glass_efs_s_95p, cliffs_efs_s_95p) = 0, 0, 0, 0
        (cohen_efs_pop_bs_95p, hedge_efs_pop_bs_95p,
        glass_efs_pop_bs_95p) = 0, 0, 0

        (spots_ch_ref_ch_tvalue, spots_ch_ref_ch_pvalue,
        spots_ch_norm_mean_min_ith) = 0, 0, 0

        backgr_INcell_OUTspots_mean, backgr_INcell_OUTspots_median = 0, 0
        backgr_INcell_OUTspots_25p, backgr_INcell_OUTspots_75p = 0, 0

        backgr_INcell_OUTspots_std = 0

        peak_to_backgr_ratio = 0

        spots_ch_ref_ch_tvalue, spots_ch_ref_ch_pvalue = 0, 0

        # Actual parallel iteration code
        spot_s_mask, spot_s_edt = sph.get_global_spot_mask(
                                    spot_mask_template, peak_coords, semiax_len,
                                    additional_local_arr=spot_edt_template
                                    )
        ID, drop = orig_data.nearest_nonzero(segm_npy_3D[peak_coords[0]],
                                             peak_coords[1],
                                             peak_coords[2])
        if df_spots_ch_norm is not None:
            spots_ch_norm_value = df_spots_ch_norm.at[ID, 'spots_ch norm.']
            V_spots_norm = V_spots/spots_ch_norm_value
            spots_ch_norm_sample = V_spots_norm[spot_s_mask]
            spots_ch_norm_mean_min_ith = np.mean(spots_ch_norm_sample)
        ref_ch_norm_value = df_ref_ch_norm.at[ID, 'ref_ch norm.']
        ref_ch_norm_sample = V_ref_ch[spot_s_mask]/ref_ch_norm_value

        ref_ch_abs_sample = V_ref_ch[spot_s_mask]
        spots_ch_abs_sample = V_spots[spot_s_mask]

        spots_ch_abs_voxel = V_spots[tuple(peak_coords)]
        ref_ch_abs_voxel = V_ref_ch[tuple(peak_coords)]

        backgr_how='segm_3D' if is_segm_3D else 'z_slice'
        (IN_cell_backgr_popul,
        IN_cell_backgr_popul_sharp) = orig_data.get_INcell_backgr_vals(
                                V_spots, segm_npy_3D, spots_mask,
                                ID, how=backgr_how, z_slice=peak_coords[0],
                                additional_V=V_spots_sharp,
                                filter_by_ref_ch=filter_by_ref_ch,
                                ref_mask=ref_ch_mask
        )
        is_peak_inside_ref_ch_mask = ref_ch_mask[tuple(peak_coords)]
        backgr_INcell_OUTspots_mean = IN_cell_backgr_popul.mean()
        backgr_INcell_OUTspots_median = np.median(IN_cell_backgr_popul)
        backgr_INcell_OUTspots_25p = np.quantile(
                                            IN_cell_backgr_popul, q=0.25)
        backgr_INcell_OUTspots_75p = np.quantile(
                                            IN_cell_backgr_popul, q=0.75)
        backgr_INcell_OUTspots_std = np.std(IN_cell_backgr_popul)

        if backgr_INcell_OUTspots_median > 0:
            peak_to_backgr_ratio = (spots_ch_abs_voxel
                                    /backgr_INcell_OUTspots_median)
        else:
            peak_to_backgr_ratio = np.inf

        if filter_by_ref_ch:
            # 3D normalised spot channel intensities multiplied by edt vs
            # 3D normalised reference channel intensities
            edt_sample = None
            test_sample = V_spots_norm[spot_s_mask]
            ref_sample = ref_ch_norm_sample
            # # BUG: Check if edt_sample adjusting is needed and check std
            # mean test, mean ref and std because effectsize were negative??
        else:
            # If we don't use the reference channel for filtering good spots
            # we use the background of the spots channel and the xy intensities
            # at z_slice for the test sample (s1)
            spot_s_mask_slice = spot_s_mask[peak_coords[0]]
            if V_spots_sharp is None:
                V = V_spots
                ref_sample = IN_cell_backgr_popul
            else:
                V = V_spots_sharp
                ref_sample = IN_cell_backgr_popul_sharp

            V_spots_slice =  V[peak_coords[0]]
            spot_edt_slice = spot_s_edt[peak_coords[0]]
            edt_sample = spot_edt_slice[spot_s_mask_slice]
            test_sample = V_spots_slice[spot_s_mask_slice]

        if df_spots_ch_norm is not None:
            # Perform Welch's t-test
            spots_ch_ref_ch_tvalue, spots_ch_ref_ch_pvalue = stats.ttest_ind(
                                                         test_sample,
                                                         ref_sample,
                                                         equal_var=False)
        if calc_effsize:
            if do_bootstrap:
                # Determine effect size 95% CI by bootstrapping
                sample_len = len(spots_ch_norm_sample)
                bs_s1 = np.random.choice(test_sample,
                                      size=(2000,sample_len))
                bs_s2 = np.random.choice(ref_sample,
                                      size=(2000,sample_len))
            # Effect size with std estimated from the sample (spot)
            (cohen_efs_s, hedge_efs_s,
            glass_efs_s, cliffs_efs_s) = effect_size(
                                            test_sample,
                                            ref_sample,
                                            adjusting_sample=edt_sample,
                                            adjust_s1=True,
                                            adjust_s2=False
            )
            # if np.all(peak_coords == np.array([20,340,355])) and self.debug:
            #     print('')
            #     print(f'Cell ID {ID}')
            #     print(f'spot max: {test_sample.max()}')
            #     print(f'adj q99: {np.quantile(test_sample*edt_sample, 0.99)}')
            #     print(f'spot mean: {test_sample.mean()}')
            #     print(f'ref sample std: {np.std(ref_sample)}')
            #     print(f'ref sample mean: {np.mean(ref_sample)}')
            if do_bootstrap:
                (cohen_efs_s_bs, hedge_efs_s_bs,
                glass_efs_s_bs, cliffs_efs_bs_s) = effect_size(
                                            bs_s1, bs_s2,
                                            adjusting_sample=edt_sample,
                                            adjust_s1=True,
                                            adjust_s2=False,
                                            bootstrap=True
                )
                cohen_efs_s_bs_95p = np.quantile(cohen_efs_s_bs, q=0.95)
                hedge_efs_s_95p = np.quantile(hedge_efs_s_bs, q=0.95)
                glass_efs_s_95p = np.quantile(glass_efs_s_bs, q=0.95)
                cliffs_efs_s_95p = np.quantile(cliffs_efs_bs_s, q=0.95)
            if ref_ch_mask is not None and filter_by_ref_ch:
                ref_ch_mask_ID = np.logical_and(segm_npy_3D==ID, ref_ch_mask)
                pop2 = V_ref_ch[ref_ch_mask_ID]
                # Effect size with std estimated from the entire control population
                (cohen_efs_pop, hedge_efs_pop,
                glass_efs_pop, _) = effect_size(
                                                test_sample, ref_sample,
                                                adjusting_sample=edt_sample,
                                                adjust_s1=True,
                                                adjust_s2=False,
                                                pop2=pop2)
                if do_bootstrap:
                    # Determine effect size 95% CI by bootstrapping
                    # Effect size with std estimated from the entire population
                    (cohen_efs_pop_bs, hedge_efs_pop_bs,
                    glass_efs_pop_bs, _) = effect_size(
                                                bs_s1, bs_s2,
                                                adjusting_sample=edt_sample,
                                                adjust_s1=True,
                                                adjust_s2=False,
                                                bootstrap=True,
                                                pop2=pop2)
                    cohen_efs_pop_bs_95p = np.quantile(cohen_efs_pop_bs, q=0.95)
                    hedge_efs_pop_bs_95p = np.quantile(hedge_efs_pop_bs, q=0.95)
                    glass_efs_pop_bs_95p = np.quantile(glass_efs_pop_bs, q=0.95)

            spots_ch_abs_mean = np.mean(spots_ch_abs_sample)
            ref_ch_abs_mean =  np.mean(ref_ch_abs_sample)
            ref_ch_norm_mean = np.mean(ref_ch_norm_sample)
        return (spots_ch_abs_voxel, ref_ch_abs_voxel, spots_ch_abs_mean,
                ref_ch_abs_mean, ref_ch_norm_mean, spots_ch_ref_ch_tvalue,
                spots_ch_ref_ch_pvalue, spots_ch_norm_mean_min_ith,
                backgr_INcell_OUTspots_mean, backgr_INcell_OUTspots_median,
                backgr_INcell_OUTspots_25p, backgr_INcell_OUTspots_75p,
                cohen_efs_s, hedge_efs_s, glass_efs_s, cliffs_efs_s,
                cohen_efs_s_bs_95p, hedge_efs_s_95p,
                glass_efs_s_95p, cliffs_efs_s_95p,
                cohen_efs_pop, hedge_efs_pop, glass_efs_pop,
                cohen_efs_pop_bs_95p, hedge_efs_pop_bs_95p, glass_efs_pop_bs_95p,
                is_peak_inside_ref_ch_mask, ID, drop, backgr_INcell_OUTspots_std,
                peak_to_backgr_ratio
        )

    def calc_metrics_spots(self, V_spots, V_ref_ch,
                local_max_coords, df_ref_ch_norm, zyx_resolution,
                zyx_vox_dim, segm_npy_3D, is_segm_3D=False,
                df_spots_ch_norm=None, orig_data=None, ref_ch_mask=None,
                calc_effsize=True, filter_by_ref_ch=True,
                do_bootstrap=True, V_spots_sharp=None):
        # Initialize spheroid class
        sph = spheroid(V_spots)
        semiax_len = sph.calc_semiax_len(0, zyx_vox_dim, zyx_resolution)
        spot_mask_template = sph.get_local_spot_mask(semiax_len)

        # Get a 3D mask of all the spots if it was never computed before.
        # Not needed if we use the reference channel for filtering peaks
        spots_mask = sph.get_spots_mask(0, zyx_vox_dim, zyx_resolution,
                                            local_max_coords)

        # Set attributes that will be used by ThreadPoolExecutor map
        self.set_attributes(spot_mask_template, semiax_len, segm_npy_3D,
                            df_spots_ch_norm, df_ref_ch_norm, V_spots,
                            V_ref_ch, ref_ch_mask, sph, orig_data,
                            filter_by_ref_ch, spots_mask, calc_effsize,
                            do_bootstrap, V_spots_sharp, is_segm_3D)

        num_peaks = local_max_coords.shape[0]
        """Initialize output arrays"""
        # ch0, ch1 absolute intensities of each peak at the single voxel peak
        spots_ch_abs_peaks_voxel = np.zeros(num_peaks)
        ref_ch_abs_peaks_voxel = np.zeros(num_peaks)

        # ch0, ch1 absolute mean intensities within the min sphere
        spots_ch_abs_means_min = np.zeros(num_peaks)
        ref_ch_abs_means_min = np.zeros(num_peaks)

        # ch0, ch1 normalized mean intensities within the min sphere
        spots_ch_norm_means_min = np.zeros(num_peaks)
        ref_ch_norm_means_min = np.zeros(num_peaks)

        # p- and t- values between ch0 and ch1 normalized intensities within min vol
        spots_ch_ref_ch_pvalues = np.zeros(num_peaks)
        spots_ch_ref_ch_tvalues = np.zeros(num_peaks)

        # Background mean, median, 25% and 75% inside the cell and outside the spots
        backgr_INcell_OUTspots_means = np.zeros(num_peaks)
        backgr_INcell_OUTspots_medians = np.zeros(num_peaks)
        backgr_INcell_OUTspots_25ps = np.zeros(num_peaks)
        backgr_INcell_OUTspots_75ps = np.zeros(num_peaks)
        backgr_INcell_OUTspots_stds = np.zeros(num_peaks)

        # Cells labels ID
        Cells_IDs = np.zeros(num_peaks, int)

        # Effect size
        effsize_cohen_s_vals = np.zeros(num_peaks)
        effsize_hedge_s_vals = np.zeros(num_peaks)
        effsize_glass_s_vals = np.zeros(num_peaks)
        effsize_cliffs_s_vals = np.zeros(num_peaks)
        effsize_cohen_pop_vals = np.zeros(num_peaks)
        effsize_hedge_pop_vals = np.zeros(num_peaks)
        effsize_glass_pop_vals = np.zeros(num_peaks)

        # 95% CI effectsize
        effsize_cohen_s_95p = np.zeros(num_peaks)
        effsize_hedge_s_95p = np.zeros(num_peaks)
        effsize_glass_s_95p = np.zeros(num_peaks)
        effsize_cliffs_s_95p = np.zeros(num_peaks)
        effsize_cohen_pop_95p = np.zeros(num_peaks)
        effsize_hedge_pop_95p = np.zeros(num_peaks)
        effsize_glass_pop_95p = np.zeros(num_peaks)

        # Peak to background ratios
        peak_to_backgr_ratios = np.zeros(num_peaks)

        # Peaks inside/outside reference channel list of booleans
        are_peaks_inside_ref_ch_mask = [1]*num_peaks

        # List of valid peaks that lies on Cell ID = 0
        keep_bool = np.ones(num_peaks, bool)

        # with ThreadPoolExecutor(8) as ex:
        #     chunksize = int(np.ceil(len(local_max_coords)/16))
        #     result = list(tqdm(ex.map(self._calc_metrics_spot,
        #                               local_max_coords,
        #                               chunksize=chunksize),
        #                        desc='Computing spots metrics',
        #                        unit=' spot', total=len(local_max_coords),
        #                        leave=False, position=1, ncols=100))
        #
        # # Iterate results and index the metrics into each array
        # for s, metrics in enumerate(result):
        #     (spots_ch_abs_voxel, ref_ch_abs_voxel, spots_ch_abs_mean,
        #     ref_ch_abs_mean, ref_ch_norm_mean, spots_ch_ref_ch_tvalue,
        #     spots_ch_ref_ch_pvalue, spots_ch_norm_mean_min_ith,
        #     backgr_INcell_OUTspots_mean, backgr_INcell_OUTspots_median,
        #     backgr_INcell_OUTspots_25p, backgr_INcell_OUTspots_75p,
        #     cohen_efs_s, hedge_efs_s, glass_efs_s, cliffs_efs_s,
        #     cohen_efs_s_bs_95p, hedge_efs_s_95p,
        #     glass_efs_s_95p, cliffs_efs_s_95p,
        #     cohen_efs_pop, hedge_efs_pop, glass_efs_pop,
        #     cohen_efs_pop_bs_95p, hedge_efs_pop_bs_95p, glass_efs_pop_bs_95p,
        #     is_peak_inside_ref_ch_mask, ID, drop, backgr_INcell_OUTspots_std,
        #     peak_to_backgr_ratio) = metrics

        for s, peak_coords in enumerate(
                                  tqdm(local_max_coords,
                                       desc='Computing spots metrics',
                                       unit=' spot', total=len(local_max_coords),
                                       leave=False, position=1, ncols=100)):
            (spots_ch_abs_voxel, ref_ch_abs_voxel, spots_ch_abs_mean,
            ref_ch_abs_mean, ref_ch_norm_mean, spots_ch_ref_ch_tvalue,
            spots_ch_ref_ch_pvalue, spots_ch_norm_mean_min_ith,
            backgr_INcell_OUTspots_mean, backgr_INcell_OUTspots_median,
            backgr_INcell_OUTspots_25p, backgr_INcell_OUTspots_75p,
            cohen_efs_s, hedge_efs_s, glass_efs_s, cliffs_efs_s,
            cohen_efs_s_bs_95p, hedge_efs_s_95p,
            glass_efs_s_95p, cliffs_efs_s_95p,
            cohen_efs_pop, hedge_efs_pop, glass_efs_pop,
            cohen_efs_pop_bs_95p, hedge_efs_pop_bs_95p, glass_efs_pop_bs_95p,
            is_peak_inside_ref_ch_mask, ID, drop, backgr_INcell_OUTspots_std,
            peak_to_backgr_ratio) = self._calc_metrics_spot(peak_coords)

            peak_to_backgr_ratios[s] = peak_to_backgr_ratio
            are_peaks_inside_ref_ch_mask[s] = int(is_peak_inside_ref_ch_mask)
            effsize_cohen_s_vals[s] = cohen_efs_s
            effsize_hedge_s_vals[s] = hedge_efs_s
            effsize_glass_s_vals[s] = glass_efs_s
            effsize_cliffs_s_vals[s] = cliffs_efs_s
            effsize_hedge_s_95p[s] = cohen_efs_s_bs_95p
            effsize_glass_s_95p[s] = hedge_efs_s_95p
            effsize_cohen_s_95p[s] = glass_efs_s_95p
            effsize_cliffs_s_95p[s] = cliffs_efs_s_95p
            effsize_cohen_pop_vals[s] = cohen_efs_pop
            effsize_hedge_pop_vals[s] = hedge_efs_pop
            effsize_glass_pop_vals[s] = glass_efs_pop
            effsize_cohen_pop_95p[s] = cohen_efs_pop_bs_95p
            effsize_hedge_pop_95p[s] = hedge_efs_pop_bs_95p
            effsize_glass_pop_95p[s] = glass_efs_pop_bs_95p
            spots_ch_abs_peaks_voxel[s] = spots_ch_abs_voxel
            ref_ch_abs_peaks_voxel[s] = ref_ch_abs_voxel
            spots_ch_abs_means_min[s] = spots_ch_abs_mean
            ref_ch_abs_means_min[s] =  ref_ch_abs_mean
            ref_ch_norm_means_min[s] = ref_ch_norm_mean
            spots_ch_ref_ch_tvalues[s] = spots_ch_ref_ch_tvalue
            spots_ch_ref_ch_pvalues[s] = spots_ch_ref_ch_pvalue
            spots_ch_norm_means_min[s] = spots_ch_norm_mean_min_ith
            backgr_INcell_OUTspots_means[s] = backgr_INcell_OUTspots_mean
            backgr_INcell_OUTspots_medians[s] = backgr_INcell_OUTspots_median
            backgr_INcell_OUTspots_25ps[s] = backgr_INcell_OUTspots_25p
            backgr_INcell_OUTspots_75ps[s] = backgr_INcell_OUTspots_75p
            backgr_INcell_OUTspots_stds[s] = backgr_INcell_OUTspots_std
            Cells_IDs[s] = ID
            keep_bool[s] = not drop
        # Construct dataframe
        unknown = np.zeros(num_peaks)
        df = (pd.DataFrame({
                 'Cell_ID': Cells_IDs,
                 'vox_spot': spots_ch_abs_peaks_voxel,
                 'vox_ref': ref_ch_abs_peaks_voxel,
                 '|abs|_spot': spots_ch_abs_means_min,
                 '|abs|_ref': ref_ch_abs_means_min,
                 '|norm|_spot': spots_ch_norm_means_min,
                 '|norm|_ref': ref_ch_norm_means_min,
                 '|spot|:|ref| t-value': spots_ch_ref_ch_tvalues,
                 '|spot|:|ref| p-value (t)': spots_ch_ref_ch_pvalues,
                 'z': local_max_coords[:,0],
                 'y': local_max_coords[:,1],
                 'x': local_max_coords[:,2],
                 'peak_to_background ratio': peak_to_backgr_ratios,
                 'effsize_cohen_s': effsize_cohen_s_vals,
                 'effsize_hedge_s': effsize_hedge_s_vals,
                 'effsize_glass_s': effsize_glass_s_vals,
                 'effsize_cliffs_s': effsize_cliffs_s_vals,
                 'effsize_cohen_pop': effsize_cohen_pop_vals,
                 'effsize_hedge_pop': effsize_hedge_pop_vals,
                 'effsize_glass_pop': effsize_glass_pop_vals,
                 'effsize_cohen_s_95p': effsize_cohen_s_95p,
                 'effsize_hedge_s_95p': effsize_glass_s_95p,
                 'effsize_glass_s_95p': effsize_glass_s_95p,
                 'effsize_cliffs_s_95p': effsize_cliffs_s_95p,
                 'effsize_cohen_pop_95p': effsize_cohen_pop_95p,
                 'effsize_hedge_pop_95p': effsize_hedge_pop_95p,
                 'effsize_glass_pop_95p': effsize_glass_pop_95p,
                 'backgr_INcell_OUTspot_mean': backgr_INcell_OUTspots_means,
                 'backgr_INcell_OUTspot_median': backgr_INcell_OUTspots_medians,
                 'backgr_INcell_OUTspot_75p': backgr_INcell_OUTspots_75ps,
                 'backgr_INcell_OUTspot_25p': backgr_INcell_OUTspots_25ps,
                 'backgr_INcell_OUTspot_std': backgr_INcell_OUTspots_stds,
                 'is_spot_inside_ref_ch': are_peaks_inside_ref_ch_mask}
                 )
                 .sort_values(by='vox_spot',
                              ascending=False)
                 [keep_bool]
            )
        local_max_coords = df[['z', 'y', 'x']].to_numpy()
        return df, local_max_coords, spots_mask



def filter_good_peaks(df, gop_thresh_val, how='t-test',
                      which_effsize='effsize_glass_s'):
    if how == 't-test':
        p_limit = gop_thresh_val[0]
        df = (df[
                 (df['|spot|:|ref| t-value'] > 0) &
                 (df['|spot|:|ref| p-value (t)'] < p_limit)
                 ]
        )
    elif how == 'effect size':
        effsize_limit = gop_thresh_val[0]
        df = df[df[which_effsize] > effsize_limit]
    elif how == 'effect size bootstrapping':
        effsize_limit = gop_thresh_val[0]
        effsize_colname = f'{which_effsize}_95p'
        df = df[df[effsize_colname] > effsize_limit]
    elif how == 'peak_to_background ratio':
        peak_to_bkgr = gop_thresh_val[0]
        df = df[df['peak_to_background ratio'] > peak_to_bkgr]
    return df


def p_test(df, col_p, col_t, p_limit, verb=False):
    """DEPRECATED"""
    indexes_not_valid_p = df[(df[col_p] > p_limit)|(df[col_t] < 0)].index
    df_p_test = df.drop(indexes_not_valid_p)
    df_p_test.reset_index(inplace = True, drop=True)
    if verb:
        print('Data after p-test:')
        print(df_p_test)
    local_max_coords_p_test = np.stack((df_p_test['z'].to_numpy(),
                                        df_p_test['y'].to_numpy(),
                                        df_p_test['x'].to_numpy()),
                                        axis=1)
    return df_p_test, local_max_coords_p_test


# Class that determines valid points. A valid point is defined as not lying
# inside of any ellipsoid centred at points with higher intensity.
# The ellipsoid size is determined by yx and z resolution limits.
class filter_points_resol_limit:
    def __init__(self, zyx_coords, ellipse_radii_pxl, V_shape,
                       filter_z_bound=False, return_valid_points=True):
        self.V_shape = V_shape
        self.radii = ellipse_radii_pxl
        self.zyx_coords = zyx_coords
        if filter_z_bound:
            self.zyx_coords = self.points_boundaries(zyx_coords)
        if return_valid_points:
            self.valid_points = self.return_valid_points(self.zyx_coords,
                                                         self.radii)
        else:
            self.valid_points = self.zyx_coords

    def points_boundaries(self, zyx_coords):
        z_dist_min = self.radii[0]
        z_upper_boundary = self.V_shape[0]-1
        # Remove points that are at least z_resol_limit_pxl away from the
        # V boundaries (V_spots shape)
        z_dist_upper = z_upper_boundary - zyx_coords[:,0]
        z_dist_upper_mask = z_dist_upper>z_dist_min
        zyx_coords = zyx_coords[z_dist_upper_mask]
        z_dist_lower_mask = zyx_coords[:,0] > z_dist_min
        zyx_coords = zyx_coords[z_dist_lower_mask]
        return zyx_coords

    # Determine if a 2D array of points coordinates lies inside a given ellipsoid
    # https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse
    def points_outside_ellips(self, radii, centre, points, return_inner=False):
        RxRyRz_sq = np.prod(radii)**2 # square of the radii product
        P_C_diff = points - centre
        RR = np.zeros_like(radii) # [ry*rz, rx*rz, rx*ry]
        for i in range(len(radii)):
            rr = np.copy(radii)
            rr[i] = 1
            RR[i] = np.prod(rr)
        outer_region = np.sum((P_C_diff * RR)**2, axis=1) > RxRyRz_sq
        outer_points = points[outer_region]
        if return_inner:
            inner_region = ~outer_region
            inner_points = points[inner_region]
            return outer_points, inner_points
        else:
            return outer_points

    # Iterate through points and discard any point that lies outside the
    # ellipsoid centred at the ith point
    def return_valid_points(self, points, radii, i=0, inner_points=np.array([])):
        points_to_check = np.copy(points)
        valid_points = np.zeros_like(points)
        not_valid_points = np.zeros_like(points)
        while points_to_check.size != 0:
            centre = points_to_check[0]
            valid_points[i] = centre
            if inner_points.size != 0:
                not_valid_points[n] = inner_points[0]
                n += 1
            points_to_check = np.delete(points_to_check, 0, axis=0)
            dist = np.linalg.norm((centre - points_to_check), axis=1)
            points_to_check = self.points_outside_ellips(radii, centre,
                                                         points_to_check)
            i += 1
        return valid_points[:i]

    def get_valid_points_idx(self, points):
        valid_points = self.valid_points
        valid_points_i = []
        for point in valid_points:
            i = np.where((points == point).all(axis=1))[0][0]
            valid_points_i.append(i)
        return valid_points_i

class df_MultiIndex_IDs:
    def __init__(self, original_df, verb=False):
        sort_df = (original_df.reset_index()
                              .sort_values(['Cell_ID', 'vox_spot'],
                                            ascending=[True, False]))
        if not sort_df.empty:
            self.IDs = sort_df['Cell_ID'].to_numpy()
            self.spot_ids = self.get_spot_ids(self.IDs)
            self.IDs_unique = np.unique(self.IDs)
            sort_df['spot_id'] = self.spot_ids
            df_IDs = sort_df.set_index(['Cell_ID','spot_id'])
            self.df_IDs = df_IDs
            self.len_sub_df = self.get_len_sub_df(df_IDs, self.IDs_unique)
            num_spots = []
            for ID_len in self.len_sub_df:
                num_spots.append(ID_len['Num'])
            self.num_spots = np.asarray(num_spots)
            if verb:
                self.print_len(self.len_sub_df)
        else:
            self.num_spots = 0
            self.IDs_unique = []
            # df_IDs = sort_df.set_index(['Cell_ID','spot_id'])
            self.df_IDs = sort_df

    def get_spot_ids(self, IDs):
        nucl_ID_str = [0]*len(IDs)
        ID_prev = IDs[0]
        nucl_ID = 0
        for i, ID in enumerate(IDs):
            if ID_prev == ID:
                nucl_ID += 1
            else:
                nucl_ID = 1
            nucl_ID_str[i] = nucl_ID
            ID_prev = ID
        return nucl_ID_str

    def get_ID_data(self, df_IDs, ID):
        return df_IDs.loc[ID]

    def len_ID_data(self, df_IDs, ID):
        return len(self.get_ID_data(df_IDs, ID).index)

    def get_len_sub_df(self, df_IDs, IDs_unique):
        dict_ID_len = [{'ID': 0, 'Num': 0} for i in range(len(IDs_unique))]
        for i, ID in enumerate(IDs_unique):
            dict_ID_len[i]['ID'] = ID
            dict_ID_len[i]['Num'] = self.len_ID_data(df_IDs, ID)
        return dict_ID_len

    def print_len(self, dict_ID_len):
        for ID_len in dict_ID_len:
            print('Cell ID {} has {} nucleoids'
                  .format(ID_len['ID'], ID_len['Num']))

def expand_labels(label_image, distance=1, zyx_vox_size=None):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    if zyx_vox_size is None:
        zyx_vox_size = [1]*label_image.ndim

    distances, nearest_label_coords = ndi.distance_transform_edt(
        label_image == 0, return_indices=True, sampling=zyx_vox_size,
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

class spheroid:
    def __init__(self, V_ch):
        self.V_ch = V_ch
        self.V_shape = V_ch.shape
        Z, Y, X = self.V_shape
        self.bp = apps.tk_breakpoint()

    def calc_semiax_len(self, i, zyx_vox_dim, zyx_resolution):
        zvd, yvd, xvd = zyx_vox_dim
        zr, yr, xr = zyx_resolution
        xys = yr + (yvd*i)  # a radius in real units
        zs = zr + (yvd*i)  # c radius in real units
        self.xys = xys
        self.zs = zs
        a = xys/yvd  # a radius in pixels (xy direction)
        c = zs/zvd  # c radius in pixels (z direction)
        return a, c

    def get_backgr_vals(self, zyx_c, semiax_len, V, spot_id):
        spot_surf_mask, spot_filled_mask = self.get_sph_surf_mask(
                                                    semiax_len,
                                                    zyx_c,
                                                    self.V_shape,
                                                    return_filled_mask=True)
        surf_pixels = V[spot_surf_mask]
        surf_mean = np.mean(surf_pixels)
        return surf_mean, spot_filled_mask

    def get_sph_surf_mask(self, semiax_len, zyx_center, V_shape,
                          return_filled_mask=False):
        """
        Generate a spheroid surface mask array that can be used to index a 3D array.
        ogrid is given by
        Z, Y, X = V.shape
        z, y, x = np.ogrid[0:Z, 0:Y, 0:X]

        The spheroid is generated by logical_xor between two spheroids that have
        1 pixel difference between their axis lengths
        """
        a, c = semiax_len
        # Outer full mask
        s_outer = self.get_local_spot_mask(semiax_len)
        a_inner = a-1
        # Make sure than c_inner is never exactly 0
        c_inner = c-1 if c-1 != 0 else c-1+1E-15
        # Inner full mask with same shape as outer mask
        s_inner = self.get_local_spot_mask((a_inner, c_inner),
                                            ogrid_bounds=semiax_len)
        # Surface mask (difference between outer and inner)
        spot_surf_mask = np.logical_xor(s_outer, s_inner)
        # Insert local mask into global
        spot_mask = self.get_global_spot_mask(spot_surf_mask, zyx_center,
                                                              semiax_len)
        if return_filled_mask:
            spot_mask_filled = self.get_global_spot_mask(
                                         s_outer, zyx_center, semiax_len)
            return spot_mask, spot_mask_filled
        else:
            return spot_mask

    def calc_mean_int(self, i, semiax_len, zyx_centers, V):
        V_shape = self.V_shape
        intens = [np.mean(V[self.get_sph_surf_mask(semiax_len,
                                                   zyx_c, V_shape)])
                                                   for zyx_c in zyx_centers]
        return intens

    def filled_mask_from_um(self, zyx_vox_dim, sph_z_um, sph_xy_um, zyx_center):
        zc, yc, xc = zyx_center
        z_vd, y_vd, x_vd = zyx_vox_dim
        a = sph_xy_um/y_vd
        c = sph_z_um/z_vd
        local_mask = self.get_local_spot_mask((a, c))
        spot_mask = sph.get_global_spot_mask(local_mask, zyx_center, (a, c))
        return spot_mask

    def intersect2D(self, a, b):
        """
        Return intersecting rows between two 2D arrays 'a' and 'b'
        """
        tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
        return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

    def get_local_spot_mask(self, semiax_len, ogrid_bounds=None,
                            return_center=False):
        a, c = semiax_len
        if ogrid_bounds is None:
            a_int = int(np.ceil(a))
            c_int = int(np.ceil(c))
        else:
            o_yx, o_z = ogrid_bounds
            a_int = int(np.ceil(o_yx))
            c_int = int(np.ceil(o_z))
        # Generate a sparse meshgrid to evaluate 3D spheroid mask
        z, y, x = np.ogrid[-c_int:c_int+1, -a_int:a_int+1, -a_int:a_int+1]
        # 3D spheroid equation
        mask_s = (x**2 + y**2)/(a**2) + z**2/(c**2) <= 1
        if return_center:
            return mask_s, None
        else:
            return mask_s

    def get_global_spot_mask(self, local_spot_mask, zyx_center, semiax_len,
                             additional_local_arr=None):
        spot_mask = np.zeros(self.V_shape, local_spot_mask.dtype)
        if additional_local_arr is not None:
            additional_global_arr = np.zeros(self.V_shape,
                                              additional_local_arr.dtype)
        else:
            additional_global_arr = None
        Z, Y, X = self.V_shape
        spot_mask, spot_mask_2 = self.index_local_into_global_mask(
                                 spot_mask, local_spot_mask,
                                 zyx_center, semiax_len, Z, Y, X,
                                 additional_global_arr=additional_global_arr,
                                 additional_local_arr=additional_local_arr
        )
        if additional_local_arr is not None:
            return spot_mask, spot_mask_2
        else:
            return spot_mask

    def get_slice_G_to_L(self, semiax_len, zyx_c, Z, Y, X):
        a, c = semiax_len
        a_int = int(np.ceil(a))
        c_int = int(np.ceil(c))
        zc, yc, xc = zyx_c

        z_min = zc-c_int
        z_max = zc+c_int+1
        z_min_crop, z_max_crop = None, None
        y_min_crop, y_max_crop = None, None
        x_min_crop, x_max_crop = None, None

        # Check z size and crop if needed
        if z_min < 0:
            z_min_crop = abs(z_min)
            z_min = 0
        if z_max > Z:
            z_max_crop = Z-z_max
            z_max = Z

        # Check y size and crop if needed
        y_min = yc-a_int
        y_max = yc+a_int+1
        if y_min < 0:
            y_min_crop = abs(y_min)
            y_min = 0
        if y_max > Y:
            y_max_crop = Y-y_max
            y_max = Y

        # Check x size and crop if needed
        x_min = xc-a_int
        x_max = xc+a_int+1
        if x_min < 0:
            x_min_crop = abs(x_min)
            x_min = 0
        if x_max > X:
            x_max_crop = X-x_max
            x_max = X

        slice_G_to_L = (slice(z_min,z_max),
                        slice(y_min,y_max),
                        slice(x_min,x_max))
        slice_crop = (slice(z_min_crop,z_max_crop),
                      slice(y_min_crop,y_max_crop),
                      slice(x_min_crop,x_max_crop))
        return slice_G_to_L, slice_crop


    def index_local_into_global_mask(self, global_mask, local_mask, zyx_c,
                                       semiax_len, Z, Y, X,
                                       additional_global_arr=None,
                                       additional_local_arr=None,
                                       do_sum=False, return_slice=False):
        """
        Insert local spot mask (which has shape = size of the spot)
        into global mask (which has shape = shape of V_spots).
        If the size of the local spot exceeds the bounds of V_spots it is
        cropped before being inserted.
        """
        slice_G_to_L, slice_crop = self.get_slice_G_to_L(
                                                     semiax_len, zyx_c, Z, Y, X)

        cropped_mask = local_mask[slice_crop].copy()

        if additional_local_arr is not None:
            cropped_mask_2 = additional_local_arr[slice_crop].copy()

        try:
            if do_sum:
                global_mask[slice_G_to_L] += cropped_mask
            else:
                global_mask[slice_G_to_L][cropped_mask] = True
            if additional_local_arr is not None:
                additional_global_arr[slice_G_to_L] = cropped_mask_2
        except:
            traceback.print_exc()
            print(Z, Y, X)
            print(zyx_c)
            print(slice_G_to_L)
            print(slice_crop)
            print(cropped_mask.shape)
            import pdb; pdb.set_trace()
        if additional_local_arr is not None:
            if return_slice:
                return (global_mask, additional_global_arr,
                        slice_G_to_L, slice_crop)
            else:
                return global_mask, additional_global_arr
        else:
            if return_slice:
                return global_mask, None, slice_G_to_L, slice_crop
            else:
                return global_mask, None

    def insert_grown_spot_id(self, grow_step_i, id, zyx_vox_dim,
                             zyx_resolution, zyx_c, spots_3D_lab):
        a, c = self.calc_semiax_len(grow_step_i, zyx_vox_dim,
                                    zyx_resolution)
        semiax_len = (np.ceil(a), np.ceil(c))
        local_spot_mask = self.get_local_spot_mask(semiax_len)
        Z, Y, X = self.V_shape

        slice_G_to_L, slice_crop = self.get_slice_G_to_L(
                                    semiax_len, zyx_c, Z, Y, X)
        cropped_mask = local_spot_mask[slice_crop]
        # Avoid spot overwriting existing spot
        cropped_mask[spots_3D_lab[slice_G_to_L] != 0] = False
        spots_3D_lab[slice_G_to_L][cropped_mask] = id
        return spots_3D_lab

    def get_spots_mask(self, i, zyx_vox_dim, zyx_resolution, zyx_centers,
                       method='min_spheroid', dtype=np.bool, ids=[]):
        if method == 'min_spheroid':
            Z, Y, X = self.V_shape
            # Calc spheroid semiaxis lengths in pixels (c: z, a: x and y)
            semiax_len = self.calc_semiax_len(i, zyx_vox_dim, zyx_resolution)
            local_spot_mask = self.get_local_spot_mask(semiax_len)
            # Pre-allocate arrays
            spots_mask = np.zeros(self.V_shape, dtype)
            temp_mask = np.zeros(self.V_shape, bool)
            # Insert local spot masks into global mask
            in_pbar = tqdm(desc='Building spots mask',
                        total=len(zyx_centers),
                        unit=' spot', leave=False, position=1, ncols=100)
            for i, zyx_c in enumerate(zyx_centers):
                (temp_mask, _, slice_G_to_L,
                slice_crop) = self.index_local_into_global_mask(
                                                 temp_mask, local_spot_mask,
                                                 zyx_c, semiax_len, Z, Y, X,
                                                 return_slice=True
                )
                if dtype == np.bool:
                    spots_mask = np.logical_or(spots_mask, temp_mask)
                elif dtype == np.uint16:
                    cropped_mask = local_spot_mask[slice_crop]
                    spots_mask[slice_G_to_L][cropped_mask] = ids[i]
                in_pbar.update(1)
            in_pbar.close()
        elif method == 'unsharp_mask':
            # result = unsharp_mask(self.V_ch, radius=10, amount=5,
            #                       preserve_range=True)
            blurred = gaussian(self.V_ch, sigma=3)
            sharp = self.V_ch - blurred
            th = threshold_isodata(sharp.max(axis=0))
            spots_mask = sharp > th
        return spots_mask

    def calc_foregr_sum(self, j, V_spots, min_int, spot_filled_mask):
        return np.sum(V_spots[spot_filled_mask] - min_int)

    def calc_mNeon_mKate_sum(self, V_spots, V_ref, mNeon_norm, mKate_norm,
                                   spot_filled_mask):
        V_mNeon_norm = V_spots[spot_filled_mask]/mNeon_norm
        V_ref_norm = V_ref[spot_filled_mask]/mKate_norm
        return np.sum(V_mNeon_norm-V_ref_norm)

    def volume(self):
        return np.pi*(self.xys**2)*self.zs*4/3

    def eval_grow_cond(self, semiax_len, zyx_centers, num_spots, grow_prev, V,
                       min_int, count_iter, verb=False):
        V_shape = self.V_shape
        grow = [False]*num_spots
        # Iterate each peak
        for b, (zyx_c, g1) in enumerate(zip(zyx_centers, grow_prev)):
            # Check if growing should continue (g1=True in grow_prev)
            if g1:
                sph_surf_mask, spot_filled_mask = self.get_sph_surf_mask(
                                                       semiax_len,
                                                       zyx_c, V_shape,
                                                       return_filled_mask=True)
                surf_pixels = V[sph_surf_mask]
                surf_mean = np.mean(surf_pixels)
                # Check if the current spheroid hit another peak
                num_zyx_c = np.count_nonzero(spot_filled_mask[
                                                          zyx_centers[:,0],
                                                          zyx_centers[:,1],
                                                          zyx_centers[:,2]])
                hit_neigh = num_zyx_c > 1
                if not hit_neigh:
                    cond = surf_mean > min_int or count_iter>20
                    grow[b] = cond
                if False:
                    print(zyx_c, surf_mean, min_int, grow[b])
                    apps.imshow_tk(sph_surf_mask[zyx_c[0]],
                                   dots_coords=zyx_centers,
                                   x_idx=2)
                    import pdb; pdb.set_trace()
                    # self.bp.pausehere()
        # print(grow)
        # self.bp.pausehere()
        return grow

class spotMAX:
    def init(self, finterval, num_frames, data_path, filename,
                 hdf=False, do_save=False,
                 replace=False, vNUM='v1', run_num=1,
                 do_ref_chQUANT=False, ref_ch_loaded=False,
                 calc_ref_ch_len=False, do_spotSIZE=False,
                 do_gaussian_fit=False):
        self.ref_ch_loaded = ref_ch_loaded
        self.do_save = do_save
        self.do_ref_chQUANT = do_ref_chQUANT
        self.calc_ref_ch_len = calc_ref_ch_len
        self.do_spotSIZE = do_spotSIZE
        self.do_gaussian_fit = do_gaussian_fit
        self.vNUM = vNUM
        self.run_num = run_num
        self.data_path = data_path
        if do_save and not os.path.exists(data_path):
            os.mkdir(data_path)
        filename_noEXT, ext = os.path.splitext(filename)
        filename = f'{run_num}_{filename_noEXT}_{vNUM}{ext}'
        self.filename = filename
        data_file_path = data_path + '/' + filename
        hdf_mode = 'w'
        self.data_file_path = data_file_path
        if hdf and do_save:
            temp_dirpath = tempfile.mkdtemp()
            # temp_dirpath = temp_dir.name
            HDF_temp_path = os.path.join(temp_dirpath, filename)
            self.HDF_temp_path = HDF_temp_path
            self.store_HDF = pd.HDFStore(HDF_temp_path,  mode=hdf_mode,
                                         complevel=5, complib = 'zlib')
        if do_ref_chQUANT and do_save:
            temp_dirpath = tempfile.mkdtemp()
            # temp_dirpath = temp_dir.name
            skel_coords_filename = f'{run_num}_mtNetQUANT_skel_coords_{vNUM}.h5'
            HDF_temp_path1 = os.path.join(temp_dirpath, skel_coords_filename)
            self.HDF_mtNet_path = f'{data_path}/{skel_coords_filename}'
            self.HDF_temp_path1 = HDF_temp_path1
            self.HDF_mtNetZYX = pd.HDFStore(HDF_temp_path1,  mode=hdf_mode,
                                            complevel=5, complib = 'zlib')
        self.dframes = []
        self.time = []
        for i in range(num_frames):
            self.time.append(i) #keys for MutiIndex concatenate
        self.bp = apps.tk_breakpoint()
        self.generate_summary_df_bp = apps.tk_breakpoint()
        self.perm_err_bp = apps.tk_breakpoint(title='File permission ERROR')


    def nearest_nonzero(self, a, y, x):
        cell_ID = a[y,x]
        if cell_ID == 0:
            r, c = np.nonzero(a)
            dist = ((r - y)**2 + (c - x)**2)
            min_idx = dist.argmin()
            min_dist = dist[min_idx]
            if min_dist > 5:
                drop = True
            else:
                drop = False
            return a[r[min_idx], c[min_idx]], drop
        else:
            drop = False
            return cell_ID, drop

    def generate_summary_df(self, IDs_unique, num_spots, segm_npy_3D,
                            zyx_vox_dim, timestamp, finterval, frame_i,
                            ref_mask, df_norm_spots, df_norm_ref_ch,
                            cca_df, V_spots_shape, rp_segm_3D, is_segm_3D,
                            ref_chQUANT_data=None, compare_IDs=True,
                            df_spotFIT=None, V_spots_raw=None,
                            gaussian_fit_done=False, spots_mask=None,
                            predict_cell_cycle=False, filter_by_ref_ch=False,
                            debug=False):
        if compare_IDs:
            segm_IDs = [cell_prop.label for cell_prop in rp_segm_3D]
            if segm_IDs != list(IDs_unique):
                IDs_unique, num_spots = self.add_row_cells_NOspots(IDs_unique,
                                                               num_spots,
                                                               rp_segm_3D)
        num_cells = len(IDs_unique)
        timestamps, times_min = self.calc_time(timestamp, finterval,
                                               frame_i, num_cells)
        if self.ref_ch_loaded:
            if self.do_ref_chQUANT:
                self.ref_chQUANT(ref_mask, segm_npy_3D, zyx_vox_dim,
                                 IDs_unique)
                mtVol_vox = self.mtVol_vox
                mtVol_um3 = self.mtVol_um3
                mtLen_um = self.mtLen_um
                n_mtFragments = self.n_mtFragments
            else:
                mtVol_vox = ref_chQUANT_data.mtVol_vox
                mtVol_um3 = ref_chQUANT_data.mtVol_um3
                mtLen_um = ref_chQUANT_data.mtLen_um
                n_mtFragments = ref_chQUANT_data.n_mtFragments
        else:
            mtVol_vox = np.zeros(num_cells, np.uint8)
            mtVol_um3 = np.zeros(num_cells, np.uint8)
            mtLen_um = np.zeros(num_cells, np.uint8)
            n_mtFragments = np.zeros(num_cells, np.uint8)
        self.calc_volume(segm_npy_3D, zyx_vox_dim, rp_segm_3D, is_segm_3D)
        self.pad_zeros_IDs(IDs_unique)  # add 0s to 0 cell IDs
        (cc_stages, cc_nums, relationships,
        relatives_IDs, OFs,
        bud_moth_area_ratios,
        bud_moth_vol_ratios) = self.get_cca_info(cca_df, frame_i, IDs_unique)
        summary_df = pd.DataFrame({
                            'frame_i': [frame_i]*num_cells,
                            'Cell_ID': IDs_unique,
                            'timestamp': timestamps,
                            'time_min': times_min,
                            'cell_area_pxl': self.areas_pxl,
                            'cell_area_um2': self.areas_um2,
                            'ratio_areas_bud_moth': bud_moth_area_ratios,
                            'cell_vol_vox': self.volumes_vox,
                            'cell_vol_fl': self.volumes_fl,
                            'ratio_volumes_bud_moth': bud_moth_vol_ratios,
                            'cell_cycle_stage': cc_stages,
                            'predicted_cell_cycle_stage': cc_stages,
                            'generation_num': cc_nums,
                            'num_spots': num_spots,
                            'ref_ch_vol_vox': mtVol_vox,
                            'ref_ch_vol_um3': mtVol_um3,
                            'ref_ch_vol_len_um': mtLen_um,
                            'ref_ch_num_fragments': n_mtFragments,
                            'relationship': relationships,
                            'relative_ID': relatives_IDs,
                            'OF': [int(OF) for OF in OFs]
                                  }).set_index('Cell_ID')
        if predict_cell_cycle:
            if cca_df is not None:
                summary_df_moth_S = summary_df[
                                        (cca_df['Cell cycle stage'] == 'S')
                                      & (cca_df['Relationship'] == 'mother')]
                bud_IDs = summary_df_moth_S['relative_ID']
                IDs = summary_df_moth_S.index
                for ID, bud_ID in zip(IDs, bud_IDs):
                    num_spots_bud = summary_df.at[bud_ID,
                                                  'ref_ch_num_fragments']
                    if num_spots_bud > 0:
                        predicted_ccs = 'G2/M'
                    else:
                        ratio_bm = summary_df.at[ID, 'ratio_volumes_bud_moth']
                        if ratio_bm > 0.3:
                            predicted_ccs = 'G2/M'
                        else:
                            predicted_ccs = 'S'
                    summary_df.at[ID,
                                  'predicted_cell_cycle_stage'] = predicted_ccs
                    summary_df.at[bud_ID,
                                  'predicted_cell_cycle_stage'] = predicted_ccs
            # print(summary_df[['Cell Cycle Stage', 'Ratio volumes bud/mother',
            #        '# of fragments in ref_ch', 'Predicted cell cycle stage']])

        if df_norm_spots is not None:
            (summary_df
                .loc[:, 'spots_ch_norm_val']) = df_norm_spots['spots_ch norm.']
        else:
            summary_df.loc[:, 'spots_ch_norm_val'] = 0
        summary_df.loc[:, 'ref_ch_norm_val'] = df_norm_ref_ch['ref_ch norm.']
        # Add mtDNA amount from spots_lab
        if spots_mask is not None and V_spots_raw is not None:
            backgr_df = df_norm_spots['spots_ch norm.']
            spots_foregr_amount = self.calc_spotSIZE_fluoresc(
                                              ref_mask, spots_mask, V_spots_raw,
                                              segm_npy_3D, rp_segm_3D,
                                              filter_by_ref_ch
            )
            # Total fluorescence intensity of the spots channel inside the
            # the reference channel
            spots_ch_amount = self.calc_INref_foregr_fluoresc(
                                            ref_mask, V_spots_raw,
                                            segm_npy_3D, rp_segm_3D,
                                            debug=debug
            )
            summary_df['spotsize_tot_fluoresc'] = spots_foregr_amount
            summary_df['spots_INref_tot_fluoresc'] = spots_ch_amount
        # Add mtDNA amount data from fit
        if df_spotFIT is not None and gaussian_fit_done:
            self.agg_spotFIT(df_spotFIT, IDs_unique)
            summary_df = self.add_agg_spotFIT(summary_df)
        summary_df['creation_datetime'] = self.datetime_now(num_cells)
        summary_df.reset_index(inplace=True)
        self.summary_df = summary_df
        self.appn_list(summary_df)

    def add_agg_spotFIT(self, summary_df):
        summary_df['spotfit_sum_foregr_integral'] = self.spotfit_sum_integrals
        summary_df['spotfit_sum_tot_integral'] = self.integral_tot_sums
        summary_df['mean_sigma_z'] = self.average_sizes_x
        summary_df['mean_sigma_y'] = self.average_sizes_y
        summary_df['mean_sigma_x'] = self.average_sizes_z
        summary_df['std_sigma_z'] = self.std_sizes_x
        summary_df['std_sigma_y'] = self.std_sizes_y
        summary_df['std_sigma_x'] = self.std_sizes_z
        summary_df['sum_A_fit'] = self.As_sums
        summary_df['mean_B_fit'] = self.averages_B
        summary_df['solution_found'] = self.solution_found
        summary_df['mean_reduced_chisq'] = self.mean_reduced_chisqs
        summary_df['combined_p_chisq'] = self.combined_p_chisq
        summary_df['mean_RMSE'] = self.RMSEs
        summary_df['mean_NRMSE'] = self.NRMSEs
        summary_df['mean_F_NRMSE'] = self.F_NRMSEs
        summary_df['mean_ks'] = self.mean_ks_stats
        summary_df['combined_p_ks'] = self.combined_p_ks
        summary_df['mean_ks_null'] = self.all_null_ks_tests
        summary_df['mean_chisq_null'] = self.all_null_chisq_tests
        summary_df['mean_QC_passed'] = self.all_QC_passed
        return summary_df


    def get_cca_info(self, cca_df, frame_i, IDs):
        cc_stages = [0]*len(IDs)
        cc_nums = [0]*len(IDs)
        relationships = ['ND']*len(IDs)
        relatives_IDs = [0]*len(IDs)
        OFs = [False]*len(IDs)
        bud_moth_area_ratios = [0]*len(IDs)
        bud_moth_vol_ratios = [0]*len(IDs)
        if cca_df is None:
            return (cc_stages, cc_nums, relationships, relatives_IDs, OFs,
                    bud_moth_area_ratios, bud_moth_vol_ratios)
        for i, ID in enumerate(IDs):
            if 'frame_i' in cca_df.columns:
                cca_df = cca_df.reset_index().set_index(['frame_i', 'Cell_ID'])
                cc_stage = cca_df['Cell cycle stage'].at[(frame_i, ID)]
                cc_num = cca_df['# of cycles'].at[(frame_i, ID)]
                relationship = cca_df['Relationship'].at[(frame_i, ID)]
                relatives_ID = cca_df['Relative\'s ID'].at[(frame_i, ID)]
                try:
                    OF = cca_df['OF'].at[(frame_i, ID)]
                except:
                    OF = cca_df['Discard'].at[(frame_i, ID)]
            else:
                cc_stage = cca_df['Cell cycle stage'].at[ID]
                cc_num = cca_df['# of cycles'].at[ID]
                relationship = cca_df['Relationship'].at[ID]
                relatives_ID = cca_df['Relative\'s ID'].at[ID]
                try:
                    OF = cca_df['OF'].at[ID]
                except:
                    OF = cca_df['Discard'].at[ID]
            if relationship == 'mother' and cc_stage == 'S':
                try:
                    moth_area = self.areas_pxl[i]
                    moth_vol = self.volumes_vox[i]
                    bud_ID = relatives_ID
                    bud_idx = list(IDs).index(bud_ID)
                    bud_area = self.areas_pxl[bud_idx]
                    bud_vol = self.volumes_vox[bud_idx]
                    bud_moth_area_ratios[i] = bud_area/moth_area
                    bud_moth_vol_ratios[i] = bud_vol/moth_vol
                    bud_moth_area_ratios[bud_idx] = bud_area/moth_area
                    bud_moth_vol_ratios[bud_idx] = bud_vol/moth_vol
                except:
                    traceback.print_exc()
                    print('')
                    print('ERROR while retrieving cell cycle information.')
            cc_stages[i] = cc_stage
            cc_nums[i] = cc_num
            relationships[i] = relationship
            relatives_IDs[i] = relatives_ID
            OFs[i] = OF
        return (cc_stages, cc_nums, relationships, relatives_IDs, OFs,
                bud_moth_area_ratios, bud_moth_vol_ratios)

    def get_tot_mtNet_len(self):
        return self.summary_df['ref_ch_vol_len_um'].sum()

    def calc_spotSIZE_fluoresc(self, ref_mask, spots_mask, V_spots_raw,
                               segm_npy_3D, rp_segm_3D, filter_by_ref_ch):
        # Calculate total foreground fluorescence for the spots channel
        # TODO: CHECK THIS CALCULATION
        IDs = [obj.label for obj in rp_segm_3D]
        fluoresc_amounts = [0]*len(IDs)
        for i, ID in enumerate(IDs):
            segm_npy_mask_ID = segm_npy_3D==ID
            spots_mask_ID = np.logical_and(spots_mask, segm_npy_mask_ID)
            if filter_by_ref_ch:
                ref_mask_ID = np.logical_and(segm_npy_mask_ID, ref_mask)
                backgr_mask = np.logical_and(ref_mask_ID, ~spots_mask_ID)
            else:
                backgr_mask = np.logical_and(segm_npy_mask_ID, ~spots_mask_ID)
            backgr_val = np.median(V_spots_raw[backgr_mask])
            in_spots_vals = V_spots_raw[spots_mask_ID]
            in_spots_mean = in_spots_vals.mean()
            spots_vol = np.count_nonzero(in_spots_vals)
            foregr_val = (in_spots_mean-backgr_val)*spots_vol
            fluoresc_amounts[i] = foregr_val
        return fluoresc_amounts

    def calc_INref_foregr_fluoresc(self, foregr_mask, V, segm_npy_3D,
                                    rp_segm_3D, debug=False):
        # TODO: CHECK THIS CALCULATION
        IDs = [obj.label for obj in rp_segm_3D]
        fluoresc_amounts = [0]*len(IDs)
        for i, ID in enumerate(IDs):
            segm_npy_mask_ID = segm_npy_3D==ID
            foregr_mask_ID = np.logical_and(segm_npy_mask_ID, foregr_mask)
            foregr_mask_vol_ID = np.count_nonzero(foregr_mask_ID)
            if foregr_mask_vol_ID > 0:
                foregr_val_ID = np.mean(V[foregr_mask_ID])
                # Backgr. val = median(outside mask AND inside segm ID)
                backgr_vals, _ = self.get_INcell_backgr_vals(
                                         V, segm_npy_3D, foregr_mask_ID, ID,
                                         debug=debug, filter_by_ref_ch=False)
                backgr_val = np.median(backgr_vals)
                masked_amount = (foregr_val_ID-backgr_val)*foregr_mask_vol_ID
            else:
                masked_amount = 0
            fluoresc_amounts[i] = masked_amount
        return fluoresc_amounts

    def ref_chQUANT(self, ref_mask, segm_npy_3D, zyx_vox_dim, IDs):
        num_cells = len(IDs)
        mtVol_vox = [0]*num_cells
        mtVol_um3 = [0]*num_cells
        mtLen_um = [0]*num_cells
        n_mtFragments = [0]*num_cells
        vox_to_um3 = np.prod(zyx_vox_dim)
        df_zyx_li = [0]*num_cells
        pbar = tqdm(desc='Computing reference channel metrics',
                    total=len(IDs), unit=' segm_obj', ncols=100)
        for c, ID in enumerate(IDs):
            ID_mKate_thresh = np.copy(ref_mask)
            ID_mKate_thresh[segm_npy_3D != ID] = False
            ID_mtVol_vox = ID_mKate_thresh.sum()
            ID_mtVol_um3 = ID_mtVol_vox * vox_to_um3
            mtVol_vox[c] = ID_mtVol_vox
            mtVol_um3[c] = ID_mtVol_um3
            skel = skeletonize(ID_mKate_thresh).astype(bool)
            mitoskel = analyze_skeleton(skel, zyx_vox_dim)
            n_mtFragments[c] = mitoskel.num_fragm
            if self.calc_ref_ch_len:
                try:
                    mitoskel.analyse(skel, zyx_vox_dim)
                    mtLen_um[c] = mitoskel.get_tot_length()
                    df_zyx_li[c] = mitoskel.df_zyx
                except:
                    df_zyx_li[c] = mitoskel.empty_df_zyx()
            else:
                df_zyx_li[c] = mitoskel.empty_df_zyx()
            pbar.update(1)
        pbar.close()
        df_zyx = pd.concat(df_zyx_li, keys=IDs)
        df_zyx.index.set_names('Cell_ID', level=0, inplace=True)
        self.df_zyx = df_zyx
        self.mtNet_done = True
        self.mtVol_vox = mtVol_vox
        self.mtVol_um3 = mtVol_um3
        self.mtLen_um = mtLen_um
        self.n_mtFragments = n_mtFragments

    def add_row_cells_NOspots(self, IDs_unique, num_spots, rp_segm_3D):
        segm_IDs = np.asarray([cell_props.label for cell_props in rp_segm_3D])
        set_IDs_unique = set(IDs_unique)
        set_segm_IDs = set(segm_IDs)

        all_IDs = np.asarray(list(set_segm_IDs.union(set_IDs_unique)))
        num_spots_with0s = np.zeros_like(all_IDs)
        for i, ID in enumerate(all_IDs):
            if ID in IDs_unique:
                nucl_i = list(IDs_unique).index(ID)
                num_spots_with0s[i] = num_spots[nucl_i]
            else:
                num_spots_with0s[i] = 0
        return all_IDs, num_spots_with0s

    def datetime_now(self, num_cells):
        datetime_now = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
        return [datetime_now for i in range(num_cells)]

    def calc_time(self, timestamp, finterval, frame_i, num_cells):
        time_s = finterval*frame_i
        frame_datetime = timestamp + timedelta(seconds=time_s)
        frame_datetime_str = frame_datetime.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
        timestamps = [frame_datetime_str for i in range(num_cells)]
        times_min = np.full(num_cells, time_s/60)
        return timestamps, times_min

    def pad_zeros_IDs(self, IDs_unique):
        num_zeros_IDs = np.count_nonzero(IDs_unique==0)
        zeros_loc = [0]*num_zeros_IDs
        self.volumes_vox = np.insert(self.volumes_vox, zeros_loc, 0)
        self.volumes_fl = np.insert(self.volumes_fl, zeros_loc, 0)
        self.areas_pxl = np.insert(self.areas_pxl, zeros_loc, 0)
        self.areas_um2 = np.insert(self.areas_um2, zeros_loc, 0)

    def agg_spotFIT(self, df_spotFIT, IDs_unique):
        self.spotfit_sum_integrals = [0]*len(IDs_unique)
        self.average_sizes_x = [0]*len(IDs_unique)
        self.average_sizes_y = [0]*len(IDs_unique)
        self.average_sizes_z = [0]*len(IDs_unique)
        self.std_sizes_x = [0]*len(IDs_unique)
        self.std_sizes_y = [0]*len(IDs_unique)
        self.std_sizes_z = [0]*len(IDs_unique)
        self.As_sums = [0]*len(IDs_unique)
        self.averages_B = [0]*len(IDs_unique)
        self.integral_tot_sums = [0]*len(IDs_unique)
        self.solution_found = [0]*len(IDs_unique)
        self.mean_reduced_chisqs = [0]*len(IDs_unique)
        self.combined_p_chisq = [0]*len(IDs_unique)
        self.RMSEs = [0]*len(IDs_unique)
        self.NRMSEs = [0]*len(IDs_unique)
        self.F_NRMSEs = [0]*len(IDs_unique)
        self.mean_ks_stats = [0]*len(IDs_unique)
        self.combined_p_ks = [0]*len(IDs_unique)
        self.all_null_ks_tests = [0]*len(IDs_unique)
        self.all_null_chisq_tests = [0]*len(IDs_unique)
        self.all_QC_passed = [0]*len(IDs_unique)
        for i, ID in enumerate(IDs_unique):
            if ID in df_spotFIT.index.get_level_values(0):
                spotfit_sum_integral = df_spotFIT.loc[ID]['I_foregr'].sum()
                integral_tot_sum = df_spotFIT.loc[ID]['I_tot'].sum()
                average_size_x = df_spotFIT.loc[ID]['sigma_z_fit'].mean()
                average_size_y = df_spotFIT.loc[ID]['sigma_y_fit'].mean()
                average_size_z = df_spotFIT.loc[ID]['sigma_x_fit'].mean()
                std_size_x = df_spotFIT.loc[ID]['sigma_z_fit'].std()
                std_size_y = df_spotFIT.loc[ID]['sigma_y_fit'].std()
                std_size_z = df_spotFIT.loc[ID]['sigma_x_fit'].std()
                As_sum = df_spotFIT.loc[ID]['A_fit'].sum()
                average_B = df_spotFIT.loc[ID]['B_fit'].mean()
                solution_found = df_spotFIT.loc[ID]['solution_found'].mean()
                mean_reduced_chisq = df_spotFIT.loc[ID]['reduced_chisq'].mean()
                _, combined_p_chisq = stats.combine_pvalues(
                                          df_spotFIT.loc[ID]['p_chisq'])
                RMSE = df_spotFIT.loc[ID]['RMSE'].mean()
                NRMSE = df_spotFIT.loc[ID]['NRMSE'].mean()
                F_NRMSE = df_spotFIT.loc[ID]['F_NRMSE'].mean()
                ks_stat_mean = df_spotFIT.loc[ID]['KS_stat'].mean()
                _, combined_p_ks = stats.combine_pvalues(df_spotFIT.loc[ID]['p_KS'])
                null_ks = df_spotFIT.loc[ID]['null_ks_test'].mean()
                null_chisq = df_spotFIT.loc[ID]['null_chisq_test'].mean()
                QC_passed = df_spotFIT.loc[ID]['QC_passed'].mean()
                self.spotfit_sum_integrals[i] = spotfit_sum_integral
                self.integral_tot_sums[i] = integral_tot_sum
                self.average_sizes_x[i] = average_size_x
                self.average_sizes_y[i] = average_size_y
                self.average_sizes_z[i] = average_size_z
                self.std_sizes_x[i] = std_size_x
                self.std_sizes_y[i] = std_size_y
                self.std_sizes_z[i] = std_size_z
                self.As_sums[i] = As_sum
                self.averages_B[i] = average_B
                self.solution_found[i] = solution_found
                self.mean_reduced_chisqs[i] = mean_reduced_chisq
                self.combined_p_chisq[i] = combined_p_chisq
                self.RMSEs[i] = RMSE
                self.NRMSEs[i] = NRMSE
                self.F_NRMSEs[i] = F_NRMSE
                self.mean_ks_stats[i] = ks_stat_mean
                self.combined_p_ks[i] = combined_p_ks
                self.all_null_ks_tests[i] = null_ks
                self.all_null_chisq_tests[i] = null_chisq
                self.all_QC_passed[i] = QC_passed

    def get_INcell_backgr_vals(self, V, segm_npy_3D, foregr_mask, ID,
                               how='max_z_proj', z_slice=None,
                               additional_V=None, filter_by_ref_ch=False,
                               ref_mask=None, debug=False):
        """
        Get the background intensity values inside the cell and outside of
        the foreground_mask (e.g. spots_mask). NOTE: If we filter by ref_ch
        then the background values are within the reference channel mask.
        NOTE: If the cell was segmented in 2D we cannot really get
        all the background values inside the cell in 3D. To solve this
        there are several options:
            1. Calculate the median of the max z-projection (simplest)
            2. Get the pixels of a 3D rotational volume from the 2D cell
               (not implemented yet and not simple to rotate the 3D volume)
            3. Threshold the image with a low threshold value to make sure
               that we get rid of the dark z-slices. Here the best would
               probably be 'threshold_triangle' but it needs to be tested.
               (not implemented yet)
            4. Get a different background value for each z-slice
        """
        segm_mask_ID = segm_npy_3D == ID

        if how=='max_z_proj':
            img = V.max(axis=0)
            if additional_V is not None:
                img2 = additional_V.max(axis=0)
            foregr_mask = foregr_mask.max(axis=0)
            segm_mask_ID = segm_mask_ID.max(axis=0)
            if ref_mask is not None:
                ref_mask_ID = np.logical_and(segm_mask_ID, ref_mask.max(axis=0))
        elif how=='z_slice':
            img = V[z_slice]
            if additional_V is not None:
                img2 = additional_V[z_slice]
            foregr_mask = foregr_mask[z_slice].copy()
            segm_mask_ID = segm_mask_ID[z_slice].copy()
            if ref_mask is not None:
                ref_mask_ID = np.logical_and(segm_mask_ID, ref_mask[z_slice])
        elif how=='segm_3D':
            img = V
            if additional_V is not None:
                img2 = additional_V
            if ref_mask is not None:
                ref_mask_ID = np.logical_and(segm_mask_ID, ref_mask)

        if filter_by_ref_ch:
            backgr_mask_ID = np.logical_and(ref_mask_ID, ~foregr_mask)
        else:
            backgr_mask_ID = np.logical_and(segm_mask_ID, ~foregr_mask)

        backgr_vals = img[backgr_mask_ID]

        if len(backgr_vals) == 0:
            backgr_vals = np.array([0.0])

        if additional_V is None:
            return backgr_vals, None
        else:
            return backgr_vals, img2[backgr_mask_ID]

    def calc_volume(self, segm_npy_3D, zyx_vox_dim, rp_segm_3D, is_segm_3D):
        num_cells = len(rp_segm_3D)
        zyx_vox_dim = np.asarray(zyx_vox_dim)
        vox_to_fl = zyx_vox_dim[1]*(zyx_vox_dim[2]**2) #revolution axis = y
        yx_pxl_to_um2 = zyx_vox_dim[1]*zyx_vox_dim[2]
        #print(zyx_vox_dim[1], vox_to_fl, yx_pxl_to_um2)
        cells_areas_pxl = np.zeros(num_cells)
        cells_areas_um2 = np.zeros(num_cells)
        cells_volumes_fl = np.zeros(num_cells)
        cells_volumes_vox = np.zeros(num_cells)
        if np.all(segm_npy_3D):
            self.volumes_vox = cells_volumes_vox
            self.volumes_fl = cells_volumes_fl
            self.areas_pxl = cells_areas_pxl
            self.areas_um2 = cells_areas_um2
            return
        elif is_segm_3D:
            for i, cell in enumerate(rp_segm_3D):
                cell_img = (segm_npy_3D[cell.slice]==cell.label)
                cell_volume_vox = np.count_nonzero(cell_img)
                cell_volume_fl = cell_volume_vox*vox_to_fl
                cells_volumes_vox[i] = cell_volume_vox
                cells_volumes_fl[i] = cell_volume_fl
            self.volumes_vox = cells_volumes_vox
            self.volumes_fl = cells_volumes_fl
            self.areas_pxl = cells_areas_pxl
            self.areas_um2 = cells_areas_um2
            return

        segm_npy_frame = segm_npy_3D.max(axis=0)
        cells_props = regionprops(segm_npy_frame)
        #Iterate through objects: rotate by orientation and calculate volume
        for i, cell in enumerate(cells_props):
            # print('Not rotated cell area = {}'.format(cell.area))
            orient_deg = cell.orientation*180/math.pi
            cell_img = (segm_npy_frame[cell.slice]==cell.label).astype(int)
            cell_aligned_long_axis = rotate(cell_img,-orient_deg,resize=True,
                                            order=3,preserve_range=True)
            # cell_aligned_area = regionprops(cell_aligned_long_axis)[0].area
            # print('Rotated cell area float = {0:.2f}'.format(np.sum(cell_aligned_long_axis)))
            # print('Rotated cell area int = {0:.2f}'.format(np.sum(np.round(cell_aligned_long_axis))))
            radii = np.sum(cell_aligned_long_axis, axis=1)/2
            cell_volume_vox = np.sum(np.pi*radii**2)
            cell_volume_fl = cell_volume_vox*vox_to_fl
            cells_areas_pxl[i] = cell.area
            cells_areas_um2[i] = cell.area*yx_pxl_to_um2
            cells_volumes_vox[i] = cell_volume_vox
            cells_volumes_fl[i] = cell_volume_fl
            # print('Cell volume = {0:.4e}'.format(cell_volume))
        self.volumes_vox = np.asarray(cells_volumes_vox)
        self.volumes_fl = np.asarray(cells_volumes_fl)
        self.areas_pxl = np.asarray(cells_areas_pxl)
        self.areas_um2 = np.asarray(cells_areas_um2)

    def appn_list(self, df):
        self.dframes.append(df)

    def spotQUANT(self, p_ellips_df_MulIDs, V_spots_raw, V_ref_mask,
                  rp_segm_3D, segm_npy_3D, IDs, zyx_vox_size, zyx_resolution,
                  verbose=0, filter_by_ref_ch=True, inspect=False):

        df_spots_h5 = p_ellips_df_MulIDs.df_IDs

        if not self.do_spotSIZE:
            return df_spots_h5, None

        IDs_with_spots = []
        dfs = []
        spots_mask = np.zeros(V_spots_raw.shape, bool)
        spots_3D_labs = []
        ID_bboxs_lower = []
        ID_3Dslices = []
        dfs_intersect = []
        pbar_IDs = tqdm(total=len(rp_segm_3D), leave=True,
                        unit=' cell', ncols=100)

        for obj_3D in rp_segm_3D:
            ID = obj_3D.label

            # Skip IDs that don't have spots
            if ID not in df_spots_h5.index.get_level_values(0):
                continue

            pbar_IDs.set_description(f'spotSIZE on ID {ID}/{len(rp_segm_3D)}')

            IDs_with_spots.append(ID)
            V_spots_ID = V_spots_raw[obj_3D.slice].copy()
            df_spots_h5_ID = df_spots_h5.loc[ID].copy()
            min_z, min_y, min_x, _, _, _ = obj_3D.bbox
            ID_bbox_lower = (min_z, min_y, min_x)
            mask_ID = obj_3D.image
            if filter_by_ref_ch:
                V_ref_mask_ID = np.logical_and(mask_ID,
                                               V_ref_mask[obj_3D.slice])
            else:
                V_ref_mask_ID = None
            spotFIT_data = spotFIT(
                               V_spots_ID, df_spots_h5_ID, zyx_vox_size,
                               zyx_resolution, ID_bbox_lower, mask_ID,
                               V_ref_mask_ID, ID, verbose=verbose,
                               inspect=inspect)
            spotFIT_data.spotSIZE()
            df_spotFIT_ID = spotFIT_data.df_spots_h5_ID

            spots_mask[obj_3D.slice][spotFIT_data.spots_3D_lab_ID>0] = True

            ID_bboxs_lower.append(ID_bbox_lower)
            ID_3Dslices.append(obj_3D.slice)

            spots_3D_labs.append(spotFIT_data.spots_3D_lab_ID)

            if not self.do_gaussian_fit:
                pbar_IDs.update()
                dfs.append(df_spotFIT_ID)
                continue

            pbar_IDs.set_description(f'spotFIT on ID {ID}/{len(rp_segm_3D)}')

            """3D Gaussian fit"""
            spotFIT_data.compute_neigh_intersect()
            spotFIT_data._fit()
            spotFIT_data._quality_control()
            if spotFIT_data.fit_again_idx:
                spotFIT_data._fit_again()

            _df_spotFIT = (spotFIT_data._df_spotFIT
                        .reset_index()
                        .drop(['intersecting_idx', 'neigh_idx',
                               'neigh_ids', 's'], axis=1)
                        .set_index('id')
                           )
            _df_spotFIT.index.names = ['spot_id']

            spotFIT_data.df_spotFIT_ID = df_spotFIT_ID.join(_df_spotFIT,
                                                            how='outer')
            spotFIT_data.df_spotFIT_ID.index.names = ['spot_id']

            dfs.append(spotFIT_data.df_spotFIT_ID)
            dfs_intersect.append(spotFIT_data.df_intersect)

            pbar_IDs.update()

        pbar_IDs.close()

        df_spotFIT = pd.concat(dfs, keys=IDs_with_spots,
                               names=df_spots_h5.index.names)

        self.spots_3D_labs = spots_3D_labs
        self.ID_bboxs_lower = ID_bboxs_lower
        self.ID_3Dslices = ID_3Dslices
        self.IDs_with_spots = IDs_with_spots
        self.dfs_intersect = dfs_intersect

        return df_spotFIT, spots_mask

    def filter_spots_by_size(self, df_spotFIT, spotsize_limits_pxl):
        _min, _max = spotsize_limits_pxl
        _df = df_spotFIT.copy()
        if _min > 0:
            _df = _df[_df['sigma_y_fit'] >= _min]
        if _max > 0:
            _df = _df[_df['sigma_y_fit'] <= _max]
        return _df


    def nearest_points_2Dzyx(self, points, all_others):
        """
        Given points (a 2D array of [z, y, x] rows) compute a 2D array of
        [z, y, x] nearest points from all_others 2D array of [z, y, x] points.
        This function can be used to assign to points coordinates in a
        3D array V the nearest nonzero element as follow:
        V[points[:,0],
          points[:,1],
          points[:,2]] = (V[nearest_points[:,0],
                            nearest_points[:,1],
                            nearest_points[:,2]])
        """
        # Compute 3D array where each ith row of each kth page is the element-wise
        # difference between kth row of points and ith row in all_others array.
        # (i.e. diff[k,i] = points[k] - all_others[i])
        diff = points[:, np.newaxis] - all_others
        # Compute 2D array of distances where
        # dist[i, j] = euclidean dist (points[i],all_others[j])
        dist = np.linalg.norm(diff, axis=2)
        # Compute a 1D array where nearest_idx[j] is the ith row point in
        # all_others that is nearest to jth row point in points.
        # (i.e. nearest_point to points[j] = all_others[nearest_idx[j]])
        nearest_idx = dist.argmin(axis=1)
        # Compute a 2D array where each ith row is the nearest all_others point
        # to ith row point in points.
        # e.g. nearest_points[i] = all_others neares point to points[i]
        nearest_points = all_others[nearest_idx]
        return nearest_points

    def normalize_ref_ch(self, V_ref, ref_mask, segm_npy_3D, rp_segm_3D,
                        use_outside_spots_mask=False, V_spots=None):
        if use_outside_spots_mask:
            outside_spots_mask = np.invert(self.spots_mask)
        mKate_norm_array = np.zeros(V_ref.shape)
        IDs = [obj.label for obj in rp_segm_3D]
        mKate_norm_means = [0]*len(IDs)
        spots_ch_ID_means = [0]*len(IDs)
        for i, ID in enumerate(IDs):
            ID_mask = segm_npy_3D==ID
            if use_outside_spots_mask:
                outside_nucl_ID = np.logical_and(outside_spots_mask, ID_mask)
                mtNet_ID_mask = np.logical_and(outside_nucl_ID, ref_mask)
                # plt.imshow_tk(mtNet_ID_mask.max(axis=0))
            else:
                mtNet_ID_mask = np.logical_and(ref_mask, ID_mask)
            mtNet_ID_popul = V_ref[mtNet_ID_mask]
            if V_spots is not None:
                # Get normalization value for V_spots using the same mask
                spots_ch_ID_popul = V_spots[mtNet_ID_mask]
                spots_ch_ID_mean = np.median(spots_ch_ID_popul)
                spots_ch_ID_means[i] = spots_ch_ID_mean
            mean_mtNet_ID = np.median(mtNet_ID_popul)
            mKate_norm_means[i] = mean_mtNet_ID
        df_norm_ref_ch = pd.DataFrame({'Cell_ID': IDs,
                                      'ref_ch norm.': mKate_norm_means})
        df_norm_ref_ch.set_index('Cell_ID', inplace=True)
        if V_spots is None:
            return df_norm_ref_ch
        else:
            df_norm_ch0 = pd.DataFrame({'Cell_ID': IDs,
                                        'spots_ch norm.': spots_ch_ID_means})
            df_norm_ch0.set_index('Cell_ID', inplace=True)
            return df_norm_ref_ch, df_norm_ch0

    def normalize_spots_ch(self, V_spots, segm_npy_3D, rp_segm_3D, zyx_vox_dim,
                        zyx_resolution, peak_coords, ref_mask,
                        df_ellips_test=None, method='min_spheroid',
                        ref_ch_mask=None):
        mNeon_norm_array = np.zeros(V_spots.shape)
        IDs = [obj.label for obj in rp_segm_3D]
        if df_ellips_test is None:
            sph = spheroid(V_spots)
            spots_mask = sph.get_spots_mask(0, zyx_vox_dim, zyx_resolution,
                                            peak_coords, method=method)
        elif method == 'thresholding':
            spots_mask = self.get_spots_mask_by_thresh(IDs, V_spots, segm_npy_3D,
                                                     df_ellips_test)
        elif method == 'reference_channel':
            spots_mask = ref_ch_mask

        self.spots_mask = spots_mask
        outside_nucl = np.invert(spots_mask)
        means_out_nucl = [0]*len(IDs)
        mNeon_mask = segm_npy_3D > 0
        for i, ID in enumerate(IDs):
            ID_mask = segm_npy_3D==ID
            outside_nucl_ID = np.logical_and(outside_nucl, ID_mask)
            out_nucl_ID_mask = np.logical_and(outside_nucl_ID, ref_mask)
            # plt.imshow_tk(out_nucl_ID_mask.max(axis=0))
            out_nucl_ID_popul = V_spots[out_nucl_ID_mask]
            means_out_nucl[i] = np.mean(out_nucl_ID_popul)
            # print(np.mean(out_nucl_ID_popul), np.median(out_nucl_ID_popul))
        df_norm_spots = pd.DataFrame({'Cell_ID': IDs,
                                      'spots_ch norm.': means_out_nucl})
        df_norm_spots.set_index('Cell_ID', inplace=True)
        return df_norm_spots, spots_mask

    def get_spots_mask_by_thresh(self, IDs, V_ch, segm_npy_3D, df_ellips_test):
        """Threshold each cell mNeon signal with the nucleoids' minimum
        intensity value"""
        spots_mask = np.zeros(V_ch.shape, bool)
        for ID in IDs:
            if ID in df_ellips_test.index.get_level_values(0):
                th = df_ellips_test.loc[ID]['|abs|_spot'].min()
                spots_mask_ID = np.logical_and(V_ch>th, segm_npy_3D==ID)
                spots_mask = np.logical_or(spots_mask, spots_mask_ID)
        return spots_mask


    def appn_HDF_mtNet(self, timestamp, finterval, frame_i):
        if self.mtNet_done:
            num_rows = len(self.df_zyx)
            timestamps, times_min = self.calc_time(timestamp, finterval,
                                                   frame_i, num_rows)
            self.df_zyx.insert(0, 'Time (min)', times_min)
            self.df_zyx.insert(0, 'Timestamp', timestamps)
            self.df_zyx['Creation DateTime'] = self.datetime_now(num_rows)
        if self.do_save:
            key = 'frame_{}'.format(frame_i)
            self.HDF_mtNetZYX.append(key, self.df_zyx)

    def appn_HDF(self, df, timestamp, finterval, frame_i):
        if self.do_save:
            num_rows = len(df)
            timestamps, times_min = self.calc_time(timestamp, finterval,
                                                   frame_i, num_rows)
            df.insert(0, 'Time (min)', times_min)
            df.insert(0, 'Timestamp', timestamps)
            df['Creation DateTime'] = self.datetime_now(num_rows)
            units=['yyyy/MM/dd HH:mm:ss.###', 'min', '8-bit intens.', '8-bit intens.',
                   '8-bit intens.', '8-bit intens.', '', '', 'voxels',
                   '', '', '', '', '', 'yyyy/MM/dd HH:mm:ss.###']
            key = 'frame_{}'.format(frame_i)
            try:
                self.store_HDF.append(key, df)
            except:
                traceback.print_exc()
                df.to_csv('df_error_appending_to_hdf.csv')
                exit('Execution aborted because of error when appending to HDF store')


    def concat_dframes(self, dframes, time):
        return pd.concat(dframes, keys=time)

    def save_spotFIT_done(self):
        if self.do_save:
            basename, _ = os.path.splitext(self.filename)
            spotFIT_done_fname = f'{basename}_spotFIT_done.txt'
            spotFIT_done_path = f'{self.data_path}/{spotFIT_done_fname}'
            with open(spotFIT_done_path, 'w') as txt:
                txt.write('3D Gaussian fit was performed!')
            spotFIT_NOT_done_fname = f'{basename}_spotFIT_NOT_done.txt'
            spotFIT_NOT_done_path = f'{self.data_path}/{spotFIT_NOT_done_fname}'
            try:
                os.remove(spotFIT_NOT_done_path)
            except:
                pass

    def save_spotFIT_NOT_done(self):
        if self.do_save:
            basename, _ = os.path.splitext(self.filename)
            spotFIT_NOT_done_fname = f'{basename}_spotFIT_NOT_done.txt'
            spotFIT_NOT_done_path = f'{self.data_path}/{spotFIT_NOT_done_fname}'
            with open(spotFIT_NOT_done_path, 'w') as txt:
                txt.write('3D Gaussian fit NOT performed!')

    def write_to_csv(self, mode='w'):
        if self.do_save:
            concat_dframes = self.concat_dframes(self.dframes, self.time)
            try:
                concat_dframes.to_csv(self.data_file_path, index=False,
                                      mode=mode, encoding='utf-8-sig')
            except PermissionError:
                str_li = self.data_file_path.split('/')
                filename = '/{}/{}/{}'.format(str_li[-3],str_li[-2],str_li[-1])
                self.perm_err_bp.label_fontsize = 10
                self.perm_err_bp.message = (f'File\n\n{filename}\n'
                             'is open in the OS (maybe Excel?).\n\n'
                             'Close the file before pressing "Continue"')
                self.perm_err_bp.pausehere()
                concat_dframes.to_csv(self.data_file_path, index=False,
                                      mode=mode, encoding='utf-8-sig')
            self.df_summary_pos = concat_dframes

    def write_header_info(self):
        if self.do_save:
            header_info_txt = (
            "\'vox_spot\':   Single voxel intensity value at the peak coordinates.\n"
            "                       (float: 0-1)\n\n"

            "\'vox_ref\':   Single voxel intensity value at the peak coordinates.\n"
            "                       (float: 0-1)\n\n"

            "\'|abs|_spot\': Mean intensity value within the resolution limit sized ellipsoid.\n"
            "                       (float: 0-1)\n\n"

            "\'|abs|_ref\': Mean intensity value within the resolution limit sized ellipsoid.\n"
            "                       (float: 0-1)\n\n"

            "\'|norm|_spot\':        Mean normalized intensity value within the\n"
            "                       resolution limit sized ellipsoid.\n"
            "                       Normalization done by diving absolute intensities by the mean of that channel\n"
            "                       outside of nucleoids and within mtNetwork.\n\n"

            "\'|norm|_ref\':        Mean normalized intensity value within the\n"
            "                       resolution limit sized ellipsoid.\n"
            "                       Normalization done by diving absolute intensities by the median of that channel.\n\n"

            "\'spot_vol_um3\':    Volume of the ellipsoid with maximum size\n"
            "                       determined by growing ellipsoid method.\n"
            "                       Growing is stopped once the new mean is\n"
            "                       below the mNeon normalization value\n\n"

            "\'spotfit_sum_integral\':     Result of analytical integration of the\n"
            "                      foreground fitted 3D gaussians\n\n"

            "\'sigma_n\':     Sigma along the n direction (z, y or x)\n"
            "                      of the fitted 3D gaussians\n\n"

            "\'A_fit\':     Amplitude of the fitted 3D gaussians\n\n"

            "\'B_fit\':     Background level of the fitted 3D gaussians\n\n"

            "\'n_fit\':     n center coordinates (z, y, or x) of the fitted 3D gaussians\n\n"

            "\'integral_tot\': Result of analytical integration of the\n"
            "                  total (background + model) fitted 3D gaussians\n\n"

            "\'spot-ref sum int\': Sum of the normalized voxels intensity values within\n"
            "                        the \'spot_vol_um3 (µm^3)\'.\n"
            "                        Normalization done by subtracting \'(z,y,x) norm mKate\'\n"
            "                        to each \'(z,y,x) norm mNeon\'\n."
            "                  e.g.: mKate at (z,y,x) = 50 with mKate median 25 --> (z,y,x) norm mKate = 2.\n"
            "                        mNeon at (z,y,x) = 150 with mNeon median 50 --> (z,y,x) norm mNeon = 3.\n"
            "                        mNeon-mKate sum int = 3-2 = 1 --> sum all voxels within spot_vol_um3.\n"
            "                        --> np.sum(V_mNeon_norm - V_ref_norm)\n\n"

            "\'|spot|:|ref| t-value\': t-value of the Welch's t-test between intensities \n"
            "                       distribution within mNeon and mKate resolution limit\n"
            "                       sized ellipsoid.\n\n"

            "\'|spot|:|ref| p-value (t)\': p-value of the Welch's t-test between intensities \n"
            "                       distribution within mNeon and mKate resolution limit\n"
            "                       sized ellipsoid.\n\n"

            "\'z\':                   peak \'z\' coordinate\n\n"

            "\'y\':                   peak \'y\' coordinate\n\n"

            "\'x\':                   peak \'x\' coordinate\n\n"
            )
            path = os.path.dirname(self.data_file_path)
            header_filename = f'header_col_description_{self.vNUM}.txt'
            header_info = open(f'{path}/{header_filename}', 'w')
            header_info.write(header_info_txt)
            header_info.close()

    def close_HDF(self):
        if self.do_save:
            self.store_HDF.close()
            temp_dir = os.path.dirname(self.HDF_temp_path)
            shutil.move(self.HDF_temp_path, self.data_file_path)
            shutil.rmtree(temp_dir)


    def close_HDF_mtNetZYX(self):
        if self.do_save:
            self.HDF_mtNetZYX.close()
            temp_dir = os.path.dirname(self.HDF_temp_path1)
            shutil.move(self.HDF_temp_path1, self.HDF_mtNet_path)
            shutil.rmtree(temp_dir)


class spotMAX_concat_pos:
    def __init__(self, pos_folder_parent_path,
                 foldername='spotMAX', vNUM='v9', run_num=1,
                 do_save=False):
        foldername = f'{foldername}_{vNUM}_run-num{run_num}'
        self.pos_folder_parent_path = pos_folder_parent_path
        self.keys = []
        self.ellips_test_df_li = []
        self.p_test_df_li = []
        self.p_ellips_test_df_li = []
        self.spotfit_df_li = []
        exp_path = os.path.dirname(pos_folder_parent_path)
        self.spotMAX_data_path = f'{exp_path}/{foldername}'
        self.bp = apps.tk_breakpoint()
        if not os.path.exists(self.spotMAX_data_path) and do_save:
            os.mkdir(self.spotMAX_data_path)

    def rename_columns(self, df):
        colnames = {
                'Timestamp': 'timestamp',
                'Time (min)': 'time_min',
                'Cell Area (pxl)': 'cell_area_pxl',
                'Cell Area (µm^2)': 'cell_area_um2',
                'Ratio areas bud/mother': 'ratio_areas_bud_moth',
                'Cell Volume (vox)': 'cell_vol_vox',
                'Cell Volume (fl)': 'cell_vol_fl',
                'Ratio volumes bud/mother': 'ratio_volumes_bud_moth',
                'Cell Cycle Stage': 'cell_cycle_stage',
                'Cycle repetition #': 'generation_num',
                '# of spots': 'num_spots',
                'ref_ch. Vol. (vox)': 'ref_ch_vol_vox',
                'ref_ch. Vol. (µm^3)': 'ref_ch_vol_um3',
                'ref_ch. Len. (µm)': 'ref_ch_len_um',
                '# of fragments in ref_ch': 'ref_ch_num_fragments',
                'Relationship': 'relationship',
                "Relative's ID": 'relative_ID',
                'spots_ch norm. val': 'spots_ch_norm_val',
                'ref_ch norm. val': 'ref_ch_norm_val',
                'spotSIZE_amount': 'spotsize_tot_fluoresc',
                'mNeon_amount': 'spots_INref_tot_fluoresc',
                'Creation DateTime': 'creation_datetime',
                'ref_ch_vol_um': 'ref_ch_len_um'
        }
        df.rename(columns=colnames, inplace=True)
        if 'OF' in df.columns:
            df['OF'] = df['OF'].astype(int)


    def load_df_from_allpos(self, vNUM='v9', run_num=1):
        pos_foldernames = natsorted(os.listdir(self.pos_folder_parent_path))
        ellips_test_filename = f'{run_num}_1_ellip_test_data_Summary_{vNUM}.csv'
        p_test_filename = f'{run_num}_2_p-_test_data_Summary_{vNUM}.csv'
        p_ellips_test_filename = f'{run_num}_3_p-_ellip_test_data_Summary_{vNUM}.csv'
        spotfit_filename = f'{run_num}_4_spotfit_data_Summary_{vNUM}.csv'
        for pos_name in pos_foldernames:
            pos_path = os.path.join(self.pos_folder_parent_path, pos_name)
            is_pos_folder = pos_name.find('Position_') != -1
            is_pos_path = os.path.isdir(pos_path)
            if not is_pos_folder or not is_pos_path:
                continue
            images_path = os.path.join(pos_path, 'Images')
            filenames = os.listdir(images_path)
            # Get original filename (e.g. czi microscopy file)
            for f in filenames:
                m = re.findall('(.*)_s(\d+)_', f)
                if m:
                    if len(m[0]) == 2:
                        basename = m[0][0]
                        break
                    else:
                        basename = f
                else:
                    basename = s
            self.keys.append((pos_name, basename))
            cca_df_list = [f for f in filenames if f.find('acdc_output.csv')!=-1]
            if not cca_df_list:
                cca_df_list = [f for f in filenames if f.find('cc_stage.csv')!=-1]
            cca_df = None
            if cca_df_list:
                cca_df_filename = cca_df_list[0]
                cca_df_path = f'{images_path}/{cca_df_filename}'
                cca_df = pd.read_csv(cca_df_path, index_col='Cell_ID')
            spotMAX_folderpath = (f'{self.pos_folder_parent_path}/{pos_name}'
                                 '/spotMAX_output')
            if not os.path.exists(spotMAX_folderpath):
                return
            mitoQ_filenames = os.listdir(spotMAX_folderpath)
            for filename in mitoQ_filenames:
                csv_path = f'{spotMAX_folderpath}/{filename}'
                idx = ['frame_i', 'Cell ID']
                idx1 = ['frame_i', 'Cell_ID']
                if filename.find(ellips_test_filename) != -1:
                    try:
                        ellips_test_df = pd.read_csv(csv_path, index_col=idx)
                    except ValueError:
                        ellips_test_df = pd.read_csv(csv_path, index_col=idx1)
                    if cca_df is not None:
                        ellips_test_df = self.add_cca_info(ellips_test_df,
                                                                   cca_df)
                    self.rename_columns(ellips_test_df)
                    self.ellips_test_df_li.append(ellips_test_df)
                elif filename.find(p_test_filename) != -1:
                    try:
                        p_test_df = pd.read_csv(csv_path, index_col=idx)
                    except ValueError:
                        p_test_df = pd.read_csv(csv_path, index_col=idx1)
                    if cca_df is not None:
                        p_test_df = self.add_cca_info(p_test_df, cca_df)
                    self.rename_columns(p_test_df)
                    self.p_test_df_li.append(p_test_df)
                elif filename.find(p_ellips_test_filename) != -1:
                    try:
                        p_ellips_test_df = pd.read_csv(csv_path, index_col=idx)
                    except ValueError:
                        p_ellips_test_df = pd.read_csv(csv_path, index_col=idx1)
                    if cca_df is not None:
                        p_ellips_test_df = self.add_cca_info(p_ellips_test_df,
                                                                   cca_df)
                    self.rename_columns(p_ellips_test_df)
                    self.p_ellips_test_df_li.append(p_ellips_test_df)
                elif filename.find(spotfit_filename) != -1:
                    try:
                        spotfit_df = pd.read_csv(csv_path, index_col=idx)
                    except ValueError:
                        spotfit_df = pd.read_csv(csv_path, index_col=idx1)
                    if cca_df is not None:
                        spotfit_df = self.add_cca_info(
                                spotfit_df, cca_df, debug=False
                        )
                    self.rename_columns(spotfit_df)
                    self.spotfit_df_li.append(spotfit_df)
                elif filename.find(f'{run_num}_{vNUM}_analysis_inputs.csv') !=-1:
                    self.analysis_inputs_path = csv_path

    def add_cca_info(self, pos_df, pos_cca_df, debug=False):
        if 'Cell Cycle Stage' in pos_df.columns:
            col = 'Cell Cycle Stage'
        else:
            col = 'cell_cycle_stage'
        frames = pos_df.index.get_level_values(0)
        IDs = pos_df.index.get_level_values(1)
        if 'frame_i' in pos_cca_df.columns:
            pos_cca_df = (
                pos_cca_df.reset_index()
                          .set_index(['frame_i', 'Cell_ID'])
            )
            if 'Cell cycle stage' in pos_cca_df.columns:
                cc_stages = pos_cca_df['Cell cycle stage'].loc[0].loc[IDs]
                cc_nums = pos_cca_df['# of cycles'].loc[0].loc[IDs]
                relationships = pos_cca_df['Relationship'].loc[0].loc[IDs]
                relatives_IDs = pos_cca_df['Relative\'s ID'].loc[0].loc[IDs]
                OFs = pos_cca_df['OF'].loc[0].loc[IDs]
            else:
                # Use new acdc_output format which always has frame_i column
                cc_stages = pos_cca_df['cell_cycle_stage'].loc[0].loc[IDs]
                cc_nums = pos_cca_df['generation_num'].loc[0].loc[IDs]
                relationships = pos_cca_df['relationship'].loc[0].loc[IDs]
                relatives_IDs = pos_cca_df['relative_ID'].loc[0].loc[IDs]
                OFs = pos_cca_df['is_cell_excluded'].loc[0].loc[IDs]
        else:
            cc_stages = pos_cca_df['Cell cycle stage'].loc[IDs]
            cc_nums = pos_cca_df['# of cycles'].loc[IDs]
            relationships = pos_cca_df['Relationship'].loc[IDs]
            relatives_IDs = pos_cca_df['Relative\'s ID'].loc[IDs]
            OFs = pos_cca_df['OF'].loc[IDs]
        pos_df['cell_cycle_stage'] = cc_stages.to_list()
        pos_df['generation_num'] = cc_nums.to_list()
        pos_df['relationship'] = relationships.to_list()
        pos_df['relative_ID'] = relatives_IDs.to_list()
        pos_df['OF'] = OFs.to_list()
        return pos_df

    def combine_moth_bud_dfs(self, df_bud, df_moth):
        # Generate total dataframe data by summing mothers to relative buds data
        df_budIDs_of_mothers = df_bud.index
        df_tot_S = df_moth.loc[df_budIDs_of_mothers]+df_bud

        # For some reason summing dataframes changes the order of columns.
        # Restore original order
        df_tot_S = df_tot_S[df_moth.columns]

        # Replace summed data that is the same for bud and mother
        cols_to_repl = ['timestamp', 'time_min',
                       'generation_num', 'spots_ch_norm_val',
                       'ref_ch_norm_val', 'creation_datetime',
                       'relationship', 'relative_ID', 'OF',
                       'ratio_areas_bud_moth', 'ratio_volumes_bud_moth',
                       'predicted_cell_cycle_stage', 'original_filename',
                       'index']
        for col in cols_to_repl:
            if col in df_tot_S.columns:
                df_tot_S[col] = (df_moth.loc[df_tot_S.index][col])

        df_tot_S.index.names = ['cell_cycle_stage', 'Moth_ID',
                                'Position_n', 'frame_i']

        # Combine the means and std (pooling)
        means_colnames = ['mean_sigma_z', 'mean_sigma_y', 'mean_sigma_x',
                          'mean_B_fit', 'solution_found', 'mean_reduced_chisq',
                          'mean_RMSE', 'mean_NRMSE', 'mean_F_NRMSE', 'mean_ks',
                          'mean_ks_null', 'mean_chisq_null', 'mean_QC_passed']
        df_moth_S = df_moth.loc[df_budIDs_of_mothers]
        for col in means_colnames:
            N1 = df_moth_S['num_spots']
            N2 = df_bud['num_spots']
            if col in df_moth.columns:
                try:
                    M1 = df_moth_S[col].astype(float)
                    M2 = df_bud[col]
                    pooled_mean = (M1*N1+N2*M2)/(N1+N2)
                    df_tot_S[col] = pooled_mean
                except:
                    traceback.print_exc()
                    import pdb; pdb.set_trace()

        stds_colnames = ['std_sigma_z', 'std_sigma_y', 'std_sigma_x']
        for col in stds_colnames:
            N1 = df_moth_S['num_spots']
            N2 = df_bud['num_spots']
            if col in df_moth.columns:
                S1 = df_moth_S[col]
                S2 = df_bud[col]
                pooled_std = ((N1-1)*S1+(N2-1)*S2)/(N1+N2-2)
                df_tot_S[col] = pooled_std

        # Combine the p-values
        p_vals_colnames = ['combined_p_chisq', 'combined_p_ks']
        for p_val_colname in p_vals_colnames:
            if p_val_colname in df_moth.columns:
                df_tot_S.drop(p_val_colname, axis=1, inplace=True)
                pval_moth = df_moth_S[p_val_colname]
                pval_bud = df_bud[p_val_colname]
                pval_tot = []
                for p_moth, p_bud in zip(pval_moth, pval_bud):
                    stat, p_tot = stats.combine_pvalues([p_moth, p_bud])
                    pval_tot.append(p_tot)
                df_tot_S[p_val_colname] = pval_tot

        return df_tot_S

    def generate_bud_moth_tot_dfs(self, df_li):
        """Create 3 dataframes for buds, mothers and total"""
        names = ['Position_n', 'original_filename', 'frame_i', 'Moth_ID']
        all_pos_df = pd.concat(df_li, keys=self.keys, names=names)

        if 'S' not in all_pos_df['cell_cycle_stage'].values:
            all_pos_df = (all_pos_df
                            .reset_index()
                            .set_index(
                                  ['cell_cycle_stage', 'relative_ID',
                                  'Position_n', 'frame_i']))
            return None, None, all_pos_df

        df_moth = (all_pos_df.loc[all_pos_df['relationship'] == 'mother']
                         .sort_values(by=['cell_cycle_stage', 'Moth_ID'])
                         .reset_index()
                         .set_index(['cell_cycle_stage', 'Moth_ID',
                                     'Position_n', 'frame_i']))
        # df_bud will have as index the mothers IDs for easier indexing later
        df_bud = (all_pos_df.loc[all_pos_df['relationship'] == 'bud']
                         .sort_values(by=['cell_cycle_stage', 'Moth_ID'])
                         .reset_index()
                         .set_index(['cell_cycle_stage', 'relative_ID',
                                     'Position_n', 'frame_i']))
        # Combine mother and bud into total
        df_tot_S = self.combine_moth_bud_dfs(df_bud, df_moth)

        # Write same ratio_areas_bud_moth to bud df
        df_bud.loc[df_tot_S.index, 'ratio_volumes_bud_moth'] = (
                                      df_tot_S['ratio_volumes_bud_moth'])
        df_bud.loc[df_tot_S.index, 'ratio_areas_bud_moth'] = (
                                      df_tot_S['ratio_areas_bud_moth'])
        df_bud.rename(columns={'Moth_ID': 'Bud_ID'}, inplace=True)

        # df_bud_save is the actual DataFrame for buds data to be saved
        df_bud_save = (df_bud.reset_index()
                        .set_index(['cell_cycle_stage', 'Bud_ID',
                                    'Position_n', 'frame_i'])
                        .sort_index())

        # Remove columns from mother dataframe that are not in total dataframe
        df_moth = df_moth[df_tot_S.columns]

        # Combine S and G1 dataframes
        if 'G1' in df_moth.index.get_level_values(level=0):
            df_tot = pd.concat([df_moth.loc['G1'], df_tot_S.loc['S']],
                                keys=['G1', 'S'],
                                names=['cell_cycle_stage', 'Moth_ID',
                                       'Position_n', 'frame_i']
                                ).sort_index()
        else:
            df_tot = df_tot_S

        return df_moth, df_bud_save, df_tot

    def save_AllPos_df(self, df, df_name):
        if df is not None:
            df.to_csv(f'{self.spotMAX_data_path}/{df_name}',
                      encoding='utf-8-sig')

    def save_ALLPos_analysis_inputs(self, spotMAX_inputs_path):
        try:
            shutil.copy2(spotMAX_inputs_path, self.spotMAX_data_path)
        except shutil.SameFileError:
            pass

class twobuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title, message, button_1_text, button_2_text,
                 geometry="+800+400"):
        self.left_b_val=False
        root = tk.Tk()
        self.root = root
        root.lift()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0,
                                       column=0,
                                       columnspan=2,
                                       pady=4, padx=4)

        tk.Button(root,
                  text=button_1_text,
                  command=self.left_b_cb,
                  width=10,).grid(row=4,
                                  column=0,
                                  pady=16, padx=8)

        tk.Button(root,
                  text=button_2_text,
                  command=self.close,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=16, padx=8)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        tk.mainloop()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user.')

    def left_b_cb(self):
        self.left_b_val=True
        self.root.quit()
        self.root.destroy()

    def close(self):
        self.root.quit()
        self.root.destroy()

class threebuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title, geometry, message, button_1_text,
                 button_2_text, button_3_text, path):
        global root
        self.path=path
        self.append=False
        root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0, column=0, columnspan=2, pady=4, padx=4)

        tk.Button(root,
                  text=button_1_text,
                  command=self.append_button,
                  width=10,).grid(row=4,
                                  column=0,
                                  pady=8, padx=8)

        tk.Button(root,
                  text=button_2_text,
                  command=self.close,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=8, padx=8)
        tk.Button(root,
                  text=button_3_text,
                  command=self.open_path_explorer,
                  width=25).grid(row=5,
                                 column=0,
                                 columnspan=2)

        tk.mainloop()

    def append_button(self):
        self.append=True
        root.quit()
        root.destroy()

    def open_path_explorer(self):
        subprocess.Popen('explorer "{}"'.format(os.path.normpath(self.path)))

    def close(self):
        root.quit()
        root.destroy()

def plot_mtNet(df_mtNet, ax, lw=1):
    try:
        for cell_ID, df_cID in df_mtNet.groupby(level=0):
            for segm_ID, df_scID in df_cID.groupby(level=2):
                ax[1,1].plot(df_scID['x'], df_scID['y'], c='lime', lw=lw)
    except:
        print('WARNING: ref_chQUANT failed! Network will not be plotted')

def plot_and_save(z_proj_V_mNeon, z_proj_V_ref, local_max_coords,
                  z_proj_norm_mtNet, z_proj_V_mNeon_norm,
                  local_max_coords_p_ellips_test, data_path, t_end, t0,
                  tot_mtNet_len, basename, df_mtNet, pdf=None, frame_i=0,
                  V_spots_sharp=None):
    fig, ax = plt.subplots(2,3)
    fig.set_size_inches(16.53,11.69)

    ax[0,0].imshow(z_proj_V_mNeon)
    ax[0,0].axis('off')
    ax[0,0].set_title('Spots z-projection')

    ax[0,1].imshow(z_proj_V_ref, cmap='gist_heat')
    ax[0,1].axis('off')
    ax[0,1].set_title('Reference channel z-projection')

    ax[0,2].imshow(z_proj_V_mNeon)
    ax[0,2].plot(local_max_coords[:, 2],
                 local_max_coords[:, 1],
                 'r.', ms=0.5) #add red dots at peaks positions
    ax[0,2].axis('off')
    ax[0,2].set_title('Spots plus local maxima')

    ax10_img = z_proj_V_mNeon_norm if V_spots_sharp is None else V_spots_sharp.max(axis=0)
    ax[1,0].imshow(ax10_img)
    ax[1,0].axis('off')
    ax[1,0].set_title('Processed (e.g. sharpened) spots z-projection')

    ax[1,1].imshow(z_proj_norm_mtNet, cmap='gist_heat')
    ax[1,1].axis('off')
    ax[1,1].set_title('Reference channel gaussian filtered z-projection')

    ax[1,2].imshow(z_proj_V_mNeon_norm)
    ax[1,2].plot(local_max_coords_p_ellips_test[:, 2],
                 local_max_coords_p_ellips_test[:, 1],
                 'r.', ms=0.5) #add red dots at peaks positions
    ax[1,2].axis('off')
    ax[1,2].set_title('Spots z-proj. plus valid local maxima')

    # Plot mtNetwork
    plot_mtNet(df_mtNet, ax)

    folder_name = os.path.basename(os.path.dirname(data_path))
    fig.suptitle('Total execution time = {0:.3f} s\nFolder name: {1}'
                 ', frame_i: {2}'
                .format(t_end-t0, folder_name, frame_i), y=0.97, size=14)

    ax02_l, ax02_b, ax02_r, ax02_t = ax[0,2].get_position().get_points().flatten()
    fig.text((ax02_r+ax02_l)/2, ax02_b-0.015,
         'Number of nucleoids = {}'.format(len(local_max_coords)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    ax12_l, ax12_b, ax12_r, ax12_t = ax[1,2].get_position().get_points().flatten()
    fig.text((ax12_r+ax12_l)/2, ax12_b-0.015,
         'Number of nucleoids = {}'.format(len(local_max_coords_p_ellips_test)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    ax11_l, ax11_b, ax11_r, ax11_t = ax[1,1].get_position().get_points().flatten()
    fig.text((ax11_r+ax11_l)/2, ax11_b-0.015, 'mtNetwork length (um) = {0:.2f}'
                             .format(tot_mtNet_len),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=11)
    if pdf is None:
        fig_path = '{}/{}_plt.pdf'.format(data_path, basename)
        plt.savefig(fig_path, orientation='landscape')
    else:
        pdf.savefig()
    plt.close()

def keep_only_one_obj(ref_mask_slice, cc_stage):
    lab = label(ref_mask_slice)
    rp = regionprops(lab)
    num_obj_to_keep = 2 if cc_stage=='S' else 1
    if rp:
        ids_remove = [obj.label for obj in rp]
        areas = [obj.area for obj in rp]
        for _ in range(num_obj_to_keep):
            if not areas:
                break
            max_obj_idx = areas.index(max(areas))
            max_obj_id = ids_remove[max_obj_idx]
            ids_remove.pop(max_obj_idx)
            areas.pop(max_obj_idx)
        for id in ids_remove:
            ref_mask_slice[lab == id] = False
    return ref_mask_slice

class lstq_Model:
    def __init__(self, nfev=0):
        pass

    @staticmethod
    @njit(parallel=True)
    def jac_gauss3D(coeffs, data, z, y, x, num_spots, num_coeffs, const=0):
        # Gradient ((m,n) Jacobian matrix):
        # grad[i,j] = derivative of f[i] wrt coeffs[j]
        # e.g. m data points with n coeffs --> grad with m rows and n col
        grad = np.empty((len(z), num_coeffs*num_spots))
        ns = np.arange(0,num_coeffs*num_spots,num_coeffs)
        for i in prange(num_spots):
            n = ns[i]
            coeffs_i = coeffs[n:n+num_coeffs]
            z0, y0, x0, sz, sy, sx, A, B = coeffs_i
            # Center rotation around peak center
            zc = z - z0
            yc = y - y0
            xc = x - x0
            # Build 3D gaussian by multiplying each 1D gaussian function
            gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
            gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
            gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
            f_x = 1/(sx*np.sqrt(2*np.pi))
            f_y = 1/(sy*np.sqrt(2*np.pi))
            f_z = 1/(sz*np.sqrt(2*np.pi))
            g = gauss_x*gauss_y*gauss_z
            f = f_x*f_y*f_z
            fg = f*g

            # Partial derivatives
            d_g_sz = g * zc**2 / (sz**3)
            d_f_sz = A/(np.sqrt(2*np.pi)*(sz**2))
            d_fg_sz = g*d_f_sz + f*d_g_sz

            d_g_sy = g * yc**2 / (sy**2)
            d_f_sy = -A/(np.sqrt(2*np.pi)*(sy**2))
            d_fg_sy = g*d_f_sz + f*d_g_sz

            d_g_sx = g * xc**2 / (sx**2)
            d_f_sx = A/(np.sqrt(2*np.pi)*(sx**2))
            d_fg_sx = g*d_f_sz + f*d_g_sz

            # Gradient array
            grad[:,n] = A*fg * zc / (sz**2) # wrt zc
            grad[:,n+1] = A*fg * yc / (sy**2) # wrt yc
            grad[:,n+2] = A*fg * xc / (sx**2) # wrt xc
            grad[:,n+3] = d_fg_sz # wrt sz
            grad[:,n+4] = d_fg_sy # wrt sy
            grad[:,n+5] = d_fg_sx # wrt sx
            grad[:,n+6] = fg # wrt A
        grad[:,-1] = np.ones(len(x)) # wrt B
        return -grad

    @staticmethod
    @njit(parallel=False)
    def _gauss3D(z, y, x, coeffs, num_spots, num_coeffs, const):
        model = np.zeros(len(z))
        n = 0
        B = coeffs[-1]
        for i in range(num_spots):
            coeffs_i = coeffs[n:n+num_coeffs]
            z0, y0, x0, sz, sy, sx, A = coeffs_i
            # Center rotation around peak center
            zc = z - z0
            yc = y - y0
            xc = x - x0
            # Build 3D gaussian by multiplying each 1D gaussian function
            gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
            gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
            gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
            f_x = 1/(sx*np.sqrt(2*np.pi))
            f_y = 1/(sy*np.sqrt(2*np.pi))
            f_z = 1/(sz*np.sqrt(2*np.pi))
            g = gauss_x*gauss_y*gauss_z
            f = f_x*f_y*f_z
            fg = f*g
            model += A*fg
            n += num_coeffs
        return model + const + B

    def gaussian_3D(self, z, y, x, coeffs, B=0):
        """Non-NUMBA version of the model"""
        z0, y0, x0, sz, sy, sx, A = coeffs
        # Center rotation around peak center
        zc = z - z0
        yc = y - y0
        xc = x - x0
        # Build 3D gaussian by multiplying each 1D gaussian function
        gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
        gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
        gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
        f_x = 1/(sx*np.sqrt(2*np.pi))
        f_y = 1/(sy*np.sqrt(2*np.pi))
        f_z = 1/(sz*np.sqrt(2*np.pi))
        g = gauss_x*gauss_y*gauss_z
        f = f_x*f_y*f_z
        fg = f*g
        gauss = A*fg+B
        return gauss

    def compute_const(self, z, y, x, const_coeffs):
        const = 0
        for const_c in const_coeffs:
            const += self.gaussian_3D(z, y, x, const_c)
        return const


    def residuals(self, coeffs, data, z, y, x, num_spots, num_coeffs, const=0):
        self.pbar.update(1)
        f = self._gauss3D
        return data - f(z, y, x, coeffs, num_spots, num_coeffs, const)

    def goodness_of_fit(self, y_obs, y_model, ddof, is_linear_regr=False):
        # Degree of freedom
        N = len(y_obs)
        dof = N-ddof

        # Reduced chi square
        try:
            # Normalize to sum 1
            y_obs_chi = y_obs/y_obs.sum()
            y_model_chi = y_model/y_model.sum()
            chisq, p_chisq = stats.chisquare(
                y_obs_chi, y_model_chi, ddof=ddof
            )
            reduced_chisq = chisq/dof
        except:
            chisq = 0
            p_chisq = 1
            reduced_chisq = 0
            print('WARNING: error calculating chisquare')

        # Sum of squared errors
        SSE = np.sum(np.square(y_obs-y_model))
        # Total sum of squares
        y_mean = y_obs.mean()
        SST = np.sum(np.square(y_obs-y_mean))
        # NOTE: R-square is valid ONLY for linear regressions
        R_sq = 1 - (SSE/SST)
        # Adjusted R squared
        adj_Rsq = 1 - (((1-R_sq)*(N-1))/(N-ddof-1))

        # Root mean squared error (a.k.a "standard error of the regression")
        RMSE = np.sqrt(SSE/dof)
        # Normalized mean squared error
        NRMSE = RMSE/y_mean
        # Frank relative NRMSE (i.e. NRMSE normalized to 0,1
        # with 1 being perfect fit)
        F_NRMSE = 2/(1+np.exp(NRMSE))

        # Kolmogorov–Smirnov test
        ks, p_ks = stats.ks_2samp(y_obs, y_model)
        if is_linear_regr:
            return (reduced_chisq, p_chisq, R_sq, RMSE, ks, p_ks, adj_Rsq,
                    NRMSE, F_NRMSE)
        else:
            return reduced_chisq, p_chisq, RMSE, ks, p_ks, NRMSE, F_NRMSE

    def get_bounds_init_guess(self, num_spots_s, num_coeffs, fit_ids,
                              fit_idx, spots_centers, spots_3D_lab_ID,
                              spots_rp, spots_radii_pxl, V_spots_ID,
                              spots_Bs_guess, spots_B_mins):

        low_limit = np.zeros(num_spots_s*num_coeffs+1)
        high_limit = np.zeros(num_spots_s*num_coeffs+1)
        init_guess_s = np.zeros(num_spots_s*num_coeffs+1)
        n = 0
        # center bounds limit
        xy_cbl = 0.2
        z_cbl = 0.1
        # Sigma bound limit multiplier
        s_f = 3
        _pi_f = np.sqrt(2*np.pi)
        max_s_z = spots_radii_pxl[:,0].max()
        max_s_yx = spots_radii_pxl[:,1].max()
        B_min = min(spots_B_mins)
        A_max = max([V_spots_ID[spots_3D_lab_ID==obj.label].sum()
                     for obj in spots_rp])+1
        for i, id in zip(fit_idx, fit_ids):
            z0, y0, x0 = spots_centers[i]
            c, b, a = spots_radii_pxl[i]
            B_guess = spots_Bs_guess[i]
            spot_mask = spots_3D_lab_ID == id
            raw_vals = V_spots_ID[spot_mask]
            # A_min = np.sum(raw_vals-raw_vals.min())
            A_guess = np.sum(raw_vals)/num_spots_s
            # z0, y0, x0, sz, sy, sx, A = coeffs
            low_lim = np.array([z0-z_cbl, y0-xy_cbl, x0-xy_cbl,
                                 0.5, 0.5, 0.5, 0])
            high_lim = np.array([z0+z_cbl, y0+xy_cbl, x0+xy_cbl,
                                 max_s_z, max_s_yx, max_s_yx, A_max])
            guess = np.array([z0, y0, x0, c, b, a, A_guess])
            low_limit[n:n+num_coeffs] = low_lim
            high_limit[n:n+num_coeffs] = high_lim
            init_guess_s[n:n+num_coeffs] = guess
            n += num_coeffs
        low_limit[-1] = B_min
        high_limit[-1] = np.inf
        init_guess_s[-1] = B_guess
        bounds = (low_limit, high_limit)
        return bounds, init_guess_s

    def integrate(self, zyx_center, zyx_sigmas, A, B,
                  sum_obs=0, lower_bounds=None, upper_bounds=None,
                  verbose=0):
        """Integrate Gaussian peaks with erf function.

        Parameters
        ----------
        zyx_center : (3,) ndarray
            [zc, yc, xc] ndarray centre coordinates of the peak
        zyx_sigmas : (3,) ndarray
            [zs, ys, xs] ndarray sigmas of the peak.
        A : float
            Amplitude of the peak
        B : float
            Background level of the peak
        lower_bounds : ndarray
            [z, y, x] lower bounds of the integration volume. If None the
            lower bounds will be equal to -1.96*zyx_sigmas (95%)
        upper_bounds : ndarray
            [z, y, x] upper bounds of the integration volume. If None the
            upper bounds will be equal to 1.96*zyx_sigmas (95%)
        sum_obs: float
            Printed alongside with the returned I_tot is verbose==3. Used for
            debugging to check that sum_obs and I_tot are in the same order
            of magnitude.


        Returns
        -------
        I_tot: float
            Result of the total integration.
        I_foregr: float
            Result of foregroung integration (i.e. background subtracted).

        """
        # Center gaussian to peak center coords
        if lower_bounds is None:
            # Use 95% of peak as integration volume
            zyx_c1 = -1.96 * zyx_sigmas
        else:
            zyx_c1 = lower_bounds - zyx_center
        if upper_bounds is None:
            zyx_c2 = 1.96 * zyx_sigmas
        else:
            zyx_c2 = upper_bounds - zyx_center

        # Substitute variable x --> t to apply erf
        t_z1, t_y1, t_x1 = zyx_c1 / (np.sqrt(2)*zyx_sigmas)
        t_z2, t_y2, t_x2 = zyx_c2 / (np.sqrt(2)*zyx_sigmas)
        D_erf_z = erf(t_z2)-erf(t_z1)
        D_erf_y = erf(t_y2)-erf(t_y1)
        D_erf_x = erf(t_x2)-erf(t_x1)
        I_z = 0.5*D_erf_z
        I_y = 0.5*D_erf_z
        I_x = 0.5*D_erf_x
        I_foregr = A * I_z*I_y*I_x
        I_tot = I_foregr + B*np.prod(zyx_c2-zyx_c1)
        if verbose==3:
            print('--------------')
            print(f'Total integral result, observed sum = {I_tot}, {sum_obs}')
            print(f'Foregroung integral values: {I_foregr}')
            print('--------------')
        return I_tot, I_foregr

class spotFIT(spheroid):
    def __init__(self, V_spots_ID, df_spots_h5_ID, zyx_vox_size,
                 zyx_spot_min_vol_um, ID_bbox_lower, mask_ID, V_ref_mask_ID, ID,
                 verbose=0, inspect=0):
        super().__init__(V_spots_ID)
        self.ID = ID
        self.V_spots_ID = V_spots_ID
        self.df_spots_h5_ID = df_spots_h5_ID
        self.zyx_vox_size = zyx_vox_size
        self.ID_bbox_lower = ID_bbox_lower
        self.mask_ID = mask_ID
        self.zyx_spot_min_vol_um = zyx_spot_min_vol_um
        self.V_ref_mask_ID = V_ref_mask_ID
        self.verbose = verbose
        self.inspect = inspect
        # z0, y0, x0, sz, sy, sx, A = coeffs; B added as one coeff
        self.num_coeffs = 7
        self._tol = 1e-10

    def fit(self):
        verbose = self.verbose
        inspect = self.inspect
        df_spots_h5_ID = self.df_spots_h5_ID

        if verbose > 0:
            print('')
            print('Segmenting spots...')
        self.spotSIZE()

        if verbose > 0:
            print('')
            print('Computing intersections...')
        self.compute_neigh_intersect()

        if verbose > 0:
            print('')
            print('Fitting 3D gaussians...')
        self._fit()

        if verbose > 0:
            print('')
            print('Running quality control...')
        self._quality_control()

        if self.fit_again_idx:
            if verbose > 0:
                print('')
                print('Attempting to fit again spots that '
                      'did not pass quality control...')
            self._fit_again()

        if verbose > 0:
            print('')
            print('Fitting process done.')

        _df_spotFIT = (self._df_spotFIT
                        .reset_index()
                        .drop(['intersecting_idx', 'neigh_idx',
                               's', 'neigh_ids'], axis=1)
                        .set_index('id')
                       )
        _df_spotFIT.index.names = ['spot_id']

        df_spots_h5_ID = self.df_spots_h5_ID

        self.df_spotFIT_ID = df_spots_h5_ID.join(_df_spotFIT, how='outer')
        self.df_spotFIT_ID.index.names = ['spot_id']

        if verbose > 1:
            print('Summary results:')
            print(_df_spotFIT)
            if 'vox mNeon (uint8)' in self.df_spotFIT_ID.columns:
                cols = ['vox mNeon (uint8)', '|abs| mNeon (uint8)',
                        'I_tot', 'I_foregr', 'sigma_y_fit']
            else:
                cols = ['vox_spot', '|abs|_spot',
                        'I_tot', 'I_foregr', 'sigma_y_fit']
            print(self.df_spotFIT_ID[cols])

    def spotSIZE(self):
        df_spots_h5_ID = self.df_spots_h5_ID
        V_spots_ID_denoise = gaussian(self.V_spots_ID, 0.8)
        min_z, min_y, min_x = self.ID_bbox_lower
        zyx_vox_dim = self.zyx_vox_size
        zyx_spot_min_vol_um = self.zyx_spot_min_vol_um
        mask_ID = self.mask_ID
        V_ref_mask_ID = self.V_ref_mask_ID

        # Build spot mask and get background values
        num_spots = len(df_spots_h5_ID)
        self.num_spots = num_spots
        spots_centers = df_spots_h5_ID[['z', 'y', 'x']].to_numpy()
        spots_centers -= [min_z, min_y, min_x]
        self.spots_centers = spots_centers
        spots_mask = self.get_spots_mask(0, zyx_vox_dim, zyx_spot_min_vol_um,
                                         spots_centers)
        if V_ref_mask_ID is None:
            backgr_mask = np.logical_and(mask_ID, ~spots_mask)
        else:
            backgr_mask = np.logical_and(V_ref_mask_ID, ~spots_mask)

        backgr_vals = V_spots_ID_denoise[backgr_mask]
        backgr_mean = backgr_vals.mean()
        backgr_std = backgr_vals.std()

        self.backgr_mean = backgr_mean
        self.backgr_std = backgr_std

        limit = backgr_mean + 3*backgr_std

        # Build seeds mask for the expansion process
        self.spot_ids = df_spots_h5_ID.index.to_list()
        seed_size = np.array(zyx_spot_min_vol_um)/2
        spots_seeds = self.get_spots_mask(0, zyx_vox_dim, seed_size,
                                         spots_centers, dtype=np.uint16,
                                         ids=self.spot_ids)
        spots_3D_lab = np.zeros_like(spots_seeds)

        # Start expanding the labels
        zs, ys, xs = seed_size
        zvd, yvd, _ = zyx_vox_dim
        stop_grow_info = [] # list of (stop_id, stop_mask, stop_slice)
        stop_grow_ids = []
        max_i = 10
        max_size = max_i*yvd
        self.spots_yx_size_um = [ys+max_size]*num_spots
        self.spots_z_size_um = [zs+max_size]*num_spots
        self.spots_yx_size_pxl = [(ys+max_size)/yvd]*num_spots
        self.spots_z_size_pxl = [(zs+max_size)/zvd]*num_spots
        expanding_steps = [0]*num_spots
        self.Bs_guess = [0]*num_spots
        _spot_surf_5ps = [0]*num_spots
        _spot_surf_means = [0]*num_spots
        _spot_surf_stds = [0]*num_spots
        _spot_B_mins = [0]*num_spots
        for i in range(max_i+1):
            # Note that expanded_labels has id from df_spots_h5_ID
            expanded_labels = expand_labels(spots_seeds, distance=yvd*(i+1),
                                            zyx_vox_size=zyx_vox_dim)

            # Replace expanded labels with the stopped growing ones.
            for stop_id, stop_mask, stop_slice in stop_grow_info:
                expanded_labels[expanded_labels==stop_id] = 0
                expanded_labels[stop_slice][stop_mask] = stop_id

            # Iterate spots to determine which ones should stop growing
            spots_rp = regionprops(expanded_labels)
            for o, s_obj in enumerate(spots_rp):
                id = s_obj.label
                # Skip spots where we stopped growing
                if id in stop_grow_ids:
                    continue
                exanped_spot_mask = expanded_labels[s_obj.slice]==id
                spot_mask = spots_seeds[s_obj.slice]==id
                local_spot_surf_mask = np.logical_xor(
                                             exanped_spot_mask, spot_mask
                )
                surf_vals = V_spots_ID_denoise[s_obj.slice][local_spot_surf_mask]
                surf_mean = surf_vals.mean()
                # print('---------------')
                # print(f'ID {id} surface mean, backgr = {surf_mean}, {limit}')


                if surf_mean <= limit or i == max_i:
                    # NOTE: we use i+1 in order to include the pixels that
                    # are <= to the limit
                    stop_grow_info.append((id, s_obj.image, s_obj.slice))
                    stop_grow_ids.append(id)
                    self.spots_yx_size_um[o] = ys+yvd*(i+1)
                    self.spots_z_size_um[o] = zs+yvd*(i+1)
                    self.spots_yx_size_pxl[o] = (ys+yvd*(i+1))/yvd
                    self.spots_z_size_pxl[o] = (zs+yvd*(i+1))/zvd
                    # Insert grown spot into spots lab used for fitting
                    c_idx = self.spot_ids.index(id)
                    zyx_c = spots_centers[c_idx]
                    spots_3D_lab = self.insert_grown_spot_id(
                                                     i+1, id, zyx_vox_dim,
                                                     zyx_spot_min_vol_um,
                                                     zyx_c, spots_3D_lab)
                    raw_spot_surf_vals = (self.V_spots_ID
                                           [s_obj.slice]
                                           [local_spot_surf_mask])
                    self.Bs_guess[o] = np.median(raw_spot_surf_vals)
                    _spot_surf_5ps[o] = np.quantile(raw_spot_surf_vals, 0.05)
                    _mean = raw_spot_surf_vals.mean()
                    _spot_surf_means[o] = _mean
                    _std = raw_spot_surf_vals.std()
                    _spot_surf_stds[o] = _std
                    _spot_B_mins[o] = _mean-5*_std

            # print(stop_grow_ids)
            # print(f'Current step = {(i+1)}')
            # print(len(stop_grow_ids), num_spots)

            # Stop loop if all spots have stopped growing
            if len(stop_grow_ids) == num_spots:
                break

        self.spots_radii_pxl = np.column_stack(
                                        (self.spots_z_size_pxl,
                                         self.spots_yx_size_pxl,
                                         self.spots_yx_size_pxl)
        )

        self.df_spots_h5_ID['spotsize_yx_radius_um'] = self.spots_yx_size_um
        self.df_spots_h5_ID['spotsize_z_radius_um'] = self.spots_z_size_um
        self.df_spots_h5_ID['spotsize_yx_radius_pxl'] = self.spots_yx_size_pxl
        self.df_spots_h5_ID['spotsize_z_radius_pxl'] = self.spots_z_size_pxl
        self.df_spots_h5_ID['spotsize_limit'] = [limit]*num_spots

        self.df_spots_h5_ID['spot_surf_50p'] = self.Bs_guess
        self.df_spots_h5_ID['spot_surf_5p'] = _spot_surf_5ps
        self.df_spots_h5_ID['spot_surf_mean'] = _spot_surf_means
        self.df_spots_h5_ID['spot_surf_std'] = _spot_surf_stds
        self.df_spots_h5_ID['spot_B_min'] = _spot_B_mins

        # Used as a lower bound for B parameter in spotfit
        self.B_mins = _spot_B_mins

        self.spots_3D_lab_ID = spots_3D_lab

    def _fit(self):
        verbose = self.verbose
        if verbose > 1:
            print('')
            print('===============')
        t0_opt = time.time()
        num_spots = self.num_spots
        df_intersect = self.df_intersect
        spots_centers = self.spots_centers
        spots_radii_pxl = self.spots_radii_pxl
        spots_Bs_guess = self.Bs_guess
        spots_B_mins = self.B_mins
        spots_3D_lab_ID = self.spots_3D_lab_ID
        V_spots_ID = self.V_spots_ID
        num_coeffs = self.num_coeffs
        inspect = self.inspect
        spots_rp = self.spots_rp

        init_guess_li = [None]*num_spots
        fitted_coeffs = [[] for _ in range(num_spots)]
        Bs_fitted = [0]*num_spots
        all_intersect_fitted_bool = [0]*num_spots
        solution_found_li = [0]*num_spots
        iter = zip(df_intersect.index,
                   df_intersect['id'],
                   df_intersect['intersecting_idx'],
                   df_intersect['neigh_idx'])
        for count, (s, s_id, intersect_idx, neigh_idx) in enumerate(iter):
            # Get the fitted coeffs of the intersecting peaks
            intersect_coeffs = [fitted_coeffs[i] for i in intersect_idx]
            if verbose > 2:
                print('-----------')
                print(f'Current spot idx: {s}')
                print(f'Neighbours indices of current spot: {intersect_idx}')
            all_intersect_fitted = all(intersect_coeffs)
            if all_intersect_fitted:
                if verbose > 2:
                    print(f'Fully fitted spot idx: {s}')
                all_intersect_fitted_bool[s] = True
                pbar = tqdm(desc=f'Spot done {count+1}/{num_spots}',
                                  total=1, unit=' fev',
                                  position=2, leave=False, ncols=100)
                pbar.update(1)
                pbar.close()
                if verbose > 2:
                    print('-----------')
                continue
            if verbose > 2:
                print(f'Intersect. coeffs: {intersect_coeffs}')
            # Set coeffs of already fitted neighbours as model constants
            non_inters_neigh_idx = [s for s in neigh_idx
                                    if s not in intersect_idx
            ]
            if verbose > 2:
                print(f'Fitted bool: {all_intersect_fitted_bool}')
                print(f'Non-intersecting neighbours idx: {non_inters_neigh_idx}')
            neigh_fitted_coeffs = [
                        fitted_coeffs[i] for i in non_inters_neigh_idx
                        if all_intersect_fitted_bool[i]
            ]
            neigh_fitted_idx = [i for i in non_inters_neigh_idx
                                        if all_intersect_fitted_bool[i]]
            if verbose > 2:
                print('All-neighbours-fitted coeffs (model constants): '
                      f'{neigh_fitted_coeffs}')
            # Use not completely fitted neigh coeffs as initial guess
            not_all_intersect_fitted_coeffs = [
                                           fitted_coeffs[i]
                                           for i in intersect_idx
                                           if not all_intersect_fitted_bool[i]]
            if verbose > 2:
                print('Not-all-neighbours-fitted coeffs (model initial guess): '
                      f'{not_all_intersect_fitted_coeffs}')

            # Fit n intersecting spots as sum of n gaussian + model constants
            fit_idx = intersect_idx
            if verbose > 2:
                print(f'Fitting spot idx: {fit_idx}, '
                      f'with centers {zyx_centers}')

            # Fit multipeaks
            fit_spots_lab = np.zeros(spots_3D_lab_ID.shape, bool)
            fit_ids = []
            num_spots_s = len(fit_idx)
            for i in fit_idx:
                fit_id = self.df_intersect.at[i, 'id']
                fit_ids.append(fit_id)
                fit_spots_lab[spots_3D_lab_ID==fit_id] = True
            z, y, x = np.nonzero(fit_spots_lab)
            s_data = self.V_spots_ID[z,y,x]
            model = lstq_Model(100*len(z))

            # Get constants
            if neigh_fitted_idx:
                const = model.compute_const(z,y,x, neigh_fitted_coeffs)
            else:
                const = 0
            # test this https://cars9.uchicago.edu/software/python/lmfit/examples/example_reduce_fcn.html#sphx-glr-examples-example-reduce-fcn-py
            bounds, init_guess_s = model.get_bounds_init_guess(
                                             num_spots_s, num_coeffs,
                                             fit_ids, fit_idx, spots_centers,
                                             spots_3D_lab_ID, spots_rp,
                                             spots_radii_pxl, V_spots_ID,
                                             spots_Bs_guess, spots_B_mins
            )
            # bar_f = '{desc:<25}{percentage:3.0f}%|{bar:40}{r_bar}'
            model.pbar = tqdm(desc=f'Fitting spot {s} ({count+1}/{num_spots})',
                              total=100*len(z), unit=' fev',
                              position=1, leave=False, ncols=100)
            try:
                leastsq_result = least_squares(model.residuals, init_guess_s,
                                               args=(s_data, z, y, x, num_spots_s,
                                                     num_coeffs),
                                               # jac=model.jac_gauss3D,
                                               kwargs={'const': const},
                                               loss='linear', f_scale=0.1,
                                               bounds=bounds, ftol=self._tol,
                                               xtol=self._tol, gtol=self._tol)
            except:
                traceback.print_exc()
                _shape = (num_spots_s, num_coeffs)
                B_fit = leastsq_result.x[-1]
                B_guess = init_guess_s[-1]
                B_min = bounds[0][-1]
                B_max = bounds[1][-1]
                lstsq_x = leastsq_result.x[:-1]
                lstsq_x = lstsq_x.reshape(_shape)
                init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
                low_bounds_2D = bounds[0][:-1].reshape(_shape)
                high_bounds_2D = bounds[1][:-1].reshape(_shape)
                print('')
                print(self.ID)
                print(fit_ids)
                for _x, _init, _l, _h in zip(lstsq_x, init_guess_s_2D,
                                             low_bounds_2D, high_bounds_2D):
                    print('')
                    print('Centers solution: ', _x[:3])
                    print('Centers init guess: ', _init[:3])
                    print('Centers low bound: ', _l[:3])
                    print('Centers high bound: ', _h[:3])
                    print('')
                    print('Sigma solution: ', _x[3:6])
                    print('Sigma init guess: ', _init[3:6])
                    print('Sigma low bound: ', _l[3:6])
                    print('Sigma high bound: ', _h[3:6])
                    print('')
                    print('A, B solution: ', _x[6], B_fit)
                    print('A, B init guess: ', _init[6], B_guess)
                    print('A, B low bound: ', _l[6], B_min)
                    print('A, B high bound: ', _h[6], B_max)
                    print('')
                    print('')
                import pdb; pdb.set_trace()

            # model.pbar.update(100*len(z)-model.pbar.n)
            model.pbar.close()

            if inspect > 2:
            # if 1 in fit_ids and self.ID == 1:
                # sum(z0, y0, x0, sz, sy, sx, A), B = coeffs
                _shape = (num_spots_s, num_coeffs)
                B_fit = leastsq_result.x[-1]
                B_guess = init_guess_s[-1]
                B_min = bounds[0][-1]
                B_max = bounds[1][-1]
                lstsq_x = leastsq_result.x[:-1]
                lstsq_x = lstsq_x.reshape(_shape)
                init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
                low_bounds_2D = bounds[0][:-1].reshape(_shape)
                high_bounds_2D = bounds[1][:-1].reshape(_shape)
                print('')
                print(self.ID)
                print(fit_ids)
                for _x, _init, _l, _h in zip(lstsq_x, init_guess_s_2D,
                                             low_bounds_2D, high_bounds_2D):
                    print('Centers solution: ', _x[:3])
                    print('Centers init guess: ', _init[:3])
                    print('Centers low bound: ', _l[:3])
                    print('Centers high bound: ', _h[:3])
                    print('')
                    print('Sigma solution: ', _x[3:6])
                    print('Sigma init guess: ', _init[3:6])
                    print('Sigma low bound: ', _l[3:6])
                    print('Sigma high bound: ', _h[3:6])
                    print('')
                    print('A, B solution: ', _x[6], B_fit)
                    print('A, B init guess: ', _init[6], B_guess)
                    print('A, B low bound: ', _l[6], B_min)
                    print('A, B high bound: ', _h[6], B_max)
                    print('')
                    print('')
                import pdb; pdb.set_trace()
                matplotlib.use('TkAgg')
                fig, ax = plt.subplots(1,3)
                img = self.V_spots_ID
                # 3D gaussian evaluated on the entire image
                V_fit = np.zeros_like(self.V_spots_ID)
                zz, yy, xx = np.nonzero(V_fit==0)
                V_fit[zz, yy, xx] = model._gauss3D(
                                       zz, yy, xx, leastsq_result.x,
                                       num_spots_s, num_coeffs, 0)

                fit_data = model._gauss3D(z, y, x, leastsq_result.x,
                                          num_spots_s, num_coeffs, 0)

                img_fit = np.zeros_like(img)
                img_fit[z,y,x] = fit_data
                img_s = np.zeros_like(img)
                img_s[z,y,x] = s_data
                y_intens = img_s.max(axis=(0, 1))
                y_intens = y_intens[y_intens!=0]
                y_gauss = img_fit.max(axis=(0, 1))
                y_gauss = y_gauss[y_gauss!=0]
                ax[0].imshow(img.max(axis=0))
                _, yyc, xxc = np.array(spots_centers[fit_idx]).T
                ax[0].plot(xxc, yyc, 'r.')
                ax[1].imshow(V_fit.max(axis=0))
                ax[1].plot(xxc, yyc, 'r.')
                ax[2].scatter(range(len(y_intens)), y_intens)
                ax[2].plot(range(len(y_gauss)), y_gauss, c='r')
                plt.show()
                matplotlib.use('Agg')

            _shape = (num_spots_s, num_coeffs)
            B_fit = leastsq_result.x[-1]
            B_guess = init_guess_s[-1]
            lstsq_x = leastsq_result.x[:-1]
            lstsq_x = lstsq_x.reshape(_shape)
            init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
            # print(f'Fitted coeffs: {lstsq_x}')
            # Store already fitted peaks
            for i, s_fit in enumerate(fit_idx):
                fitted_coeffs[s_fit] = list(lstsq_x[i])
                init_guess_li[s_fit] = list(init_guess_s_2D[i])
                Bs_fitted[s_fit] = B_fit
                solution_found_li[s_fit] = leastsq_result.success
            # Check if now the fitted spots are fully fitted
            all_intersect_fitted = all([True if fitted_coeffs[i] else False
                                         for i in intersect_idx])
            if all_intersect_fitted:
                if verbose > 2:
                    print(f'Fully fitted spot idx: {s}')
                all_intersect_fitted_bool[s] = True
            if verbose == 2:
                print('-----------')

        self.model = model
        self.fitted_coeffs = fitted_coeffs
        self.Bs_fitted = Bs_fitted
        self.init_guess_li = init_guess_li
        self.solution_found_li = solution_found_li

        t1_opt = time.time()
        exec_time = t1_opt-t0_opt
        exec_time_delta = timedelta(seconds=exec_time)
        if verbose > 1:
            print('')
            print(f'Fitting process done in {exec_time_delta} HH:mm:ss')

    def compute_neigh_intersect(self):
        inspect = self.inspect
        verbose = self.verbose
        zyx_vox_dim = self.zyx_vox_size
        zvd, yvd, _ = zyx_vox_dim
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_3D_lab_ID_connect = label(spots_3D_lab_ID>0)
        self.spots_3D_lab_ID_connect = spots_3D_lab_ID_connect
        spots_rp = regionprops(spots_3D_lab_ID)
        self.spots_rp = spots_rp
        # Get intersect ids by expanding each single object by 2 pixels
        all_intersect_idx = []
        all_neigh_idx = []
        obj_ids = []
        num_intersect = []
        num_neigh = []
        all_neigh_ids = []
        for s, s_obj in enumerate(spots_rp):
            spot_3D_lab = np.zeros_like(spots_3D_lab_ID)
            spot_3D_lab[s_obj.slice][s_obj.image] = s_obj.label
            spot_3D_mask = spot_3D_lab>0
            expanded_spot_3D = expand_labels(spot_3D_lab, distance=yvd*2,
                                             zyx_vox_size=zyx_vox_dim)
            spot_surf_mask = np.logical_xor(expanded_spot_3D>0, spot_3D_mask)
            intersect_ids = np.unique(spots_3D_lab_ID[spot_surf_mask])
            intersect_idx = [self.spot_ids.index(id)
                             for id in intersect_ids if id!=0]
            intersect_idx.append(s)
            all_intersect_idx.append(intersect_idx)
            num_intersect.append(len(intersect_idx))

            # Get neigh idx by indexing the spots labels with the
            # connected component mask
            obj_id = np.unique(spots_3D_lab_ID_connect[spot_3D_mask])[-1]
            obj_ids.append(obj_id)
            obj_mask = np.zeros_like(spot_3D_mask)
            obj_mask[spots_3D_lab_ID_connect == obj_id] = True
            neigh_ids = np.unique(spots_3D_lab_ID[obj_mask])
            neigh_ids = [id for id in neigh_ids if id!=0]
            neigh_idx = [self.spot_ids.index(id) for id in neigh_ids]
            all_neigh_idx.append(neigh_idx)
            all_neigh_ids.append(neigh_ids)
            num_neigh.append(len(neigh_idx))


        self.df_intersect = (pd.DataFrame({
                                      'id': self.spot_ids,
                                      'obj_id': obj_ids,
                                      'num_intersect': num_intersect,
                                      'num_neigh': num_neigh,
                                      'intersecting_idx': all_intersect_idx,
                                      'neigh_idx': all_neigh_idx,
                                      'neigh_ids': all_neigh_ids})
                                      .sort_values('num_intersect')
                        )
        self.df_intersect.index.name = 's'




        if verbose > 1:
            print('Intersections info:')
            print(self.df_intersect)
            print('')

        if inspect > 1:
            apps.imshow_tk(self.V_spots_ID,
                           additional_imgs=[spots_3D_lab_ID,
                                            spots_3D_lab_ID_connect])

    def _quality_control(self):
        """
        Calculate goodness_of_fit metrics for each spot
        and determine which peaks should be fitted again
        """
        df_spotFIT = (self.df_intersect
                                 .reset_index()
                                 .set_index(['obj_id', 's']))
        df_spotFIT['QC_passed'] = 0
        df_spotFIT['null_ks_test'] = 0
        df_spotFIT['null_chisq_test'] = 0
        df_spotFIT['solution_found'] = 0

        self._df_spotFIT = df_spotFIT
        verbose = self.verbose
        inspect = self.inspect
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_3D_lab_ID_connect = self.spots_3D_lab_ID_connect
        fitted_coeffs = self.fitted_coeffs
        init_guess_li = self.init_guess_li
        Bs_fitted = self.Bs_fitted
        solution_found_li = self.solution_found_li
        num_coeffs = self.num_coeffs
        model = self.model
        img = self.V_spots_ID

        all_gof_metrics = np.zeros((self.num_spots, 7))
        self.fit_again_idx = []
        for obj_id, df_obj in df_spotFIT.groupby(level=0):
            obj_s_idxs = df_obj['neigh_idx'].iloc[0]
            # Iterate single spots
            for s in obj_s_idxs:
                s_id = df_obj.at[(obj_id, s), 'id']
                s_intersect_idx = df_obj.at[(obj_id, s), 'intersecting_idx']
                z_s, y_s, x_s = np.nonzero(spots_3D_lab_ID==s_id)

                # Compute fit data
                B_fit = Bs_fitted[s]
                s_coeffs = fitted_coeffs[s]
                s_fit_data = model.gaussian_3D(z_s, y_s, x_s, s_coeffs, B=B_fit)
                for n_s in obj_s_idxs:
                    neigh_coeffs = fitted_coeffs[n_s]
                    s_fit_data += model.gaussian_3D(z_s, y_s, x_s, neigh_coeffs)

                # Goodness of fit
                ddof = num_coeffs
                s_data = img[z_s, y_s, x_s]
                (reduced_chisq, p_chisq, RMSE, ks, p_ks, NRMSE,
                F_NRMSE) = model.goodness_of_fit(s_data, s_fit_data, ddof)

                all_gof_metrics[s] = [reduced_chisq, p_chisq, RMSE,
                                      ks, p_ks, NRMSE, F_NRMSE]

                if inspect > 2:
                # if True:
                # if s_id==3 and self.ID==5:
                    print('')
                    print('----------------------------')
                    print(f'Spot data max = {s_data.max():.3f}, '
                          f'spot fit max = {s_fit_data.max():.3f}')
                    print(f'Intersecting idx = {s_intersect_idx}')
                    print(f'Neighbours idx = {obj_s_idxs}')
                    print('Spot idx =', s)
                    print(f'Reduced chisquare = {reduced_chisq:.3f}, '
                          f'p = {p_chisq:.4f}')
                    print(f'KS stat = {ks:.3f}, p = {p_ks:.4f}')
                    # print(f'R_sq = {R_sq:.3f}, Adj. R-sq = {adj_Rsq:.3f}')
                    print(f'RMSE = {RMSE:.3f}')
                    print(f'NRMSE = {NRMSE:.3f}')
                    print(f'F_NRMSE = {F_NRMSE:.3f}')

                    # Initial guess
                    (z0_guess, y0_guess, x0_guess,
                    sz_guess, sy_guess, sx_guess,
                    A_guess) = init_guess_li[s]

                    # Fitted coeffs
                    (z0_fit, y0_fit, x0_fit,
                    sz_fit, sy_fit, sx_fit,
                    A_fit) = fitted_coeffs[s]

                    print('----------------------------')
                    print(f'Init guess center = ({z0_guess:.2f}, '
                                               f'{y0_guess:.2f}, '
                                               f'{x0_guess:.2f})')
                    print(f'Fit center =        ({z0_fit:.2f}, '
                                               f'{y0_fit:.2f}, '
                                               f'{x0_fit:.2f})')
                    print('')
                    print(f'Init guess sigmas = ({sz_guess:.2f}, '
                                               f'{sy_guess:.2f}, '
                                               f'{sx_guess:.2f})')
                    print(f'Sigmas fit        = ({sz_fit:.2f}, '
                                               f'{sy_fit:.2f}, '
                                               f'{sx_fit:.2f})')
                    print('')
                    print(f'A, B init guess   = ({A_guess:.3f}, '
                                               f'{np.nan})')
                    print(f'A, B fit          = ({A_fit:.3f}, '
                                               f'{B_fit:.3f})')
                    print('----------------------------')


                    matplotlib.use('TkAgg')
                    fig, ax = plt.subplots(1,3, figsize=[18,9])

                    img_s = np.zeros_like(img)
                    img_s[z_s, y_s, x_s] = s_data

                    img_s_fit = np.zeros_like(img)
                    img_s_fit[z_s, y_s, x_s] = s_fit_data

                    y_intens = img[int(z0_guess), int(y0_guess)]
                    # y_intens = y_intens[y_intens!=0]

                    y_gauss = img_s_fit[int(z0_guess), int(y0_guess)]
                    x_gauss = [i for i, yg in enumerate(y_gauss) if yg != 0]
                    y_gauss = y_gauss[y_gauss!=0]


                    ax[0].imshow(img.max(axis=0), vmax=img.max())
                    ax[1].imshow(img_s_fit.max(axis=0), vmax=img.max())
                    ax[2].scatter(range(len(y_intens)), y_intens)
                    ax[2].plot(x_gauss, y_gauss, c='r')
                    # ax[2].scatter(range(len(y_intens)), y_intens)
                    # ax[2].plot(range(len(y_gauss)), y_gauss, c='r')

                    # l = x_obj.min()
                    # b = y_obj.min()
                    #
                    # r = x_obj.max()
                    # t = y_obj.max()
                    #
                    # ax[0].set_xlim((l-2, r+2))
                    # ax[0].set_ylim((t+2, b-2))
                    #
                    # ax[1].set_xlim((l-2, r+2))
                    # ax[1].set_ylim((t+2, b-2))

                    plt.show()
                    matplotlib.use('Agg')

        # Automatic outliers detection
        NRMSEs = all_gof_metrics[:,5]
        Q1, Q3 = np.quantile(NRMSEs, q=(0.25, 0.75))
        IQR = Q3-Q1
        self.QC_limit = Q3 + (1.5*IQR)

        if False:
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(2,4)
            ax = ax.flatten()

            sns.histplot(x=all_gof_metrics[:,0], ax=ax[0])
            sns.boxplot(x=all_gof_metrics[:,0], ax=ax[4])
            ax[0].set_title('Reduced chisquare')

            sns.histplot(x=all_gof_metrics[:,2], ax=ax[1])
            sns.boxplot(x=all_gof_metrics[:,2], ax=ax[5])
            ax[1].set_title('RMSE')

            sns.histplot(x=all_gof_metrics[:,5], ax=ax[2])
            sns.boxplot(x=all_gof_metrics[:,5], ax=ax[6])
            ax[2].axvline(self.QC_limit, color='r', linestyle='--')
            ax[6].axvline(self.QC_limit, color='r', linestyle='--')
            ax[2].set_title('NMRSE')

            sns.histplot(x=all_gof_metrics[:,6], ax=ax[3])
            sns.boxplot(x=all_gof_metrics[:,6], ax=ax[7])
            ax[3].set_title('F_NRMSE')

            plt.show()
            matplotlib.use('Agg')

        # Given QC_limit determine which spots should be fitted again
        for obj_id, df_obj in df_spotFIT.groupby(level=0):
            obj_s_idxs = df_obj['neigh_idx'].iloc[0]
            # Iterate single spots
            for s in obj_s_idxs:
                gof_metrics = all_gof_metrics[s]

                (reduced_chisq, p_chisq, RMSE,
                ks, p_ks, NRMSE, F_NRMSE) = gof_metrics

                # Initial guess
                (z0_guess, y0_guess, x0_guess,
                sz_guess, sy_guess, sx_guess,
                A_guess) = init_guess_li[s]

                # Fitted coeffs
                B_fit = Bs_fitted[s]
                (z0_fit, y0_fit, x0_fit,
                sz_fit, sy_fit, sx_fit,
                A_fit) = fitted_coeffs[s]

                # Solution found
                solution_found = solution_found_li[s]

                # Store s idx of badly fitted peaks
                num_s_in_obj = len(obj_s_idxs)
                s_intersect_idx = df_obj.at[(obj_id, s), 'intersecting_idx']
                num_intersect_s = len(s_intersect_idx)
                if NRMSE > self.QC_limit and num_intersect_s < num_s_in_obj:
                    if verbose > 2:
                        print('')
                        print(f'Fit spot idx {s} again.')
                        print('----------------------------')
                    self.fit_again_idx.append(s)
                    continue

                # Store properties of good peaks
                zyx_c = np.abs(np.array([z0_fit, y0_fit, x0_fit]))
                zyx_sigmas = np.abs(np.array([sz_fit, sy_fit, sx_fit]))

                I_tot, I_foregr = model.integrate(
                                zyx_c, zyx_sigmas, A_fit, B_fit,
                                lower_bounds=None, upper_bounds=None
                )

                gof_metrics = (reduced_chisq, p_chisq,
                               ks, p_ks, RMSE, NRMSE, F_NRMSE)

                self.store_metrics_good_spots(obj_id, s, fitted_coeffs[s],
                                              I_tot, I_foregr, gof_metrics,
                                              solution_found, B_fit)

                if verbose > 1:
                    print('')
                    print(f'Sigmas fit = ({sz_fit:.3f}, {sy_fit:.3f}, {sx_fit:.3f})')
                    print(f'A fit = {A_fit:.3f}, B fit = {B_fit:.3f}')
                    print('Total integral result, fit sum, observed sum = '
                          f'{I_tot:.3f}, {s_fit_data.sum():.3f}, {s_data.sum():.3f}')
                    print(f'Foregroung integral value: {I_foregr:.3f}')
                    print('----------------------------')

    def _fit_again(self):
        fit_again_idx = self.fit_again_idx
        df_intersect_fit_again = (
                               self.df_intersect
                               .loc[fit_again_idx]
                               .sort_values(by='num_intersect')
                               .reset_index()
                               .set_index(['obj_id', 's'])
        )

        bad_fit_idx = fit_again_idx.copy()
        num_spots = len(df_intersect_fit_again)
        num_coeffs = self.num_coeffs
        model = self.model
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_centers = self.spots_centers
        spots_radii_pxl = self.spots_radii_pxl
        spots_Bs_guess = self.Bs_guess
        spots_B_mins = self.B_mins
        fitted_coeffs = self.fitted_coeffs
        init_guess_li = self.init_guess_li
        img = self.V_spots_ID
        verbose = self.verbose
        inspect = self.inspect
        spots_rp = self.spots_rp

        # Iterate each badly fitted spot and fit individually again
        for count, (obj_id, s) in enumerate(df_intersect_fit_again.index):
            neigh_idx = df_intersect_fit_again.loc[(obj_id, s)]['neigh_idx']
            s_id = df_intersect_fit_again.loc[(obj_id, s)]['id']
            s_intersect_idx = df_intersect_fit_again.at[(obj_id, s),
                                                        'intersecting_idx']
            good_neigh_idx = [s for s in neigh_idx if s not in bad_fit_idx]

            z_s, y_s, x_s = np.nonzero(spots_3D_lab_ID==s_id)

            # Constants from good neigh idx
            const_coeffs = [fitted_coeffs[good_s] for good_s in good_neigh_idx]
            const = model.compute_const(z_s, y_s, x_s, const_coeffs)

            # Bounds and initial guess
            num_spots_s = 1
            bounds, init_guess_s = model.get_bounds_init_guess(
                                         num_spots_s, num_coeffs,
                                         [s_id], [s], spots_centers,
                                         spots_3D_lab_ID, spots_rp,
                                         spots_radii_pxl, img,
                                         spots_Bs_guess, spots_B_mins
            )

            # Fit with constants
            s_data = img[z_s, y_s, x_s]
            model.pbar = tqdm(desc=f'Fitting spot {s} ({count+1}/{num_spots})',
                                  total=100*len(z_s), unit=' fev',
                                  position=1, leave=False, ncols=100)
            leastsq_result = least_squares(model.residuals, init_guess_s,
                                           args=(s_data, z_s, y_s, x_s,
                                                 num_spots_s, num_coeffs),
                                           # jac=model.jac_gauss3D,
                                           kwargs={'const': const},
                                           loss='linear', f_scale=0.1,
                                           bounds=bounds, ftol=self._tol,
                                           xtol=self._tol, gtol=self._tol)
            model.pbar.close()

            # Goodness of fit
            ddof = num_coeffs
            s_fit_data =  model._gauss3D(z_s, y_s, x_s,
                                         leastsq_result.x,
                                         1, num_coeffs, const)
            (reduced_chisq, p_chisq, RMSE, ks, p_ks,
            NRMSE, F_NRMSE) = model.goodness_of_fit(s_data, s_fit_data, ddof)

            # Initial guess
            (z0_guess, y0_guess, x0_guess,
            sz_guess, sy_guess, sx_guess,
            A_guess) = init_guess_li[s]

            # Fitted coeffs
            (z0_fit, y0_fit, x0_fit,
            sz_fit, sy_fit, sx_fit,
            A_fit, B_fit) = leastsq_result.x


            zyx_c = np.abs(np.array([z0_fit, y0_fit, x0_fit]))
            zyx_sigmas = np.abs(np.array([sz_fit, sy_fit, sx_fit]))

            I_tot, I_foregr = model.integrate(
                            zyx_c, zyx_sigmas, A_fit, B_fit,
                            lower_bounds=None, upper_bounds=None
            )

            gof_metrics = (reduced_chisq, p_chisq,
                           ks, p_ks, RMSE, NRMSE, F_NRMSE)

            self.store_metrics_good_spots(
                                     obj_id, s, leastsq_result.x[:-1],
                                     I_tot, I_foregr, gof_metrics,
                                     leastsq_result.success, B_fit=B_fit
            )

            if inspect > 2:
            # if True:
                print('')
                print('----------------------------')
                if NRMSE > self.QC_limit:
                    print('Quality control NOT passed!')
                else:
                    print('Quality control passed!')
                print(f'Spot data max = {s_data.max():.3f}, '
                      f'spot fit max = {s_fit_data.max():.3f}')
                print(f'Intersecting idx = {s_intersect_idx}')
                print(f'Neighbours idx = {neigh_idx}')
                print('Spot idx =', s)
                print(f'Reduced chisquare = {reduced_chisq:.3f}, '
                      f'p = {p_chisq:.4f}')
                print(f'KS stat = {ks:.3f}, p = {p_ks:.4f}')
                # print(f'R_sq = {R_sq:.3f}, Adj. R-sq = {adj_Rsq:.3f}')
                print(f'RMSE = {RMSE:.3f}')
                print(f'NRMSE = {NRMSE:.3f}')
                print(f'F_NRMSE = {F_NRMSE:.3f}')
                print('')
                print(f'Sigmas fit = ({sz_fit:.3f}, {sy_fit:.3f}, {sx_fit:.3f})')
                print(f'A fit = {A_fit:.3f}, B fit = {B_fit:.3f}')
                print('Total integral result, fit sum, observed sum = '
                      f'{I_tot:.3f}, {s_fit_data.sum():.3f}, {s_data.sum():.3f}')
                print(f'Foregroung integral value: {I_foregr:.3f}')
                print('----------------------------')


                matplotlib.use('TkAgg')
                fig, ax = plt.subplots(1,3, figsize=[18,9])

                img_s = np.zeros_like(img)
                img_s[z_s, y_s, x_s] = s_data

                img_s_fit = np.zeros_like(img)
                img_s_fit[z_s, y_s, x_s] = s_fit_data

                y_intens = img_s.max(axis=0)[int(y0_guess)]
                y_intens = y_intens[y_intens!=0]

                y_gauss = img_s_fit.max(axis=0)[int(y0_guess)]
                y_gauss = y_gauss[y_gauss!=0]

                ax[0].imshow(img.max(axis=0), vmax=img.max())
                ax[1].imshow(img_s_fit.max(axis=0), vmax=img.max())
                ax[2].scatter(range(len(y_intens)), y_intens)
                ax[2].plot(range(len(y_gauss)), y_gauss, c='r')

                l = x_s.min()
                b = y_s.min()

                r = x_s.max()
                t = y_s.max()

                ax[0].set_xlim((l-2, r+2))
                ax[0].set_ylim((t+2, b-2))

                ax[1].set_xlim((l-2, r+2))
                ax[1].set_ylim((t+2, b-2))

                plt.show()
                matplotlib.use('Agg')

    def store_metrics_good_spots(self, obj_id, s, fitted_coeffs_s,
                                 I_tot, I_foregr, gof_metrics,
                                 solution_found, B_fit):

        (z0_fit, y0_fit, x0_fit,
        sz_fit, sy_fit, sx_fit,
        A_fit) = fitted_coeffs_s

        min_z, min_y, min_x = self.ID_bbox_lower

        self._df_spotFIT.at[(obj_id, s), 'z_fit'] = z0_fit+min_z
        self._df_spotFIT.at[(obj_id, s), 'y_fit'] = y0_fit+min_y
        self._df_spotFIT.at[(obj_id, s), 'x_fit'] = x0_fit+min_x

        # self._df_spotFIT.at[(obj_id, s), 'AoB_fit'] = A_fit/B_fit

        self._df_spotFIT.at[(obj_id, s), 'sigma_z_fit'] = abs(sz_fit)
        self._df_spotFIT.at[(obj_id, s), 'sigma_y_fit'] = abs(sy_fit)
        self._df_spotFIT.at[(obj_id, s), 'sigma_x_fit'] = abs(sx_fit)
        self._df_spotFIT.at[(obj_id, s),
                            'sigma_yx_mean'] = (abs(sy_fit)+abs(sx_fit))/2

        _vol = 4/3*np.pi*abs(sz_fit)*abs(sy_fit)*abs(sx_fit)
        self._df_spotFIT.at[(obj_id, s), 'spotfit_vol_vox'] = _vol

        self._df_spotFIT.at[(obj_id, s), 'A_fit'] = A_fit
        self._df_spotFIT.at[(obj_id, s), 'B_fit'] = B_fit

        self._df_spotFIT.at[(obj_id, s), 'I_tot'] = I_tot
        self._df_spotFIT.at[(obj_id, s), 'I_foregr'] = I_foregr

        (reduced_chisq, p_chisq,
        ks, p_ks, RMSE, NRMSE, F_NRMSE) = gof_metrics

        self._df_spotFIT.at[(obj_id, s), 'reduced_chisq'] = reduced_chisq
        self._df_spotFIT.at[(obj_id, s), 'p_chisq'] = p_chisq

        self._df_spotFIT.at[(obj_id, s), 'KS_stat'] = ks
        self._df_spotFIT.at[(obj_id, s), 'p_KS'] = p_ks

        self._df_spotFIT.at[(obj_id, s), 'RMSE'] = RMSE
        self._df_spotFIT.at[(obj_id, s), 'NRMSE'] = NRMSE
        self._df_spotFIT.at[(obj_id, s), 'F_NRMSE'] = F_NRMSE

        QC_passed = int(NRMSE<self.QC_limit)
        self._df_spotFIT.at[(obj_id, s), 'QC_passed'] = QC_passed

        self._df_spotFIT.at[(obj_id, s), 'null_ks_test'] = int(p_ks > 0.05)
        self._df_spotFIT.at[(obj_id, s), 'null_chisq_test'] = int(p_chisq > 0.05)

        self._df_spotFIT.at[(obj_id, s), 'solution_found'] = int(solution_found)


if __name__ == '__main__':
    gmail_send('francescopadovani89@mail.com', 'test', 'Ciao bello')
