import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import cv2
import traceback
from collections import OrderedDict
from MyWidgets import Slider, Button, MyRadioButtons
from skimage.measure import label, regionprops
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from pyglet.canvas import Display
from skimage.color import gray2rgba, label2rgb
from skimage.exposure import equalize_adapthist
from skimage import img_as_float
from skimage.filters import (threshold_otsu, threshold_yen, threshold_isodata,
            threshold_li, threshold_mean, threshold_triangle, threshold_minimum)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import matplotlib.ticker as ticker

import core

# overwrite default matplotlib zoom functionalities
release_zoom = NavigationToolbar2.release_zoom
def my_release_zoom(self, event):
    release_zoom(self, event)
    # Disconnect zoom to rect after having used it once
    self.zoom()
    self.push_current()
    # self.release(event)
NavigationToolbar2.release_zoom = my_release_zoom

zoom = NavigationToolbar2.zoom
def my_zoom(self, *args):
    zoom(self, *args)
NavigationToolbar2.zoom = my_zoom

class tk_breakpoint:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title='Breakpoint', geometry="+800+400",
                 message='Breakpoint', button_1_text='Continue',
                 button_2_text='Abort', button_3_text='Delete breakpoint'):
        self.abort = False
        self.next_i = False
        self.del_breakpoint = False
        self.title = title
        self.geometry = geometry
        self.message = message
        self.button_1_text = button_1_text
        self.button_2_text = button_2_text
        self.button_3_text = button_3_text

    def pausehere(self):
        global root
        if not self.del_breakpoint:
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.title(self.title)
            root.geometry(self.geometry)
            tk.Label(root,
                     text=self.message,
                     font=(None, 11)).grid(row=0, column=0,
                                           columnspan=2, pady=4, padx=4)

            tk.Button(root,
                      text=self.button_1_text,
                      command=self.continue_button,
                      width=10,).grid(row=4,
                                      column=0,
                                      pady=8, padx=8)

            tk.Button(root,
                      text=self.button_2_text,
                      command=self.abort_button,
                      width=15).grid(row=4,
                                     column=1,
                                     pady=8, padx=8)
            tk.Button(root,
                      text=self.button_3_text,
                      command=self.delete_breakpoint,
                      width=20).grid(row=5,
                                     column=0,
                                     columnspan=2,
                                     pady=(0,8))

            root.mainloop()

    def continue_button(self):
        self.next_i=True
        root.quit()
        root.destroy()

    def delete_breakpoint(self):
        self.del_breakpoint=True
        root.quit()
        root.destroy()

    def abort_button(self):
        self.abort=True
        exit('Execution aborted by the user')
        root.quit()
        root.destroy()

class imshow_tk:
    def __init__(self, img, dots_coords=None, x_idx=1, axis=None,
                       additional_imgs=[], titles=[], fixed_vrange=False,
                       run=True):
        if type(additional_imgs) != list:
            additional_imgs = [additional_imgs]
        if img.ndim == 3:
            if img.shape[-1] > 4:
                img = img.max(axis=0)
                h, w = img.shape
            else:
                h, w, _ = img.shape
        elif img.ndim == 2:
            h, w = img.shape
        elif img.ndim != 2 and img.ndim != 3:
            raise TypeError(f'Invalid shape {img.shape} for image data. '
            'Only 2D or 3D images.')
        for i, im in enumerate(additional_imgs):
            if im.ndim == 3 and im.shape[-1] > 4:
                additional_imgs[i] = im.max(axis=0)
            elif im.ndim != 2 and im.ndim != 3:
                raise TypeError(f'Invalid shape {im.shape} for image data. '
                'Only 2D or 3D images.')
        n_imgs = len(additional_imgs)+1
        if w/h > 1:
            fig, ax = plt.subplots(n_imgs, 1, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(1, n_imgs, sharex=True, sharey=True)
        if n_imgs == 1:
            ax = [ax]
        self.ax0img = ax[0].imshow(img)
        if dots_coords is not None:
            ax[0].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
        if axis:
            ax[0].axis('off')
        if fixed_vrange:
            vmin, vmax = img.min(), img.max()
        else:
            vmin, vmax = None, None
        self.additional_aximgs = []
        for i, img_i in enumerate(additional_imgs):
            axi_img = ax[i+1].imshow(img_i, vmin=vmin, vmax=vmax)
            self.additional_aximgs.append(axi_img)
            if dots_coords is not None:
                ax[i+1].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
            if axis:
                ax[i+1].axis('off')
        for title, a in zip(titles, ax):
            a.set_title(title)
        sub_win = embed_tk('Imshow embedded in tk', [800,600,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        self.fig = fig
        self.ax = ax
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        if run:
            sub_win.root.mainloop()

    def _close(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

class embed_tk:
    """Example:
    -----------
    img = np.ones((600,600))
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(img)

    sub_win = embed_tk('Embeddding in tk', [1024,768,300,100], fig)

    def on_key_event(event):
        print('you pressed %s' % event.key)

    sub_win.canvas.mpl_connect('key_press_event', on_key_event)

    sub_win.root.mainloop()
    """
    def __init__(self, win_title, geom, fig):
        root = tk.Tk()
        root.wm_title(win_title)
        root.geometry("{}x{}+{}+{}".format(*geom)) # WidthxHeight+Left+Top
        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar
        self.root = root

class auto_select_slice:
    def __init__(self, auto_focus=True, prompt_use_for_all=False):
        self.auto_focus = auto_focus
        self.prompt_use_for_all = prompt_use_for_all
        self.use_for_all = False

    def run(self, frame_V, segm_slice=0, segm_npy=None, IDs=None):
        if self.auto_focus:
            auto_slice = self.auto_slice(frame_V)
        else:
            auto_slice = 0
        self.segm_slice = segm_slice
        self.slice = auto_slice
        self.abort = True
        self.data = frame_V
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.20)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        ok_width = 0.13
        ok_left = 0.5 - (ok_width/2)
        (self.ax).imshow(frame_V[auto_slice])
        if segm_npy is not None:
            self.contours = self.find_contours(segm_npy, IDs, group=True)
            for cont in self.contours:
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                (self.ax).plot(x, y, c='r')
        (self.ax).axis('off')
        (self.ax).set_title('Select slice for amount calculation\n\n'
                    f'Slice used for segmentation: {segm_slice}\n'
                    f'Best focus determined by algorithm: slice {auto_slice}')
        """Embed plt window into a tkinter window"""
        sub_win = embed_tk('Mother-bud zoom', [1024,768,400,150], self.fig)
        self.ax_sl = self.fig.add_subplot(
                                position=[sl_left, 0.12, sl_width, 0.04],
                                facecolor='0.1')
        self.sl = Slider(self.ax_sl, 'Slice', -1, len(frame_V),
                                canvas=sub_win.canvas,
                                valinit=auto_slice,
                                valstep=1,
                                color='0.2',
                                init_val_line_color='0.3',
                                valfmt='%1.0f')
        (self.sl).on_changed(self.update_slice)
        self.ax_ok = self.fig.add_subplot(
                                position=[ok_left, 0.05, ok_width, 0.05],
                                facecolor='0.1')
        self.ok_b = Button(self.ax_ok, 'Happy with that', canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.25',
                                presscolor='0.35')
        (self.ok_b).on_clicked(self.ok)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        (sub_win.canvas).mpl_connect('key_press_event', self.set_slvalue)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def find_contours(self, label_img, cells_ids, group=False, concat=False,
                      return_hull=False):
        contours = []
        for id in cells_ids:
            label_only_cells_ids_img = np.zeros_like(label_img)
            label_only_cells_ids_img[label_img == id] = id
            uint8_img = (label_only_cells_ids_img > 0).astype(np.uint8)
            cont, hierarchy = cv2.findContours(uint8_img,cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
            cnt = cont[0]
            if return_hull:
                hull = cv2.convexHull(cnt,returnPoints = True)
                contours.append(hull)
            else:
                contours.append(cnt)
        if concat:
            all_contours = np.zeros((0,2), dtype=int)
            for contour in contours:
                contours_2D_yx = np.fliplr(np.reshape(contour, (contour.shape[0],2)))
                all_contours = np.concatenate((all_contours, contours_2D_yx))
        elif group:
            # Return a list of n arrays for n objects. Each array has i rows of
            # [y,x] coords for each ith pixel in the nth object's contour
            all_contours = [[] for _ in range(len(cells_ids))]
            for c in contours:
                c2Dyx = np.fliplr(np.reshape(c, (c.shape[0],2)))
                for y,x in c2Dyx:
                    ID = label_img[y, x]
                    idx = list(cells_ids).index(ID)
                    all_contours[idx].append([y,x])
            all_contours = [np.asarray(li) for li in all_contours]
            IDs = [label_img[c[0,0],c[0,1]] for c in all_contours]
        else:
            all_contours = [np.fliplr(np.reshape(contour,
                            (contour.shape[0],2))) for contour in contours]
        return all_contours

    def auto_slice(self, frame_V):
        # https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
        means = []
        for i, img in enumerate(frame_V):
            edge = sobel(img)
            means.append(np.mean(edge))
        slice = means.index(max(means))
        print('Best slice = {}'.format(slice))
        return slice

    def set_slvalue(self, event):
        if event.key == 'left':
            self.sl.set_val(self.sl.val - 1)
        if event.key == 'right':
            self.sl.set_val(self.sl.val + 1)
        if event.key == 'enter':
            self.ok(None)

    def update_slice(self, val):
        self.slice = int(val)
        img = self.data[int(val)]
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def ok(self, event):
        use_for_all = False
        if self.prompt_use_for_all:
            use_for_all = tk.messagebox.askyesno('Use same slice for all',
                          f'Do you want to use slice {self.slice} for all positions?')
        if use_for_all:
            self.use_for_all = use_for_all
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

    def abort_exec(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        exit('Execution aborted by the user')

class inspect_spotFIT_app:
    def __init__(self, fig_title):
        self.fig_title = fig_title
        matplotlib.use('TkAgg')

    def run(self, V_spots, segm_npy_3D, IDs, df_spots, channel_name,
            spotsize_limits_pxl, spots_3D_labs, ID_bboxs_lower,
            ID_3Dslices, IDs_with_spots, dfs_intersect, sharp_V_spots=None,
            which_ax1_data='sigma_yx_mean'):
        print('\n\n\n\n\n')
        self._df = df_spots.copy()
        self._df_drop = None
        self.df_spots = df_spots
        self.next = False

        self.ID_3Dslices = ID_3Dslices
        self.ID_bboxs_lower = ID_bboxs_lower
        self.IDs_with_spots = IDs_with_spots

        self.spots_3D_labs = spots_3D_labs
        self.IDs = IDs

        self.V_spots = V_spots
        self.sharp_V_spots = sharp_V_spots

        self.is_z_proj = True

        self.is_mouse_down = False
        self.selected_cursor = None
        self.is_auto_filter = False
        self.which_cursor = None
        self.inspect_fit = False

        self.dfs_intersect = dfs_intersect

        self.which_ax1_data = which_ax1_data

        # self._min, self._max = spotsize_limits_pxl

        fig = plt.figure()
        ax = [None]*3
        ax[0] = fig.add_subplot(121)
        ax[1] = fig.add_subplot(222)
        ax[2] = fig.add_subplot(224)

        fig.patch.set_facecolor('0.05')
        self.fig = fig
        plt.subplots_adjust(bottom=0.2)
        ch_img = V_spots.max(axis=0)
        self.ax = ax
        if not np.all(segm_npy_3D):
            segm_npy_2D = segm_npy_3D.max(axis=0)
            contours = auto_select_slice().find_contours(segm_npy_2D,
                                                         IDs, group=True)
        else:
            contours = []
        self.ax0_img_data = ax[0].imshow(ch_img)
        self.spots_plot, = ax[0].plot(df_spots['x'].to_numpy(),
                                      df_spots['y'].to_numpy(), 'r.')
        self.dropped_spots_plot, = ax[0].plot([], [], 'kx')
        ax[0].set_title(f'{channel_name} max z-projection')
        for cont in contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax[0].plot(x, y, c='r', alpha=0.5, lw=1)

        ax[0].axis('off')

        s_yx_min = df_spots[self.which_ax1_data].min()
        s_yx_max = df_spots[self.which_ax1_data].max()
        self.s_step = (s_yx_max-s_yx_min)/len(self.df_spots)
        sns.boxplot(x=df_spots[self.which_ax1_data], ax=ax[1],
                    color='cadetblue')
        sns.stripplot(x=df_spots[self.which_ax1_data], ax=ax[1],
                      color=(199/255, 151/255, 68/255, 0.8))
        ax[1].set_facecolor('0.1')
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1].spines['bottom'].set_color('0.9')
        ax[1].spines['top'].set_color('0.9')
        ax[1].spines['right'].set_color('0.9')
        ax[1].spines['left'].set_color('0.9')
        for art in ax[1].artists:
            plt.setp(art, edgecolor='0.7')
        for line in ax[1].lines:
            line.set_color('0.7')
        # ax[1].set_xscale('log')
        # ax[1].xaxis.set_major_formatter(
        #          ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # ax[1].set_xticks(np.round(np.linspace(s_yx_min, s_yx_max, 5), 1))

        self.axvl_s_yx_min = ax[1].axvline(s_yx_min-self.s_step,
                                           color='firebrick', ls='--')
        self.axvl_s_yx_max = ax[1].axvline(s_yx_max+self.s_step,
                                           color='firebrick', ls='--')
        self.s_yx_min = s_yx_min
        self.s_yx_max = s_yx_max

        A_min = df_spots['A_fit'].min()
        A_max = df_spots['A_fit'].max()
        self.A_step = (A_max-A_min)/len(self.df_spots)
        sns.boxplot(x=df_spots['A_fit'], ax=ax[2], color='0.4')
        sns.stripplot(x=df_spots['A_fit'], ax=ax[2],
                      color=(199/255, 151/255, 68/255, 0.8))
        ax[2].set_facecolor('0.1')
        ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[2].spines['bottom'].set_color('0.9')
        ax[2].spines['top'].set_color('0.9')
        ax[2].spines['right'].set_color('0.9')
        ax[2].spines['left'].set_color('0.9')
        for art in ax[2].artists:
            plt.setp(art, edgecolor='0.7')
        for line in ax[2].lines:
            line.set_color('0.7')
        # ax[2].set_xscale('log')
        # ax[2].xaxis.set_major_formatter(
        #             ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # ax[2].set_xticks(np.round(np.linspace(A_min, A_max, 5), 1))


        self.axvl_A_min = ax[2].axvline(A_min-self.A_step,
                                        color='firebrick', ls='--')
        self.axvl_A_max = ax[2].axvline(A_max+self.A_step,
                                        color='firebrick', ls='--')
        self.A_min = A_min
        self.A_max = A_max

        # Cursors value text
        self.axvl_s_yx_min_txt = ax[1].text(0, -0.45, f'', c='r')
        self.axvl_s_yx_max_txt = ax[1].text(1, -0.45, f'', c='r', ha='right')

        self.A_min_txt = ax[2].text(0, -0.45, f'', c='r')
        self.A_max_txt = ax[2].text(1, -0.45, f'', c='r', ha='right')

        # Widgets colors
        axcolor = '0.15'
        slider_color = '0.2'
        hover_color = '0.25'
        presscolor = '0.35'
        button_true_color = '0.4'
        self.button_true_color = button_true_color
        self.axcolor = axcolor
        self.hover_color =  hover_color

        # Widgets axis
        self.ax_next = plt.axes([0.1, 0.65, 0.25, 0.2])
        self.ax_continue = plt.axes([0.1, 0.66, 0.25, 0.2])
        self.ax_abort = plt.axes([0.1, 0.67, 0.25, 0.2])
        self.ax_z_slice_slider = plt.axes([0.1, 0.78, 0.25, 0.2])
        self.ax_z_proj_button = plt.axes([0.1, 0.78, 0.25, 0.39])
        self.ax_radiob_hide_dropped = plt.axes([0.1, 0.78, 0.37, 0.48])
        self.ax_auto_filter_b = plt.axes([0.1, 0.9587, 0.37, 0.48])
        # self.ax_min_limit_slider = plt.axes([0.1, 0.69, 0.25, 0.2])
        # self.ax_max_limit_slider = plt.axes([0.1, 0.78, 0.37, 0.39])
        # self.ax_min_A_limit_slider = plt.axes([0.1, 0.67, 0.258, 0.2])
        # self.ax_max_A_limit_slider = plt.axes([0.1, 0.78, 0.396, 0.39])
        if sharp_V_spots is not None:
            self.ax_radiob_sharp = plt.axes([0.1, 0.97, 0.34, 0.2])

        # Widgets
        next_b = Button(self.ax_next, 'Skip to next pos./frame',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
        continue_b = Button(self.ax_continue, 'Continue',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
        abort_b = Button(self.ax_abort, 'Abort',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

        z_proj_b = Button(self.ax_z_proj_button, 'Max',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

        self.radiob_hide_dropped = MyRadioButtons(self.ax_radiob_hide_dropped,
                          ('Show all', 'Hide dropped', 'Hide all'),
                          active = 0,
                          activecolor = button_true_color,
                          orientation = 'vertical',
                          size = 59,
                          circ_p_color = button_true_color)

        if sharp_V_spots is not None:
            self.radiob_sharp = MyRadioButtons(self.ax_radiob_sharp,
                              ('Original', 'Sharp'),
                              active = 0,
                              activecolor = button_true_color,
                              orientation = 'vertical',
                              size = 59,
                              circ_p_color = button_true_color)

        self.z_slice_slider = Slider(self.ax_z_slice_slider,
                'z-slice', 0, V_spots.shape[0],
                valinit=V_spots.shape[0]/2,
                valstep=1,
                orientation='horizontal',
                color=slider_color,
                init_val_line_color=hover_color,
                valfmt='%1.0f')

        self.auto_filter_b = Button(self.ax_auto_filter_b, 'Auto-filter',
                                color=axcolor, hovercolor=hover_color,
                                presscolor=presscolor)

        next_b.on_clicked(self._next)
        continue_b.on_clicked(self._continue)
        abort_b.on_clicked(self._abort)
        z_proj_b.on_clicked(self._draw_V_spots_max_proj)
        self.radiob_hide_dropped.on_clicked(self.show_hide_spots)
        self.auto_filter_b.on_clicked(self.auto_filter_cb)
        self.z_slice_slider.on_changed(self.view_z_slice)
        if sharp_V_spots is not None:
            self.radiob_sharp.on_clicked(self.radiob_sharp_cb)

        fig.canvas.mpl_connect('button_press_event', self.mouse_down)
        fig.canvas.mpl_connect('button_release_event', self.mouse_up)
        fig.canvas.mpl_connect('resize_event', self.resize_widgets)
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_motion)
        fig.canvas.mpl_connect('key_press_event', self.key_down)
        fig.canvas.mpl_connect('key_release_event', self.key_up)
        try:
            win_size()
        except:
            pass

        fig.suptitle(f'{self.fig_title}.\n'
            '(Right-click on a spot to print some of its metrics to the console,\n'
            ' "v" + right-click to also visualize the gaussian fit.)')

        self.update_plot(None)

        plt.show()
        matplotlib.use('Agg')

    def show_hide_spots(self, label):
        if label == 'Show all':
            self.spots_plot.set_visible(True)
            self.dropped_spots_plot.set_visible(True)
        elif label == 'Hide dropped':
            self.spots_plot.set_visible(True)
            self.dropped_spots_plot.set_visible(False)
        else:
            self.spots_plot.set_visible(False)
            self.dropped_spots_plot.set_visible(False)
        self.fig.canvas.draw_idle()

    def auto_filter_cb(self, event):
        self.is_auto_filter = True
        self.update_plot(event)
        self.is_auto_filter = False

    def radiob_sharp_cb(self, label):
        if label == 'Original':
            if self.is_z_proj:
                img = self.V_spots.max(axis=0)
            else:
                img = self.V_spots[int(self.z_slice_slider.val)]
        else:
            if self.is_z_proj:
                img = self.sharp_V_spots.max(axis=0)
            else:
                img = self.sharp_V_spots[int(self.z_slice_slider.val)]
        self.ax0_img_data.set_data(img)
        self.ax0_img_data.set_clim(vmin=img.min(), vmax=img.max())
        self.fig.canvas.draw_idle()

    def _draw_V_spots_max_proj(self, event):
        self.is_z_proj = True
        self.update_plot(event)

    def view_z_slice(self, event):
        self.is_z_proj = False
        self.update_plot(event)

    def filter_by_A(self):
        self.dropped_spots_plot.set_data([[],[]])

        max_s_1, _  = self.df_spots[self.which_ax1_data].nlargest(2)
        min_s_1, _ = self.df_spots[self.which_ax1_data].nsmallest(2)
        s_step = self.s_step
        self.axvl_s_yx_max.set_xdata([max_s_1+s_step, max_s_1+s_step])
        self.axvl_s_yx_min.set_xdata([min_s_1-s_step, min_s_1-s_step])
        self.ax[1].set_xlim(min_s_1-2*s_step, max_s_1+2*s_step)


        self.A_min = self.axvl_A_min.get_xdata()[0]
        self.A_max = self.axvl_A_max.get_xdata()[0]
        _min, _ = self.ax[2].get_xlim()
        self.ax[2].set_xlim(_min, self.A_max+self.A_step)
        _df = self.df_spots.copy()

        # Filter by min
        _df_drop_min = _df[_df['A_fit'] < self.A_min].copy()
        _df = _df[_df['A_fit'] >= self.A_min]

        # Filter by max
        _df_drop_max = _df[_df['A_fit'] > self.A_max].copy()
        _df = _df[_df['A_fit'] <= self.A_max]
        _df_drop = pd.concat([_df_drop_min, _df_drop_max])
        self._df = _df
        self._df_drop = _df_drop


    def filter_by_size(self):
        self.dropped_spots_plot.set_data([[],[]])

        max_A_1, _  = self.df_spots['A_fit'].nlargest(2)
        min_A_1, _ = self.df_spots['A_fit'].nsmallest(2)
        A_step = self.A_step
        self.axvl_A_max.set_xdata([max_A_1+A_step, max_A_1+A_step])
        self.axvl_A_min.set_xdata([min_A_1-A_step, min_A_1-A_step])
        self.ax[2].set_xlim(min_A_1-2*A_step, max_A_1+2*A_step)


        self.s_yx_min = self.axvl_s_yx_min.get_xdata()[0]
        self.s_yx_max = self.axvl_s_yx_max.get_xdata()[0]
        _min, _ = self.ax[1].get_xlim()
        self.ax[1].set_xlim(_min, self.s_yx_max+2*self.s_step)
        _df = self.df_spots.copy()

        # Filter by min
        _df_drop_min = _df[_df[self.which_ax1_data] < self.s_yx_min].copy()
        _df = _df[_df[self.which_ax1_data] >= self.s_yx_min]

        # Filter by max
        _df_drop_max = _df[_df[self.which_ax1_data] > self.s_yx_max].copy()
        _df = _df[_df[self.which_ax1_data] <= self.s_yx_max]

        # Concat drop
        _df_drop = pd.concat([_df_drop_min, _df_drop_max])
        self._df = _df
        self._df_drop = _df_drop


    def update_plot(self, event):
        if self.is_auto_filter:
            self._df = self.df_spots[self.df_spots['QC_passed'] == 1]
            self._df_drop = self.df_spots[self.df_spots['QC_passed'] == 0]
            drop_color = 'k'

        if self.which_cursor is not None:
            if self.which_cursor == 's_max':
                if not self.is_auto_filter:
                    self.filter_by_size()
                    drop_color = 'k'
                # max_A_1, max_A_2  = self._df['A_fit'].nlargest(2)
                # max_val_A = max_A_2 + abs(max_A_1-max_A_2)/2
                # self.axvl_A_max.set_xdata([max_val_A, max_val_A])
            elif self.which_cursor == 's_min':
                if not self.is_auto_filter:
                    self.filter_by_size()
                    drop_color = 'k'
                # min_A_1, min_A_2 = self._df['A_fit'].nsmallest(2)
                # min_val_A = min_A_1 + abs(min_A_2-min_A_1)/2
                # self.axvl_A_min.set_xdata([min_val_A, min_val_A])
            elif self.which_cursor == 'A_max':
                if not self.is_auto_filter:
                    self.filter_by_A()
                    drop_color = 'm'
                # max_s_1, max_s_2  = self._df[self.which_ax1_data].nlargest(2)
                # max_val_s = max_s_2 + abs(max_s_1-max_s_2)/2
                # self.axvl_s_yx_max.set_xdata([max_val_s, max_val_s])
            elif self.which_cursor == 'A_min':
                if not self.is_auto_filter:
                    self.filter_by_A()
                    drop_color = 'm'
                # min_s_1, min_s_2 = self._df[self.which_ax1_data].nsmallest(2)
                # min_val_s = min_s_1 + abs(min_s_2-min_s_1)/2
                # self.axvl_s_yx_min.set_xdata([min_val_s, min_val_s])

        # self.axvl_s_yx_min.set_xdata()
        # self.axvl_s_yx_max.set_xdata()


        if self.is_z_proj:
            xy_spots_data = self._df[['x', 'y']].to_numpy().T
            if self._df_drop is not None:
                droppedxy_spots_data = self._df_drop[['x', 'y']].to_numpy().T
                self.dropped_spots_plot.set_data(droppedxy_spots_data)
                self.dropped_spots_plot.set_color(drop_color)
            if self.radiob_sharp.value_selected == 'Original':
                img = self.V_spots.max(axis=0)
            else:
                img = self.sharp_V_spots.max(axis=0)
        else:
            _z = int(self.z_slice_slider.val)
            _df = self._df[self._df['z']==_z]
            xy_spots_data = _df[['x', 'y']].to_numpy().T
            if self._df_drop is not None:
                _df_drop = self._df_drop[self._df_drop['z']==_z]
                droppedxy_spots_data = _df_drop[['x', 'y']].to_numpy().T
                self.dropped_spots_plot.set_data(droppedxy_spots_data)
                self.dropped_spots_plot.set_color(drop_color)
            if self.radiob_sharp.value_selected == 'Original':
                img = self.V_spots[int(self.z_slice_slider.val)]
            else:
                img = self.sharp_V_spots[int(self.z_slice_slider.val)]
        self.spots_plot.set_data(xy_spots_data)
        self.show_hide_spots(self.radiob_hide_dropped.value_selected)
        self.ax0_img_data.set_data(img)
        self.ax0_img_data.set_clim(vmin=img.min(), vmax=img.max())
        size_curs_left = self.axvl_s_yx_min.get_xdata()[0]
        size_curs_right = self.axvl_s_yx_max.get_xdata()[0]
        area_curs_left = self.axvl_A_min.get_xdata()[0]
        area_curs_right = self.axvl_A_max.get_xdata()[0]

        self.axvl_s_yx_min_txt.set_text(f'  {size_curs_left:.3f}')
        self.axvl_s_yx_max_txt.set_text(f'{size_curs_right:.3f}  ')
        self.A_min_txt.set_text(f'  {area_curs_left:.3f}')
        self.A_max_txt.set_text(f'{area_curs_right:.3f}  ')

        self.axvl_s_yx_min_txt.set_x(size_curs_left)
        self.axvl_s_yx_max_txt.set_x(size_curs_right)
        self.A_min_txt.set_x(area_curs_left)
        self.A_max_txt.set_x(area_curs_right)

        self.fig.canvas.draw_idle()
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.write("\x1b[1A\x1b[2K")
        print('------------------------------------')
        print(f'Size cursor left value = {size_curs_left:.3f}')
        print(f'Size cursor right value = {size_curs_right:.3f}')
        print(f'Area cursor left value = {area_curs_left:.3f}')
        print(f'Area cursor right value = {area_curs_right:.3f}')
        print('------------------------------------')


    def mouse_motion(self, event):
        if event.inaxes == self.ax[1]:
            x = event.x
            s_yx_min = self.axvl_s_yx_min.get_xdata()[0]
            s_yx_max = self.axvl_s_yx_max.get_xdata()[0]
            x_min_curs, _ = self.ax[1].transData.transform((s_yx_min, 0))
            x_max_curs, _ = self.ax[1].transData.transform((s_yx_max, 0))
            dist = [abs(x-x_min_curs), abs(x-x_max_curs)]
            min_dist = min(dist)
            if min_dist < 10 and self.selected_cursor is None:
                idx_min_dist = dist.index(min_dist)
                if idx_min_dist == 0:
                    self.axvl_s_yx_min.set_linestyle('-')
                    self.axvl_s_yx_min.set_linewidth(2.5)
                    self.selected_cursor = self.axvl_s_yx_min
                    self.which_cursor = 's_min'
                else:
                    self.axvl_s_yx_max.set_linestyle('-')
                    self.axvl_s_yx_max.set_linewidth(2.5)
                    self.selected_cursor = self.axvl_s_yx_max
                    self.which_cursor = 's_max'
                self.curs_ax = self.ax[1]
                self.fig.canvas.draw_idle()
            elif not self.is_mouse_down:
                if min_dist > 10:
                    self.selected_cursor = None
                    idx_min_dist = dist.index(min_dist)
                    if idx_min_dist == 0:
                        self.axvl_s_yx_min.set_linestyle('--')
                        self.axvl_s_yx_min.set_linewidth(1.5)
                    else:
                        self.axvl_s_yx_max.set_linestyle('--')
                        self.axvl_s_yx_max.set_linewidth(1.5)
                    self.fig.canvas.draw_idle()
        elif event.inaxes == self.ax[2]:
            x = event.x
            s_yx_min = self.axvl_A_min.get_xdata()[0]
            s_yx_max = self.axvl_A_max.get_xdata()[0]
            x_min_curs, _ = self.ax[2].transData.transform((s_yx_min, 0))
            x_max_curs, _ = self.ax[2].transData.transform((s_yx_max, 0))
            dist = [abs(x-x_min_curs), abs(x-x_max_curs)]
            min_dist = min(dist)
            if min_dist < 10 and self.selected_cursor is None:
                idx_min_dist = dist.index(min_dist)
                if idx_min_dist == 0:
                    self.axvl_A_min.set_linestyle('-')
                    self.axvl_A_min.set_linewidth(2.5)
                    self.selected_cursor = self.axvl_A_min
                    self.which_cursor = 'A_min'
                else:
                    self.axvl_A_max.set_linestyle('-')
                    self.axvl_A_max.set_linewidth(2.5)
                    self.selected_cursor = self.axvl_A_max
                    self.which_cursor = 'A_max'
                self.curs_ax = self.ax[2]
                self.fig.canvas.draw_idle()
            elif not self.is_mouse_down:
                if min_dist > 10:
                    self.selected_cursor = None
                    idx_min_dist = dist.index(min_dist)
                    if idx_min_dist == 0:
                        self.axvl_A_min.set_linestyle('--')
                        self.axvl_A_min.set_linewidth(1.5)
                    else:
                        self.axvl_A_max.set_linestyle('--')
                        self.axvl_A_max.set_linewidth(1.5)
                    self.fig.canvas.draw_idle()
        if self.selected_cursor is not None and self.is_mouse_down:
            x = event.x
            x_data, _ = self.curs_ax.transData.inverted().transform((x, 0))
            self.selected_cursor.set_xdata([x_data, x_data])
            self.fig.canvas.draw_idle()

    def mouse_up(self, event):
        if event.button == 3:
            self.is_mouse_down = False
            if self.selected_cursor is not None:
                x = event.x
                x_data, _ = self.curs_ax.transData.inverted().transform((x, 0))
                self.selected_cursor.set_xdata([x_data, x_data])
                self.selected_cursor.set_linestyle('--')
                self.selected_cursor.set_linewidth(1.5)
                self.update_plot(None)
            self.selected_cursor = None
            self.which_cursor = None


    def mouse_down(self, event):
        if event.inaxes == self.ax[1] and event.button == 3:
            self.is_mouse_down = True
        elif event.inaxes == self.ax[2] and event.button == 3:
            self.is_mouse_down = True
        elif event.inaxes == self.ax[0] and event.button == 3:
            x = event.xdata
            y = event.ydata
            df = self.df_spots
            if df is not None:
                dist = ((df['y']-y)**2 + (df['x']-x)**2).apply(np.sqrt)
                idx_min_dist = dist.idxmin()
                min_dist = dist.min()
                if min_dist < 3:
                    pd.set_option('display.max_columns', 20)
                    print('')
                    print('===================================================')
                    print(df.loc[[idx_min_dist]]
                                             [['|norm|_spot',
                                               'effsize_glass_s',
                                               self.which_ax1_data,
                                               'sigma_y_fit',
                                               'sigma_x_fit',
                                               'sigma_z_fit',
                                               'F_NRMSE',
                                               'A_fit',
                                               'B_fit',
                                               'QC_passed',
                                               'spot_surf_5p',
                                               'spot_surf_mean',
                                               'spot_surf_50p',
                                               'spotsize_limit',
                                               'spot_surf_std',
                                               'spot_B_min']])
                    print('===================================================')
                    print('')
                    ID = idx_min_dist[0]
                    idx = self.IDs.index(ID)
                    if self.inspect_fit:
                        self.inspect_fit = False
                        try:
                            inspect_gaussian_fit(
                                self.V_spots, df.loc[[idx_min_dist]],
                                self.spots_3D_labs, self.ID_3Dslices,
                                self.ID_bboxs_lower, self.IDs_with_spots,
                                self.dfs_intersect, self.df_spots)
                        except:
                            traceback.print_exc()
                            # import pdb; pdb.set_trace()

    def _next(self, event):
        plt.close()
        self.next = True

    def _continue(self, event):
        plt.close()

    def _abort(self, event):
        plt.close()
        exit('Execution aborted.')

    def key_down(self, event):
        if event.key == 'v':
            self.inspect_fit = True

    def key_up(self, event):
        self.inspect_fit = False

    def resize_widgets(self, event):
        # [left, bottom, width, height]
        H = 0.03
        W = 0.1
        spF = 0.01
        ax0 = self.ax[0]
        ax0_l, ax0_b, ax0_r, ax0_t = ax0.get_position().get_points().flatten()
        ax0_w = ax0_r-ax0_l
        b = ax0_b-spF/2-H
        self.ax_z_slice_slider.set_position([ax0_l, b, ax0_w, H])
        L = ax0_l+ax0_w+2*spF
        self.ax_z_proj_button.set_position([L, b, W/3, H])
        # b -= H+spF
        # self.ax_min_limit_slider.set_position([ax0_l, b, ax0_w, H])
        # b -= H+spF
        # self.ax_max_limit_slider.set_position([ax0_l, b, ax0_w, H])


        # b22 -= H+spF
        # self.ax_min_A_limit_slider.set_position([ax2_l, b22, ax2_w, H])
        # b22 -= H+spF
        # self.ax_max_A_limit_slider.set_position([ax2_l, b22, ax2_w, H])

        l1 = ax0_l
        b1 = b-H-spF
        self.ax_continue.set_position([l1, b1, W*2/3, H])
        l2 = l1+W*2/3+spF
        self.ax_next.set_position([l2, b1, W, H])
        b2 = b1-H-spF
        self.ax_abort.set_position([l1, b2, W*2/3, H])
        l3 = l2+W+spF
        self.ax_radiob_hide_dropped.set_position([l3, b2, W*0.9, H*2+spF])
        l4 = l3+(W*0.9)+(spF/2)
        W1 = (ax0_r-(l4))
        try:
            self.ax_radiob_sharp.set_position([l4, b2, W1, H*2+spF])
        except AttributeError:
            pass

        ax2 = self.ax[2]
        ax2_l, ax2_b, ax2_r, ax2_t = ax2.get_position().get_points().flatten()
        ax2_w = ax2_r-ax2_l
        ax2_c = ax2_l + (ax2_w/2)
        w5 = W*2/3
        l5 = ax2_c-(w5/2)
        b5 = b2
        self.ax_auto_filter_b.set_position([l5, b5, w5, H])

class inspect_effect_size_app:
    def __init__(self, fig_title):
        self.fig_title = fig_title
        matplotlib.use('TkAgg')

    def run(self, load_ref_ch, V_spots, segm_npy_3D, IDs, local_max_coords,
            channel_name, V_ref, ref_mask, ref_channel_name, z_resol_limit_pxl,
            filtered_zyx_coords, gop_bounds=None, prev_df_spots=None,
            df_spots=None, gop_how=None, which_effsize=None,
            sharp_V_spots=None):
        self.df_spots = df_spots
        self.next = False
        self.gop_how = gop_how
        self.which_effsize = which_effsize
        self.load_ref_ch = load_ref_ch

        self.prev_df_spots = prev_df_spots

        self.local_max_coords = local_max_coords
        self.dropped_zyx_coords = self.get_dropped_zyx_coords(local_max_coords,
                                                           filtered_zyx_coords)

        self.z_resol_limit_pxl = z_resol_limit_pxl

        self.do_filter_zbounds = False

        self.V_spots = V_spots
        self.V_ref = V_ref
        self.sharp_V_spots = sharp_V_spots
        self.ref_channel_name = ref_channel_name
        self.ref_mask = ref_mask

        self.is_z_proj = True

        num_plots = 2 if load_ref_ch else 1
        fig, ax = plt.subplots(1, num_plots, figsize=(19, 9), dpi=100,
                               sharex=True, sharey=True)
        self.fig = fig
        plt.subplots_adjust(bottom=0.2)
        ch_img = V_spots.max(axis=0)
        if not load_ref_ch:
            ax = [ax]
        self.ax = ax
        if not np.all(segm_npy_3D):
            segm_npy_2D = segm_npy_3D.max(axis=0)
            contours = auto_select_slice().find_contours(segm_npy_2D,
                                                         IDs, group=True)
        else:
            contours = []
        self.ax0_img_data = ax[0].imshow(ch_img)
        self.spots_plot, = ax[0].plot(local_max_coords[:,2],
                                      local_max_coords[:,1], 'r.')
        self.dropped_spots_plot, = ax[0].plot(self.dropped_zyx_coords[:,2],
                                              self.dropped_zyx_coords[:,1],
                                              'kx')
        ax[0].set_title(f'{channel_name} max z-projection')
        for cont in contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax[0].plot(x, y, c='r', alpha=0.5, lw=1)
            if load_ref_ch:
                ax[1].plot(x, y, c='r', alpha=0.5, lw=1)

        if load_ref_ch:
            ref_img = equalize_adapthist(V_ref.max(axis=0)/V_ref.max())
            img_RGB = plt.cm.gray(ref_img)
            ref_mask_img = ref_mask.max(axis=0).astype(float)
            ref_mask_RGB = gray2rgba(ref_mask_img)*np.array([1,1,0,1])
            alpha = 0.2
            overlay = (img_RGB*(1.0 - alpha) + ref_mask_RGB*alpha)*1
            overlay = np.clip(overlay, 0, 1)
            self.overlay = overlay
            ax[1].imshow(overlay)
            ax[1].set_title(f'{ref_channel_name} max z-projection')
        for a in ax:
            a.axis('off')

        # Widgets colors
        axcolor = '0.1'
        slider_color = '0.2'
        hover_color = '0.25'
        presscolor = '0.35'
        button_true_color = '0.4'
        self.button_true_color = button_true_color
        self.axcolor = axcolor
        self.hover_color =  hover_color

        # Widgets axis
        self.ax_next = plt.axes([0.1, 0.65, 0.25, 0.2])
        self.ax_continue = plt.axes([0.1, 0.66, 0.25, 0.2])
        self.ax_abort = plt.axes([0.1, 0.67, 0.25, 0.2])
        self.ax_z_slice_slider = plt.axes([0.1, 0.78, 0.25, 0.2])
        self.ax_z_proj_button = plt.axes([0.1, 0.78, 0.25, 0.39])
        self.ax_filter_zbounds_button = plt.axes([0.1, 0.78, 0.37, 0.39])
        self.ax_radiob_hide_dropped = plt.axes([0.1, 0.78, 0.37, 0.48])
        if df_spots is not None:
            self.ax_gop_limit_slider = plt.axes([0.1, 0.69, 0.25, 0.2])
        if load_ref_ch:
            self.ax_radiob_refch = plt.axes([0.1, 0.78, 0.34, 0.2])
        if sharp_V_spots is not None:
            self.ax_radiob_sharp = plt.axes([0.1, 0.97, 0.34, 0.2])

        # Widgets
        next_b = Button(self.ax_next, 'Skip to next pos./frame',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
        continue_b = Button(self.ax_continue, 'Continue',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
        abort_b = Button(self.ax_abort, 'Abort',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

        z_proj_b = Button(self.ax_z_proj_button, 'Max',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

        self.filter_zbounds_b = Button(self.ax_filter_zbounds_button,
               'Filter z-bounds',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

        self.radiob_hide_dropped = MyRadioButtons(self.ax_radiob_hide_dropped,
                          ('Show all', 'Hide dropped'),
                          active = 0,
                          activecolor = button_true_color,
                          orientation = 'vertical',
                          size = 59,
                          circ_p_color = button_true_color)

        if sharp_V_spots is not None:
            self.radiob_sharp = MyRadioButtons(self.ax_radiob_sharp,
                              ('Original', 'Sharp'),
                              active = 0,
                              activecolor = button_true_color,
                              orientation = 'vertical',
                              size = 59,
                              circ_p_color = button_true_color)

        if load_ref_ch:
            radiob_refch = MyRadioButtons(self.ax_radiob_refch,
                              ('Overlay', 'Mask'),
                              active = 0,
                              activecolor = button_true_color,
                              orientation = 'horizontal',
                              size = 59,
                              circ_p_color = button_true_color)

        self.z_slice_slider = Slider(self.ax_z_slice_slider,
                'z-slice', 0, V_spots.shape[0],
                valinit=V_spots.shape[0]/2,
                valstep=1,
                orientation='horizontal',
                color=slider_color,
                init_val_line_color=hover_color,
                valfmt='%1.0f')

        if df_spots is not None:
            valmin, valmax = gop_bounds
            # # BUG: If t-test slider should go in the other direction
            # because low p-value means good peak. Ad also the range should
            # be different becuase we want to explore up to 0.2 maybe
            self.gop_slider = Slider(self.ax_gop_limit_slider,
                    f'{gop_how} limit', valmin, valmax,
                    valinit=0,
                    valstep=0.1,
                    orientation='horizontal',
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.1f')

            self.gop_slider.on_changed(self.filter_peaks)

        next_b.on_clicked(self._next)
        continue_b.on_clicked(self._continue)
        abort_b.on_clicked(self._abort)
        z_proj_b.on_clicked(self._draw_V_spots_max_proj)
        self.filter_zbounds_b.on_clicked(self._filter_z_bounds)
        self.radiob_hide_dropped.on_clicked(self.show_hide_spots)
        self.z_slice_slider.on_changed(self.view_z_slice)
        if load_ref_ch:
            radiob_refch.on_clicked(self.radiob_refch_cb)
        if sharp_V_spots is not None:
            self.radiob_sharp.on_clicked(self.radiob_sharp_cb)

        fig.canvas.mpl_connect('button_press_event', self.mouse_down)
        fig.canvas.mpl_connect('resize_event', self.resize_widgets)
        try:
            win_size()
        except:
            pass

        fig.suptitle(f'{self.fig_title}.\n'
            '(Right-click on a spot to print some of its metrics to the console)')

        plt.show()
        matplotlib.use('Agg')

    def show_hide_spots(self, label):
        if label == 'Show all':
            self.spots_plot.set_visible(True)
            self.dropped_spots_plot.set_visible(True)
        elif label == 'Hide dropped':
            self.spots_plot.set_visible(True)
            self.dropped_spots_plot.set_visible(False)
        else:
            self.spots_plot.set_visible(False)
            self.dropped_spots_plot.set_visible(False)
        self.fig.canvas.draw_idle()

    def radiob_sharp_cb(self, label):
        if label == 'Original':
            if self.is_z_proj:
                img = self.V_spots.max(axis=0)
            else:
                img = self.V_spots[int(self.z_slice_slider.val)]
        else:
            if self.is_z_proj:
                img = self.sharp_V_spots.max(axis=0)
            else:
                img = self.sharp_V_spots[int(self.z_slice_slider.val)]
        self.ax0_img_data.set_data(img)
        self.ax0_img_data.set_clim(vmin=img.min(), vmax=img.max())
        self.fig.canvas.draw_idle()


    def _filter_z_bounds(self, event):
        if not self.do_filter_zbounds:
            self.filter_zbounds_b.color = self.button_true_color
            self.filter_zbounds_b.hovercolor = self.button_true_color
            self.filter_zbounds_b.label._text = 'Spots at z-bounds dropped!'
            self.filter_zbounds_b.ax.set_facecolor(self.button_true_color)
            self.do_filter_zbounds = True
        else:
            self.filter_zbounds_b.color = self.axcolor
            self.filter_zbounds_b.hovercolor = self.hover_color
            self.filter_zbounds_b.label._text = 'Filter z-bounds'
            self.filter_zbounds_b.ax.set_facecolor(self.axcolor)
            self.do_filter_zbounds = False
        self.filter_peaks(event)

    def _draw_V_spots_max_proj(self, event):
        self.is_z_proj = True
        self.filter_peaks(event)

    def _draw_ref_mask_contours(self):
        ax = self.ax
        ref_img = self.V_ref.max(axis=0)
        ax[1].imshow(ref_img)

        lab = label(self.ref_mask).max(axis=0)
        rp = regionprops(lab)
        ids = [obj.label for obj in rp]
        contours = (auto_select_slice()
                    .find_contours(lab, ids, group=True))
        for cont in contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax[1].plot(x, y, c='r', alpha=0.5, lw=2, ls='--')

    def _draw_ref_masked(self):
        ax = self.ax
        ref_img = self.V_ref.max(axis=0)
        ref_img[~self.ref_mask.max(axis=0)] = 0
        ax[1].clear()
        ax[1].imshow(ref_img)
        ax[1].set_title(f'{self.ref_channel_name} max z-projection')
        ax[1].axis('off')

    def _draw_ref_overlay(self):
        ax = self.ax
        ax[1].clear()
        ax[1].imshow(self.overlay)
        ax[1].set_title(f'{self.ref_channel_name} max z-projection')
        ax[1].axis('off')


    def radiob_refch_cb(self, label):
        self.ax[1].clear()
        if label == 'Overlay':
            self._draw_ref_overlay()
        elif label == 'Mask':
            self._draw_ref_masked()
        self.ax[1].axis('off')
        self.fig.canvas.draw_idle()


    def view_z_slice(self, event):
        self.is_z_proj = False
        self.filter_peaks(event)

    def filter_peaks_z_slice(self, zyx_coords, current_z):
        z_coords = zyx_coords[:,0]-current_z
        z_coords_mask = np.abs(z_coords) == 0# < self.z_resol_limit_pxl
        zyx_coords_filtered = zyx_coords[z_coords_mask]
        xy_coords_filtered = np.stack((zyx_coords_filtered[:,2],
                                       zyx_coords_filtered[:,1]))
        return xy_coords_filtered

    def get_dropped_zyx_coords(self, zyx_coords, filtered_zyx_coords):
        df1 = pd.DataFrame(data=zyx_coords, columns=['z', 'y', 'x'])
        df2 = pd.DataFrame(data=filtered_zyx_coords, columns=['z', 'y', 'x'])
        df = pd.concat([df1, df2]).drop_duplicates(keep=False)
        dropped_zyx_coords = df[['z', 'y', 'x']].to_numpy()
        return dropped_zyx_coords

    def get_spots_xy_data(self, filtered_zyx_coords, dropped_zyx_coords):
        if self.is_z_proj:
            xy_spots_data = np.stack((filtered_zyx_coords[:,2],
                                      filtered_zyx_coords[:,1]))
            droppedxy_spots_data = np.stack((dropped_zyx_coords[:,2],
                                             dropped_zyx_coords[:,1]))
        else:
            curr_z = int(self.z_slice_slider.val)
            xy_spots_data = self.filter_peaks_z_slice(filtered_zyx_coords,
                                                                   curr_z)
            droppedxy_spots_data = self.filter_peaks_z_slice(
                                               dropped_zyx_coords, curr_z)
        return xy_spots_data, droppedxy_spots_data


    def filter_peaks(self, event):
        # Filter z-bounds
        if self.do_filter_zbounds:
            ellips_test = core.filter_points_resol_limit(
                                         self.local_max_coords,
                                        [self.z_resol_limit_pxl],
                                        self.V_spots.shape,
                                        filter_z_bound=True,
                                        return_valid_points=False)
            filtered_zyx_coords = ellips_test.zyx_coords
            if self.df_spots is not None:
                df_spots = (self.df_spots
                                .set_index(['z', 'y', 'x'])
                                .loc[filtered_zyx_coords[:,0],
                                     filtered_zyx_coords[:,1],
                                     filtered_zyx_coords[:,2]]
                                .reset_index()
                )
        else:
            filtered_zyx_coords = self.local_max_coords.copy()
            df_spots = self.df_spots

        # Filter by gop test
        if self.df_spots is not None:
            gop_thresh_val = [self.gop_slider.val]
            df_gop_test = core.filter_good_peaks(
                                  df_spots, gop_thresh_val,
                                  how=self.gop_how,
                                  which_effsize=self.which_effsize
            )
            filtered_zyx_coords = df_gop_test[['z', 'y', 'x']].to_numpy()

        # Plot data
        dropped_zyx_coords = self.get_dropped_zyx_coords(self.local_max_coords,
                                                         filtered_zyx_coords)

        xy_spots_data, droppedxy_spots_data = self.get_spots_xy_data(
                                                     filtered_zyx_coords,
                                                     dropped_zyx_coords)
        if self.is_z_proj:
            if self.sharp_V_spots is not None:
                if self.radiob_sharp.value_selected == 'Original':
                    img = self.V_spots.max(axis=0)
                else:
                    img = self.sharp_V_spots.max(axis=0)
            else:
                img = self.V_spots.max(axis=0)
        else:
            if self.sharp_V_spots is not None:
                if self.radiob_sharp.value_selected == 'Original':
                    img = self.V_spots[int(self.z_slice_slider.val)]
                else:
                    img = self.sharp_V_spots[int(self.z_slice_slider.val)]
            else:
                img = self.V_spots.max(axis=0)
        self.spots_plot.set_data(xy_spots_data)
        self.dropped_spots_plot.set_data(droppedxy_spots_data)
        self.hidden_dropped_xydata = droppedxy_spots_data
        self.ax0_img_data.set_data(img)
        self.show_hide_spots(self.radiob_hide_dropped.value_selected)
        self.ax0_img_data.set_clim(vmin=img.min(), vmax=img.max())
        self.fig.canvas.draw_idle()


    def mouse_down(self, event):
        if event.inaxes == self.ax[0] and event.button == 3:
            x = event.xdata
            y = event.ydata
            df = self.df_spots
            if df is not None:
                dist = ((df['y']-y)**2 + (df['x']-x)**2).apply(np.sqrt)
                idx_min_dist = dist.idxmin()
                min_dist = dist.min()
                if min_dist < 3:
                    print('')
                    print('--------------------')
                    print(df.loc[[idx_min_dist]]
                                [['vox_spot',
                                  '|norm|_spot',
                                  '|norm|_ref',
                                  self.which_effsize,
                                  'z', 'y', 'x',
                                  'peak_to_background ratio',
                                  'backgr_INcell_OUTspot_mean',
                                  'backgr_INcell_OUTspot_std',
                                  'backgr_INcell_OUTspot_median']])
                elif self.prev_df_spots is not None:
                    df = self.prev_df_spots
                    dist = ((df['y']-y)**2 + (df['x']-x)**2).apply(np.sqrt)
                    idx_min_dist = dist.idxmin()
                    min_dist = dist.min()
                    if min_dist < 3:
                        print('')
                        print('--------------------')
                        print(df.loc[[idx_min_dist]]
                                    [['vox_spot',
                                      '|norm|_spot',
                                      '|norm|_ref',
                                      self.which_effsize,
                                      'z', 'y', 'x',
                                      'peak_to_background ratio',
                                      'backgr_INcell_OUTspot_mean',
                                      'backgr_INcell_OUTspot_std',
                                      'backgr_INcell_OUTspot_median']])


    def _next(self, event):
        plt.close()
        self.next = True

    def _continue(self, event):
        plt.close()

    def _abort(self, event):
        plt.close()
        exit('Execution aborted.')

    def resize_widgets(self, event):
        # [left, bottom, width, height]
        H = 0.03
        W = 0.1
        spF = 0.01
        ax0 = self.ax[0]
        ax0_l, ax0_b, ax0_r, ax0_t = ax0.get_position().get_points().flatten()
        ax0_w = ax0_r-ax0_l
        b = ax0_b-0.01-H
        self.ax_z_slice_slider.set_position([ax0_l, b, ax0_w, H])
        L = ax0_l+ax0_w+2*spF
        self.ax_z_proj_button.set_position([L, b, W/3, H])
        b -= H+spF
        if self.df_spots is not None:
            self.ax_gop_limit_slider.set_position([ax0_l, b, ax0_w, H])

        l1 = ax0_l
        b1 = b-H-spF
        self.ax_continue.set_position([l1, b1, W*2/3, H])
        l2 = l1+W*2/3+spF
        self.ax_filter_zbounds_button.set_position([l2, b1, W, H])
        b2 = b1-H-spF
        self.ax_next.set_position([l2, b2, W, H])
        self.ax_abort.set_position([l1, b2, W*2/3, H])
        l3 = l2+W+spF
        self.ax_radiob_hide_dropped.set_position([l3, b2, W*0.9, H*2+spF])
        l4 = l3+(W*0.9)+(spF/2)
        W1 = (ax0_r-(l4))
        try:
            self.ax_radiob_sharp.set_position([l4, b2, W1, H*2+spF])
        except AttributeError:
            pass
        if self.load_ref_ch:
            ax1 = self.ax[1]
            (ax1_l, ax1_b,
            ax1_r, ax1_t) = ax1.get_position().get_points().flatten()
            w = (ax1_r-ax1_l)*1/3
            ax1_c = ax1_l + (ax1_r-ax1_l)/2
            l = ax1_c - w/2
            b = ax1_b-spF-H
            self.ax_radiob_refch.set_position([l, b, w, H])

def inspect_effect_size(gop_thresh_val, df, filter_by_ref_ch, do_bootstrap,
                        peaks_coords_gop_test, V_spots):
    # Testing
    p = 95
    effsize_temp_lim = gop_thresh_val[0]
    matplotlib.use('TkAgg')
    x = np.arange(len(df))+1
    num_rows = 3 if filter_by_ref_ch else 2
    fig, ax = plt.subplots(num_rows,3)
    ax = ax.flatten()

    i = 0
    colname = (f'effsize_cohen_s_{p}p'
               if do_bootstrap else 'effsize_cohen_s')
    ax[i].scatter(x, df[colname])
    gp = (df[colname] > effsize_temp_lim).sum()
    ax[i].axhline(y=effsize_temp_lim)
    ax[i].set_title(f'{colname}, good peaks = {gp}')

    i += 1
    colname = (f'effsize_hedge_s_{p}p'
               if do_bootstrap else 'effsize_hedge_s')
    ax[i].scatter(x, df[colname])
    gp = (df[colname] > effsize_temp_lim).sum()
    ax[i].axhline(y=effsize_temp_lim)
    ax[i].set_title(f'{colname}, good peaks = {gp}')

    i += 1
    colname = (f'effsize_glass_s_{p}p'
               if do_bootstrap else 'effsize_glass_s')
    ax[i].scatter(x, df[colname])
    gp = (df[colname] > effsize_temp_lim).sum()
    ax[i].axhline(y=effsize_temp_lim)
    ax[i].set_title(f'{colname}, good peaks = {gp}')

    if filter_by_ref_ch:
        i += 1
        colname = (f'effsize_cohen_pop_{p}p'
                   if do_bootstrap else 'effsize_cohen_pop')
        ax[i].scatter(x, df[colname])
        gp = (df[colname] > effsize_temp_lim).sum()
        ax[i].axhline(y=effsize_temp_lim)
        ax[i].set_title(f'{colname}, good peaks = {gp}')

        i += 1
        colname = (f'effsize_hedge_pop_{p}p'
                   if do_bootstrap else 'effsize_hedge_pop')
        ax[i].scatter(x, df[colname])
        gp = (df[colname] > effsize_temp_lim).sum()
        ax[i].axhline(y=effsize_temp_lim)
        ax[i].set_title(f'{colname}, good peaks = {gp}')

        i += 1
        colname = (f'effsize_glass_pop_{p}p'
                   if do_bootstrap else 'effsize_glass_pop')
        ax[i].scatter(x, df[colname])
        gp = (df[colname] > effsize_temp_lim).sum()
        ax[i].axhline(y=effsize_temp_lim)
        ax[i].set_title(f'{colname}, good peaks = {gp}')

    # t-test
    gp = ((df['|spot|:|ref| t-value'] > 0) &
          (df['|spot|:|ref| p-value (t)'] < 0.001)
          ).sum()

    # i += 1
    # ax[i].scatter(x, df['|mNeon|:|mKate| t-value'])
    # ax[i].axhline(y=0)
    # ax[i].set_title(f'|mNeon|:|mKate| t-value, good peaks = {gp}')

    i += 1
    ax[i].imshow(V_spots.max(axis=0))
    ax[i].plot(peaks_coords_gop_test[:,2],
               peaks_coords_gop_test[:,1], 'r.')


    i += 1
    ax[i].scatter(x, df['|spot|:|ref| p-value (t)'])
    ax[i].axhline(y=0.001)
    ax[i].set_title(f'|spot|:|ref| p-value (t), good peaks = {gp}')

    i += 1
    ax[i].scatter(x, df['|norm|_spot'])
    ax[i].set_title('|norm|_spot')
    try:
        win_size()
    except:
        pass
    plt.show()
    matplotlib.use('Agg')


class inspect_gaussian_fit:
    def __init__(self, V_spots, df_spotFIT_click, spots_3D_labs, ID_3Dslices,
                 ID_bboxs_lower, IDs, dfs_intersect, df_spotFIT):

        ID = df_spotFIT_click.index.get_level_values(0)[0]
        spot_id = df_spotFIT_click.index.get_level_values(1)[0]
        ID_idx = IDs.index(ID)
        ID_slice = ID_3Dslices[ID_idx]
        min_z, min_y, min_x = ID_bboxs_lower[ID_idx]
        spots_3D_lab_ID = spots_3D_labs[ID_idx]
        z, y, x = df_spotFIT_click.iloc[0][['z', 'y', 'x']].astype(int)
        V_spots_ID = V_spots[ID_slice]
        img = V_spots_ID.max(axis=0)
        fig = plt.figure()
        ax = [None]*4
        ax[0] = fig.add_subplot(131)
        ax[1] = fig.add_subplot(232)
        ax[2] = fig.add_subplot(235)
        ax[3] = fig.add_subplot(133)
        ax[0].imshow(img)
        ax[0].plot(x-min_x, y-min_y, 'r.')

        # Get clicked spot coeffs
        Z, Y, X = V_spots_ID.shape
        zz, yy, xx = np.ogrid[0:Z, 0:Y, 0:X]
        model = core.lstq_Model()
        z0, y0, x0, sz, sy, sx, A, B = (df_spotFIT_click.iloc[0]
                            [['z_fit', 'y_fit', 'x_fit',
                              'sigma_z_fit', 'sigma_y_fit', 'sigma_x_fit',
                              'A_fit', 'B_fit']])
        z0 -= min_z
        y0 -= min_y
        x0 -= min_x
        coeffs = [z0, y0, x0, sz, sy, sx, A]

        V_fit = model.gaussian_3D(zz, yy, xx, coeffs)
        V_fit_show = V_fit.copy()
        V_fit_show[spots_3D_lab_ID==spot_id] += B

        # Get neighbouring spots coeffs
        df_intersect = dfs_intersect[ID_idx].reset_index().set_index('id')
        neigh_ids = df_intersect.at[spot_id, 'neigh_ids']
        neigh_ids.remove(spot_id)
        neigh_coeffs = []
        df_spotFIT_ID = df_spotFIT.loc[ID]

        for id in neigh_ids:
            n_coeffs = (df_spotFIT_ID.loc[id]
                            [['z_fit', 'y_fit', 'x_fit',
                              'sigma_z_fit', 'sigma_y_fit', 'sigma_x_fit',
                              'A_fit']]).to_list()
            n_B = df_spotFIT_ID.at[id, 'B_fit']
            _z, _y, _x = n_coeffs[:3]
            _z -= min_z
            _y -= min_y
            _x -= min_x
            n_coeffs[:3] = _z, _y, _x
            V_fit += model.gaussian_3D(zz, yy, xx, n_coeffs)
            V_fit_show += V_fit
            V_fit_show[spots_3D_lab_ID==id] += n_B

        V_fit_show[spots_3D_lab_ID==0] = B

        img_gauss = V_fit_show.max(axis=0)
        ax[1].imshow(img_gauss)
        ax[1].plot(x-min_x, y-min_y, 'r.')
        ax[1].plot(x0, y0, 'k.')

        ax[2].imshow(spots_3D_lab_ID.max(axis=0))
        ax[2].plot(x-min_x, y-min_y, 'r.')
        ax[2].plot(x0, y0, 'k.')

        yy_raw = V_spots_ID[z-min_z, y-min_y]
        yy_fit = (V_fit+B)[z-min_z, y-min_y]
        spot_mask = spots_3D_lab_ID==spot_id
        xx_fit = np.nonzero(spot_mask)[2]
        min_x_fit = xx_fit.min()
        max_x_fit = xx_fit.max()

        ax[3].scatter(range(len(yy_raw)), yy_raw)
        ax[3].plot(range(len(yy_fit)), yy_fit, c='r')
        ax[3].axvline(min_x_fit, color='firebrick', ls='--')
        ax[3].axvline(max_x_fit, color='firebrick', ls='--')

        sub_win = embed_tk('3D Gaussian fits', [1280,960,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        self.fig = fig
        self.ax = ax
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def _close(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()


class win_size:
    def __init__(self, w=1, h=1, swap_screen=False):
        try:
            monitor = Display()
            screens = monitor.get_screens()
            num_screens = len(screens)
            displ_w = int(screens[0].width*w)
            displ_h = int(screens[0].height*h)
            x_displ = screens[0].x
            #Display plots maximized window
            mng = plt.get_current_fig_manager()
            if swap_screen:
                geom = "{}x{}+{}+{}".format(displ_w,(displ_h-70),(displ_w-8), 0)
                mng.window.wm_geometry(geom) #move GUI window to second monitor
                                             #with string "widthxheight+x+y"
            else:
                geom = "{}x{}+{}+{}".format(displ_w,(displ_h-70),-8, 0)
                mng.window.wm_geometry(geom) #move GUI window to second monitor
                                             #with string "widthxheight+x+y"
        except:
            try:
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
            except:
                pass

def _crop_collage_two_lines(image, collage_idxs=None):
    try:
        if collage_idxs is not None:
            h, w = image.shape
            ratio = 8
            if w/h > ratio:
                new_w_idx = np.abs(np.array(collage_idxs)-ratio*h).argmax()
                new_w = collage_idxs[new_w_idx]
                img_2lines = np.zeros((2*h, new_w), image.dtype)
                img_2lines[:h, :new_w] = image[:, :new_w]
                remain_w = image[:, new_w:].shape[1]
                remain_w = new_w if remain_w>new_w else remain_w
                img_2lines[h:, :remain_w] = image[:, new_w:new_w+remain_w]
                return img_2lines
            else:
                return image
        else:
            return image
    except TypeError:
        traceback.print_exc()
        return image


def _try_all(image, methods=None, figsize=None, num_cols=2,
             verbose=True, collage_idxs=None, idx_crop=None):
    """Returns a figure comparing the outputs of different methods.
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    methods : dict, optional
        Names and associated functions.
        Functions must take and return an image.
    figsize : tuple, optional
        Figure size (in inches).
    num_cols : int, optional
        Number of columns.
    verbose : bool, optional
        Print function name for each method.
    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    from matplotlib import pyplot as plt

    # Handle default value
    methods = methods or {}

    num_rows = math.ceil((len(methods) + 1.) / num_cols)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize,
                           sharex=True, sharey=True)
    ax = ax.ravel()

    shown_img = image.copy()
    if idx_crop is not None:
        shown_img = shown_img[:idx_crop]

    shown_img = _crop_collage_two_lines(image, collage_idxs=collage_idxs)
    if shown_img.shape != image.shape:
        is_cropped = True
    else:
        is_cropped = False
    ax[0].imshow(shown_img, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    i = 1
    for name, func in methods.items():
        ax[i].set_title(name)
        try:
            thresh_img = func(image)
            if idx_crop is not None:
                thresh_img = thresh_img[:idx_crop]
            shown_thresh = _crop_collage_two_lines(thresh_img,
                                        collage_idxs=collage_idxs)
            ax[i].imshow(shown_thresh, cmap=plt.cm.gray)
        except Exception as e:
            ax[i].text(0.5, 0.5, "%s" % type(e).__name__,
                       ha="center", va="center", transform=ax[i].transAxes)
        i += 1
        if verbose:
            print(func.__orifunc__)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    return fig, ax, is_cropped


def my_try_all_threshold(image, figsize=(8, 5), verbose=False,
                         collage_idxs=None,
                         idx_crop=None):
    """Returns a figure comparing the outputs of different thresholding methods.
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    figsize : tuple, optional
        Figure size (in inches).
    verbose : bool, optional
        Print function name for each method.
    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    Notes
    -----
    The following algorithms are used:
    * isodata
    * li
    * mean
    * minimum
    * otsu
    * triangle
    * yen
    Examples
    --------
    >>> from skimage.data import text
    >>> fig, ax = try_all_threshold(text(), figsize=(10, 6), verbose=False)
    """
    def thresh(func):
        """
        A wrapper function to return a thresholded image.
        """
        def wrapper(im):
            return im > func(im)
        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper

    # Global algorithms.
    methods = OrderedDict({'Isodata': thresh(threshold_isodata),
                           'Li': thresh(threshold_li),
                           'Mean': thresh(threshold_mean),
                           'Minimum': thresh(threshold_minimum),
                           'Otsu': thresh(threshold_otsu),
                           'Triangle': thresh(threshold_triangle),
                           'Yen': thresh(threshold_yen)})

    return _try_all(image, figsize=figsize, methods=methods, verbose=verbose,
                    collage_idxs=collage_idxs, idx_crop=idx_crop)

if __name__ == '__main__':
    imshow_tk(np.random.randint(0,255, size=(500,500)))
