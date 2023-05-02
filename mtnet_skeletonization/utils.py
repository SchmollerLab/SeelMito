import sys
import os
import inspect
from datetime import datetime
from pprint import pprint

import numpy as np
import skimage.morphology

from natsort import natsorted

def printl(*objects, pretty=False, is_decorator=False, **kwargs):
    # Copy current stdout, reset to default __stdout__ and then restore current
    current_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = 2 if is_decorator else 1
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filpath = callingframe_info.filename
    filename = os.path.basename(filpath)
    print_func = pprint if pretty else print
    print('*'*30)
    print(f'{timestap} - File "{filename}", line {callingframe_info.lineno}:')
    print_func(*objects, **kwargs)
    print('='*30)
    sys.stdout = current_stdout

def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
    ])

def get_pos_foldernames(exp_path):
    ls = listdir(exp_path)
    pos_foldernames = [
        pos for pos in ls if pos.find('Position_')!=-1
        and os.path.isdir(os.path.join(exp_path, pos))
        and os.path.exists(os.path.join(exp_path, pos, 'Images'))
    ]
    return pos_foldernames

def get_skel_len(mtnet_mask, obj):
    mtnet_mask_obj = mtnet_mask[obj.slice].copy()
    mtnet_mask_obj[~obj.image] = False

    mtnet_skel_obj = skimage.morphology.skeletonize(mtnet_mask_obj)

    skel_len = np.count_nonzero(mtnet_skel_obj)
    return skel_len, mtnet_skel_obj
