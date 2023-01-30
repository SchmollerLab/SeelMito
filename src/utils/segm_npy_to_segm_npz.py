import os
import sys
import numpy as np
import skimage.io

script_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(script_dir)
sys.path.append(main_dir)

import prompts

exp_path = prompts.folder_dialog(
    title='Select folder containing Position_n folders'
)

filename_end = 'segm.npy'
rename_end = 'segm.npz'

for pos in os.listdir(exp_path):
    pos_path = os.path.join(exp_path, pos)
    if pos.find('Position_')==-1 or not os.path.isdir(pos_path):
        continue

    images_path = os.path.join(pos_path, 'Images')
    if not os.path.exists(images_path):
        continue

    files = os.listdir(images_path)
    searched_files = [f for f in files if f.endswith(filename_end)]
    if not searched_files:
        continue

    file_to_rename = os.path.join(images_path, searched_files[0])
    data = np.load(file_to_rename).astype(np.uint8)

    new_filename = searched_files[0].replace(filename_end, rename_end)
    new_path = os.path.join(images_path, new_filename)

    # skimage.io.imsave(new_path, data)

    np.savez_compressed(new_path, data)
