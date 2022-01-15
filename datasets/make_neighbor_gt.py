from PIL import Image
import numpy as np
from glob import glob
import argparse
import os
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import utils

DILATIONS = (1,2,4,8,16,32,64)
save_folder = '/home/lab/ssd1/DynamicConv/Detectron_Datasets/cityscapes/gtFine/neighbor'

def neighbor_affinity(panoptic, dilation):
    kernel_size = 2 * dilation + 1
    pad_inst = np.pad(panoptic, dilation, 'constant', constant_values=0)
    windows = sliding_window_view(pad_inst, (kernel_size, kernel_size))

    h, w, _, _ =  windows.shape
    neighbor = windows[:, :, ::dilation, ::dilation].reshape(*panoptic.shape, -1)
    neighbor = np.delete(neighbor, 4, axis=2)
    current_pixel = np.expand_dims(panoptic, axis=2).repeat(8, axis=2)
    affinity = current_pixel == neighbor
    return affinity



def main():
    panoptic_paths = glob(
        '/home/lab/ssd1/DynamicConv/Detectron_Datasets/cityscapes/gtFine/cityscapes_panoptic_*/*_panoptic.png')
    print("Panoptic images : ", len(panoptic_paths))

    for path in tqdm(panoptic_paths):
        color = np.array(Image.open(path))
        id = utils.rgb2id(color)

        split = path.split('/')[-2].split('_')[-1]
        if split == 'test' : continue

        affinity = []
        for d in DILATIONS:
            affinity.append(neighbor_affinity(id, d))

        affinity = np.concatenate(affinity, axis=2)


        np.save(os.path.join(save_folder, split, os.path.basename(path).replace('panoptic.png', 'neighbor')),
                affinity)



if __name__ == '__main__':
    main()

