#!/usr/bin/env python3
import numpy as n
import os;
import matplotlib.pyplot as pyplot;


def main(file):
    print('File: ', file);
    image    = n.load(file);
    shape    = image.shape;
    show_max = n.amax(image);
    for i in range(0,shape[0]):
        f, ax = pyplot.subplots(1,3);
        ax[0].set_title('Slice: ' + str(i));
        ax[0].axis('off');ax[1].axis('off');ax[2].axis('off');
        ax[0].imshow(image[i,:,:,0], cmap = pyplot.cm.turbo, vmin = 0.0, vmax = show_max);
        ax[1].imshow(image[:,i,:,0], cmap = pyplot.cm.turbo, vmin = 0.0, vmax = show_max);
        ax[2].imshow(image[:,:,i,0], cmap = pyplot.cm.turbo, vmin = 0.0, vmax = show_max);
        f.subplots_adjust(hspace=0,wspace=0,top=1, bottom = 0, right = 1, left = 0);
        pyplot.show();
    



import argparse
if __name__ == '__main__' :
    parser = argparse.ArgumentParser();
    parser.add_argument('file', type = str, help = 'Path to the .npy file you want to open and view slice by slice.');
    args = parser.parse_args()
    main(args.file);
