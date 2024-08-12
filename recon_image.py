#!/usr/bin/env python3
import numpy as n
print('Import tf');
import tensorflow as tf
import time
print('Import tf layers');
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D, Flatten, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
import scipy.sparse as sparse
import math
import os;
import subprocess;
import sys;
import math;
from LatentSpaceKernelFunctions  import DeepKernelNetwork; 


#
#
#
#
#
#
"""
Trainable PET image reconstruction network.
"""
#
#
#
#
#

def get_image_metrics(data_set, target_data, show_images = False):
        psnr_out, ssim_out, rmse_out = [],[],[]
        for i in range(len(data_set)):
            dat = data_set[i];
            tar = target_data[i];
            
            ssim  = SSIM(tar, dat, data_range = tar.max() - tar.min() );
            psnr  = PSNR(tar, dat, data_range = tar.max() - tar.min() );
            nrmse = nRMSE(tar, dat, normalization='euclidean');
            
            
            #
            if math.isnan(psnr) :
                pass
            else:
                psnr_out.append(psnr);
            
            if math.isnan(ssim) :
                pass
            else:
                ssim_out.append(ssim);
            
            if math.isnan(nrmse) :
                pass
            else:
                rmse_out.append(nrmse);
        psnr_out = n.array(psnr_out);
        ssim_out = n.array(ssim_out);
        rmse_out = n.array(rmse_out);
        return psnr_out, ssim_out, rmse_out;




def load_fd_data(FD_data_directory, nof_data = None):
    if FD_data_directory[-1] !='/':
        FD_data_directory+='/'
    else:
        pass
    label_dir  = FD_data_directory;
    file_names = os.popen('ls ' + label_dir).read().split('\n')[:-1];
    print('Nof files: ', len(file_names));
    full_label = []
    if nof_data == None:
        file_range = len(file_names);
    else:
        file_range = nof_data;
        
    #idx = [0];
    for i in range(file_range):
        file = file_names[i];
        l = n.load(label_dir + file);
        full_label.append(l)
    return n.array(full_label)

def load_data(training_data_directory, nof_data = None, slice_per_input = 3):

    pad = slice_per_input//2;

    if training_data_directory[-1] !='/':
        training_data_directory+='/'
    else:
        pass
    data_dir   = training_data_directory + 'data/'
    label_dir  = training_data_directory + 'label/'
    file_names = os.popen('ls ' + label_dir).read().split('\n')[:-1]
    print('Nof files: ', len(file_names));
    full_data  = []
    full_label = []
    if nof_data == None:
        file_range = len(file_names);
    else:
        file_range = nof_data;
        
    #idx = [0];
    for i in range(file_range):
        file = file_names[i];
        dat = [];
        lab = [];
        d = n.load(data_dir+file);
        l = n.load(label_dir + file);
        #(155,192,192,1), (155,192,192,3);
        z = d.shape[0];
        for j in range(pad,z-pad):
            d_stack = d[j-pad:j+pad+1];
            l_stack = l[j];
            dat.append(d_stack);
            lab.append(l_stack);

        dat = n.array(dat);
        lab = n.array(lab);
        b, s, h, w, c = dat.shape;
        
        dat = dat.transpose((0,2,3,4,1))
        dat = dat.reshape((b,h,w,c*s));
        
        print('Dat: ', dat.shape);
        print('Lab: ', lab.shape);
        full_data.append(dat);
        full_label.append(lab);
    return n.array(full_data), n.array(full_label)


def get_checkpoints(checkpoint_directory):
    files = os.popen('ls ' + checkpoint_directory).read().split('\n')[:-1];
    try: files.remove('checkpoint');
    except: pass;
    checkpoints = [];
    epochs      = [];
    for file in files:
        file_name = file.split('.')[0];
        epoch     = file_name.split('-')[1];
        if checkpoint_directory + file_name in checkpoints:
            pass;
        else:
            checkpoints.append( checkpoint_directory + file_name);
            epochs.append((int(epoch)-1)*5);
    epochs, checkpoints = zip(*sorted(zip(epochs, checkpoints)))
    print('(Training epochs, checkpoint name)');
    for i in range(len(epochs)):
        print(epochs[i], checkpoints[i]);
    return checkpoints, epochs

def main(data_directory, checkpoint, kernel_patch_size, kernel_stride_size, slice_per_input, save_file):
    
    gpus       = tf.config.list_physical_devices('GPU');
    if gpus:
       try:
          tf.config.set_visible_devices([   ],'GPU');
          #tf.config.experimental.set_memory_growth(gpus[0], True);
          #tf.config.experimental.set_memory_growth(gpus[1], True);
          #tf.config.experimental.set_memory_growth(gpus[2], True);
          logical_gpus = tf.config.list_logical_devices('GPU');
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU");
       except RuntimeError as e:
          print(e)


    network            = DeepKernelNetwork(kernel_patch_size = kernel_patch_size,  kernel_stride_factor = kernel_stride_size, slice_per_batch = slice_per_input); 
    schedule_lr        = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 50, 0.99, staircase=False, name=None)
    optimiser          = tf.keras.optimizers.Adam(learning_rate=schedule_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    optimiser          = optimiser;
    network_checkpoint = tf.train.Checkpoint(optimizer = optimiser, model = network);
    network_checkpoint.restore(checkpoint);
    data, label        = load_data(data_directory, slice_per_input = slice_per_input);
    print(' data: ', data.shape );
    print('label: ', label.shape);
    proc_batch_size    = 10;
    if len(data)%proc_batch_size == 0:
        nof_batches = data.shape[1]//proc_batch_size;     
    else:
        nof_batches = (data.shape[1]//proc_batch_size) ;
    nof_datasets    = data.shape[0];
    print('Nof Datasets: ', nof_datasets);
    for j in range(nof_datasets):
        preds = [];
        for i in range(nof_batches):
            print('Processing: ', (i+1),'/',nof_batches, end = '\r');
            input_batch = data[j,i*proc_batch_size:(i+1)*proc_batch_size];
            print('Input batch: ', input_batch.shape);
            pred, pt_d1_, pt_d3, pt_d5, mixed1, mixed2, mixed3        = network(input_batch); 
            print('Pred Max: ', n.amax(pred));
            print('Pred Min: ', n.amin(pred));
            print('Pred Mean: ', n.mean(pred));
            print('Pred: ', pred.shape);
            preds.append(pred);
        print('Len preds: ', len(preds));
        preds = n.concatenate(preds,axis = 0);
        print('Saving file: ', save_file);
        datum_save_file = save_file + '_Ex' + str(j) + '.npy'; 
        print('J:', j);
        print('Save file: ', datum_save_file);
        n.save(datum_save_file, preds); 
 
import argparse
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help = 'Data you want to reconstruct.');
    parser.add_argument('checkpoint'    , help = 'Set this to a path you wish to save the reconstructed images to. If you dont, they will not save, and instead just be displayed in a matplotlib.figure.');
    parser.add_argument('kernel_patch_size', type=str, help = 'The kernel function patch size at each layer.');
    parser.add_argument('kernel_stride_size', type=str, help = 'The kernel function stride size at each layer.');
    parser.add_argument('slice_per_input', type=int, help = 'The z slices to use per input.');
    parser.add_argument('save_file', type = str, help = 'The file you want to save.');
    args  = parser.parse_args();
    args.kernel_patch_size  = n.array(args.kernel_patch_size.split(',')).astype(n.int16);
    args.kernel_stride_size = n.array(args.kernel_stride_size.split(',')).astype(n.int16);
    print('   Data Directory: ', args.data_directory);
    print('       Checkpoint: ', args.checkpoint);
    print('Kernel patch size: ', args.kernel_patch_size);
    print('kernel stride size: ', args.kernel_stride_size);
    print('  Slice_per_input: ', args.slice_per_input);
    print('        Save file: ', args.save_file);
    main(args.data_directory, args.checkpoint, args.kernel_patch_size, args.kernel_stride_size, args.slice_per_input, args.save_file);
