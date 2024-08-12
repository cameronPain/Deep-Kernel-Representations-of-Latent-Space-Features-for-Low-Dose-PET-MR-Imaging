#!/usr/bin/env python3
print('Start');
import numpy as n
import sys
import tensorflow as tf
import time
import os;
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import subprocess;
from LatentSpaceKernelFunctions.py  import DeepKernelNetwork, PT_PT_Feature_Discriminator, MR_PT_Feature_Discriminator;

#
#
#
#
#
#
"""

Latent-space kernel functions for low-dose PET-MR robust to variable dose reduction.

"""
#
#
#
#
#
#


class DiscriminatorLoss(tf.keras.losses.Loss):
      def __init__(self):
          super().__init__(reduction = tf.keras.losses.Reduction.NONE);

      def call(self, label, pred):
          loss = label * -1*tf.math.log(1e-3 + pred) + (1-label) *  -1*tf.math.log(1e-3 + 1 - pred)   ;
          #loss  = tf.math.reduce_mean(tf.math.square(tf.math.subtract(label, pred)), axis=-1);
          tf.print('Computing disc loss: ', loss.shape);
          tf.print('Computing disc loss: ', loss);
          return loss;
        

class TrainingLoss(tf.keras.losses.Loss):
      def __init__(self, g_disc = 0.1, g_mrpt_disc = 0.1):
          super().__init__(reduction = tf.keras.losses.Reduction.NONE);
          self.g_disc      = g_disc;
          self.g_mrpt_disc = g_mrpt_disc;

      def call(self, y_true, y_pred):
          im_pred, disc_pred_j, disc_pred_m, mrpt_disc_pred_j = y_pred;
          #loss = -1*tf.math.log(1e-4 + y_pred);
          print('Label: ', y_true.shape);
          print(' Pred: ', im_pred.shape);
          disc_loss       = -1*tf.math.log(1e-3 + disc_pred_j) - tf.math.log(1e-3 + 1 - disc_pred_m);
          mrpt_disc_loss  = -1*tf.math.log(1e-3 + 1 - mrpt_disc_pred_j);
          image_loss      = tf.math.reduce_mean(tf.math.square( y_true - im_pred), axis=(1,2));
          print('     Image loss: ', tf.math.reduce_mean(image_loss));
          print(' MRPT disc loss: ', tf.math.reduce_mean(mrpt_disc_loss) * self.g_mrpt_disc);
          print('      Disc loss: ', tf.math.reduce_mean(disc_loss) * self.g_disc);
          loss       = image_loss + self.g_disc* disc_loss + self.g_mrpt_disc * mrpt_disc_loss;
          print('Generator Loss: ', loss.shape);
          print('Generator Loss: ', loss);
          return loss;

class Distributed_Trainer:
      def __init__(self, strategy, model, data_directory, checkpoint_directory, disc_checkpoint_directory, batch_size = 12):
          self.strategy             = strategy            ;
          self.model                = model               ;
          self.data_directory       = data_directory      ;
          self.checkpoint_directory      = checkpoint_directory;
          self.disc_checkpoint_directory = disc_checkpoint_directory;
          self.batch_size           = batch_size          ;
          self.file_names           = os.popen('ls ' + self.data_directory  + '/data' ).read().split('\n')[:-1];
          self.pt_pt_disc           = PT_PT_Feature_Discriminator();
          self.mr_pt_disc           = MR_PT_Feature_Discriminator();

      def __load_data__(self, training_data_directory,  nof_slices):
        if training_data_directory[-1] !='/':
            training_data_directory+='/'
        else:
            pass
        
        file_names   = self.file_names;
        data_dir     = training_data_directory + 'data/'
        label_dir    = training_data_directory + 'label/'
        #print('Data directory: ', data_dir);
        #print('File Names: ', file_names);
        full_data  = []
        full_label = []
        for file in file_names:
            #print('Loading file: ', file);
            datum  = n.load(data_dir + file)[:152,:,:,:3];
            lab    = n.load(label_dir + file)[:152,:,:,:];
            datum  = n.pad(datum, ((4,4),(0,0),(0,0),(0,0)));
            lab    = n.pad(lab, ((4,4),(0,0),(0,0),(0,0)));
            full_label.append( lab);
            full_data.append(datum); #Autoencoder like model.
        data  = full_data;
        label = full_label;
        data  = n.array(data);#Slice them to 144 slices along z for an easily divisible number so no remainder when converting to blocks.
        label = n.array(label);
        #print(' Data: ', data.shape);
        #print('Label: ', label.shape);
        data       = tf.concat([data,label],axis=-1);
        print('Data: ', data.shape);
        rolled_data = [];
        #roll_sizes = [i for i in range(1,nof_slices)];
        if n.random.random() > 0.5:
           #print('Roll: 1');
           data        = tf.roll(data, 1, 1);
        else:
           #print('Roll: 0');
           pass;
        #print('          Ims: ', data.shape);
        im,z,y,x,c = data.shape
        z          = z-10;
        data       = data[:,4:-6,:,:,:];
        nof_blocks = (150//((nof_slices//2)+1))-1;
        #print('     nof blocks: ', nof_blocks);
        data       = tf.transpose(data, perm = (1,2,3,4,0));
        data       = tf.reshape(data, shape = (z,y,x,im*c));
        data       = tf.expand_dims(data, 0);
        print('Data: ', data.shape);
        blocks     = tf.extract_volume_patches(data, (1,nof_slices,192,192,1), (1,(nof_slices//2) + 1 ,1,1,1), "VALID");
        print('Blocks: ', blocks.shape);
        #print('        blocks: ', blocks.shape);
        blocks     = tf.reshape(blocks,   shape = (nof_blocks, nof_slices,192,192,c,im));
        blocks     = tf.transpose(blocks, perm  = (5,0,1,2,3,4));
        blocks     = tf.reshape(blocks,   shape = (im*nof_blocks,nof_slices,192,192,c));
        labels     = tf.cast(blocks[:,nof_slices//2,:,:,-1:], dtype=n.float32); #These are the central slices for targets.
        blocks     = blocks[:,:,:,:,:-1];
        blocks     = tf.transpose(blocks, perm  = (0,2,3,4,1)                             );
        blocks     = tf.cast(tf.reshape(blocks,   shape = (im*nof_blocks,192,192,(c-1)*nof_slices)), dtype=n.float32)  ;
        print('Blocks: ', blocks.shape);
        print('Labels: ', labels.shape);
        self.data  = tf.concat([blocks, labels],axis=-1);


      def __create_model__(self, kernel_patch_size = (32,16,8,4,2,1), slice_per_batch = 3, kernel_stride_factor = (8,4,2,1,1,1)):
        self.base_model      = self.model(kernel_patch_size= kernel_patch_size, slice_per_batch=slice_per_batch, kernel_stride_factor = kernel_stride_factor);
        self.slice_per_batch = slice_per_batch;

      def __create_optimiser__(self):
        schedule_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 25, 0.99, staircase=False, name=None)
        optimiser = tf.keras.optimizers.Adam(learning_rate=schedule_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        self.optimiser = optimiser;

      def __create_ptpt_disc_model__(self):
        self.pt_pt_disc   = PT_PT_Feature_Discriminator();

      def __create_mrpt_disc_model__(self):
        self.mr_pt_disc   = MR_PT_Feature_Discriminator();

      def __create_checkpoint_directory__(self):
          proc = subprocess.Popen(['mkdir', self.checkpoint_directory]);
          proc.wait();

      def __create_loss_function__(self, g_disc = 0.1, g_mrpt_disc = 0.1):
          self.custom_loss = TrainingLoss( g_disc = g_disc , g_mrpt_disc = g_mrpt_disc);
      
      def __compute_loss__(self, labels, prediction):
          print('Computing loss');
          per_example_loss = self.custom_loss(labels, prediction);
          loss = tf.nn.compute_average_loss(per_example_loss);
          return loss;

      def __create_tracking_metrics__(self):
          self.track_loss =  tf.keras.metrics.Mean(name = 'test_loss');

      def __create_mrpt_disc_optimiser__(self):
        schedule_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 25, 0.99, staircase=False, name=None)
        optimiser = tf.keras.optimizers.Adam(learning_rate=schedule_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        self.mrpt_disc_optimiser = optimiser;

      def __create_ptpt_disc_optimiser__(self):
        schedule_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 25, 0.99, staircase=False, name=None)
        optimiser = tf.keras.optimizers.Adam(learning_rate=schedule_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        self.ptpt_disc_optimiser = optimiser;

      def __compute_disc_loss__(self, joint_distn, marginal_distn): #__compute_mrpt_loss__
          loss =   -1*tf.math.log(1e-3 + joint_distn) +  -1*tf.math.log(1e-3 + 1 - marginal_distn)   ;
          return loss;

      def __create_disc_loss_function__(self):
          self.custom_disc_loss = DiscriminatorLoss(); #tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE);

      def __create_checkpoint_directory__(self):
          proc = subprocess.Popen(['mkdir', self.checkpoint_directory]);
          proc.wait();

      def __create_disc_checkpoint_directory__(self):
          proc = subprocess.Popen(['mkdir', self.disc_checkpoint_directory]);
          proc.wait();

      def __compile_model_optimiser_checkpoint__(self, slice_per_batch = 5, init_learning_rate = 0.001, batch_size = 12, checkpoint = None, mrpt_disc_checkpoint = None, ptpt_disc_checkpoint = None, kernel_patch_size=(8,4,2,1,1,1), kernel_stride_factor = (1,1,1,1,1,1), g_mrpt = 0.1, g_ptpt = 0.1):
               self.slice_per_batch    = slice_per_batch
               self.init_learning_rate = init_learning_rate;
               self.__create_model__(slice_per_batch = slice_per_batch, kernel_patch_size = kernel_patch_size, kernel_stride_factor = kernel_stride_factor);
               self.__create_optimiser__();
               self.__create_ptpt_disc_optimiser__();
               self.__create_mrpt_disc_optimiser__();
               self.__create_loss_function__(g_disc = g_ptpt, g_mrpt_disc = g_mrpt);
               self.__create_checkpoint_directory__();
               self.__create_ptpt_disc_model__();
               self.__create_mrpt_disc_model__();
               self.__create_disc_loss_function__();
               self.__create_disc_checkpoint_directory__();
               self.checkpoint           = tf.train.Checkpoint(optimizer = self.optimiser          , model = self.base_model);
               self.ptpt_disc_checkpoint = tf.train.Checkpoint(optimizer = self.ptpt_disc_optimiser, model = self.pt_pt_disc);
               self.mrpt_disc_checkpoint = tf.train.Checkpoint(optimizer = self.mrpt_disc_optimiser, model = self.mr_pt_disc);

               if checkpoint:
                  self.checkpoint.restore(checkpoint);
               else:
                  pass;
               if ptpt_disc_checkpoint:
                  self.ptpt_disc_checkpoint.restore(ptpt_disc_checkpoint);
               else:
                  pass;
               if mrpt_disc_checkpoint:
                  self.mrpt_disc_checkpoint.restore(mrpt_disc_checkpoint);
               else:
                  pass;


      def __training_step__(self, inputs, labels):
          with tf.GradientTape(persistent=True) as gen_tape:
              print('Input shape: ', inputs.shape);
              pred, pt_d1, pt_d3, pt_d5, mr_d1, mr_d2, mr_d3 = self.base_model(inputs, training=True);
              b,h,w,c      = labels.shape;
              b,px,py,cd   = pt_d3.shape;
              m_pt_d3      = tf.stop_gradient(tf.roll(pt_d3, 1, 0));
              m_pt_d3      = tf.reshape(m_pt_d3, shape = (b,1,px,1,py,cd));
              pt_d3        = tf.reshape(pt_d3, shape = (b,1,px,1,py,cd));
              print('m_pt_d3: ', m_pt_d3.shape);
              b,b_,dh,c_,dw,dc   = m_pt_d3.shape;
              Px, Py       = 24, 24
              PNx, PNy     = 192//Px, 192//Py;
              Nxt,Nyt      = Px//dh, Py//dw;

              fd_patches   = tf.reshape(labels, shape = (b,PNx,Px,PNy,Py,c));
              pt_d3_tile   = tf.tile(pt_d3, (1, PNx, Nxt, PNy, Nyt, 1));
              m_pt_d3_tile = tf.tile(m_pt_d3, (1, PNx, Nxt, PNy, Nyt,1));

              m_disc_in    = tf.concat([fd_patches, m_pt_d3_tile],axis=-1);
              m_disc_in    = tf.transpose(m_disc_in, perm = (2,4,5,0,1,3))
              m_disc_in    = tf.reshape(m_disc_in, shape = (Px,Px,c + cd,b*PNx*PNy));
              m_disc_in    = tf.transpose(m_disc_in, perm = (3,0,1,2));

              j_disc_in    = tf.concat([fd_patches, pt_d3_tile],axis=-1);
              j_disc_in    = tf.transpose(j_disc_in, perm = (2,4,5,0,1,3))
              j_disc_in    = tf.reshape(j_disc_in, shape = (Px,Py,c + cd,b*PNx*PNy));
              j_disc_in    = tf.transpose(j_disc_in, perm = (3,0,1,2));

              tf.print('disc pred in: ', j_disc_in.shape);
              tf.print('disc  lab in: ', m_disc_in.shape);
              disc_inp     = tf.concat([j_disc_in, m_disc_in],axis=0);
              disc_pred    = self.pt_pt_disc(disc_inp, training=True);
              j_disc_out   = disc_pred[:len(j_disc_in)];
              m_disc_out   = disc_pred[len(j_disc_in):];

              j_disc_out   = tf.transpose(j_disc_out, perm = (1,0));
              j_disc_out   = tf.reshape(j_disc_out, shape = (1,b,PNx,PNy));
              j_disc_out   = tf.transpose(j_disc_out, perm = (1,2,3,0));
              j_disc_out   = tf.math.reduce_mean(j_disc_out, axis=(1,2)); #(b,1);

              m_disc_out   = tf.transpose(m_disc_out, perm = (1,0));
              m_disc_out   = tf.reshape(m_disc_out, shape = (1,b,PNx,PNy));
              m_disc_out   = tf.transpose(m_disc_out, perm = (1,2,3,0));
              m_disc_out   = tf.math.reduce_mean(m_disc_out, axis=(1,2)); #(b,1);
             
              tf.print('disc pred J out: ', j_disc_out.shape);
              tf.print('disc  lab M out: ', m_disc_out.shape);
              tf.print('disc pred 1 : disc pred 0');
              tf.print(tf.math.reduce_mean(j_disc_out), ' : ', tf.math.reduce_mean(m_disc_out))
              for i in range(len(j_disc_out)):
                  tf.print(j_disc_out[i], ' : ', m_disc_out[i]);

              j_mrpt_disc_in  = [tf.stop_gradient(mr_d1), tf.stop_gradient(mr_d2), tf.stop_gradient(mr_d3), pt_d5];
              m_mrpt_disc_in  = [tf.stop_gradient(mr_d1), tf.stop_gradient(mr_d2), tf.stop_gradient(mr_d3), tf.stop_gradient( tf.roll(pt_d5,1,0))];
              j_mrpt_disc_out = self.mr_pt_disc(j_mrpt_disc_in, training=True);
              m_mrpt_disc_out = self.mr_pt_disc(m_mrpt_disc_in, training=True);

              tf.print('disc pred J out: ', j_mrpt_disc_out.shape);
              tf.print('disc  lab M out: ', m_mrpt_disc_out.shape);
              tf.print('mrpt disc pred 1 : mrpt disc pred 0');
              tf.print(tf.math.reduce_mean(j_mrpt_disc_out), ' : ', tf.math.reduce_mean(m_mrpt_disc_out))
              for i in range(len(j_mrpt_disc_out)):
                  tf.print(j_mrpt_disc_out[i], ' : ', m_mrpt_disc_out[i]);
              loss            = self.__compute_loss__(labels, [pred, j_disc_out, m_disc_out, j_mrpt_disc_out]  );
              mrpt_loss       = self.__compute_disc_loss__(j_mrpt_disc_out, m_mrpt_disc_out);
          gen_gradients, ptpt_gradients  = gen_tape.gradient(loss      , [self.base_model.trainable_variables, self.pt_pt_disc.trainable_variables]);
          mrpt_gradients                 = gen_tape.gradient(mrpt_loss, self.mr_pt_disc.trainable_variables);
          self.optimiser.apply_gradients(zip(gen_gradients , self.base_model.trainable_variables));
          self.ptpt_disc_optimiser.apply_gradients(zip(ptpt_gradients, self.pt_pt_disc.trainable_variables));
          self.mrpt_disc_optimiser.apply_gradients(zip(mrpt_gradients, self.mr_pt_disc.trainable_variables));
          return loss;

      def __get_batch__(self, start, batch_size):
          block  = self.data[start:start+batch_size];
          data   = block[:,:,:,:-1];
          label  = block[:,:,:,-1:];
          return data, label;

      def __distributed_training__(self, nof_epochs = 100):
          self.base_model.trainable = True;
          self.pt_pt_disc.trainable = True;
          self.__load_data__(self.data_directory, self.slice_per_batch);
          for i in range(nof_epochs):
              batch_number    = 0  ;
              disc_batch_no   = 0  ;
              epoch_loss      = 0.0;
              epoch_ptpt_loss = 0.0;
              k               = 0  ;
              self.batches_per_epoch = len(self.data)//self.batch_size
              self.data       = tf.random.shuffle(self.data);
              for j in range(self.batches_per_epoch):
                     start_point  = j*self.batch_size;
                     data, labels =  self.__get_batch__( start_point, self.batch_size);
                     step_loss    =  self.__training_step__(data, labels);
                     epoch_loss      += step_loss;
                     batch_number += 1;
                     tf.print('epoch ' + str(i) + '/' + str(nof_epochs) + ': batch ' + str(int(batch_number)) + '/' + str(self.batches_per_epoch) + ': Epoch Loss ', epoch_loss/batch_number , ': Step Loss ' , step_loss);
                     k+=1;
              if i%10 == 0:
                 self.checkpoint.save(self.checkpoint_directory + 'base_model' );
                 self.ptpt_disc_checkpoint.save(self.disc_checkpoint_directory + 'ptpt_disc');
                 self.mrpt_disc_checkpoint.save(self.disc_checkpoint_directory + 'mrpt_disc');
              else:
                 pass;

def main(data_directory, checkpoint_directory, disc_checkpoint_directory, slice_per_batch, kernel_patch_size, kernel_stride_factor, init_learning_rate = 0.001, checkpoint = None, batch_size = 12, nof_training_epochs = 100,  g_ptpt = 0.0, g_mrpt = 0.0):
    gpus       = tf.config.list_physical_devices('GPU');
    if gpus:
       try:
          tf.config.set_visible_devices([ gpus[1]  ],'GPU');
          tf.config.experimental.set_memory_growth(gpus[1], True);
          #tf.config.experimental.set_memory_growth(gpus[1], True);
          #tf.config.experimental.set_memory_growth(gpus[2], True);
          logical_gpus = tf.config.list_logical_devices('GPU');
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU");
       except RuntimeError as e:
          print(e)
    strategy = None;
    dist_trainer = Distributed_Trainer(strategy, DeepKernelNetwork, data_directory, checkpoint_directory, disc_checkpoint_directory, batch_size = batch_size); 
    dist_trainer.__compile_model_optimiser_checkpoint__( init_learning_rate = init_learning_rate, slice_per_batch = slice_per_batch, kernel_patch_size = kernel_patch_size, kernel_stride_factor = kernel_stride_factor, checkpoint = checkpoint, g_ptpt = g_ptpt, g_mrpt = g_mrpt);  
    dist_trainer.__distributed_training__(nof_epochs = nof_training_epochs);


import argparse
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory'                                    , type = str                    , help = 'Path to the data you want to train on.'                );
    parser.add_argument('checkpoint_directory'                              , type = str                    , help = 'Specify the directory to save checkpoints.'            );
    parser.add_argument('disc_checkpoint_directory'                         , type = str                    , help = 'Specify the directory to save disc checkpoints.'       );
    parser.add_argument('slice_per_batch'                                   , type = int                    , help = 'The number of slices to use per input.'                );
    parser.add_argument('kernel_patch_size'                                 , type = str                    , help = 'The patch sizes used to build the kernel matrices.'    );
    parser.add_argument('kernel_stride_factor'                              , type = str                    , help = 'The stride factor used for deep kernel matrices.'      );
    parser.add_argument('g_ptpt'                                            , type = float                  , help = 'PET PET info maximisation constraint hyperparameter.'  );
    parser.add_argument('g_mrpt'                                            , type = float                  , help = 'MR PET info minimisation constraint hyperparameter.'   );
    parser.add_argument('--batch_size'         , dest = 'batch_size'        , type = int  , default = 12    , help = 'Training batch size.'                                  );
    parser.add_argument('--nof_epochs'         , dest = 'nof_epochs'        , type = int  , default = 100   , help = 'Number of times to cycle through the training dataset.');
    parser.add_argument('--checkpoint'         , dest = 'checkpoint'        , type = str  , default = None  , help = 'Specify the path to previously saved checkpoint.'      );
    parser.add_argument('--init_learning_rate' , dest = 'init_learning_rate', type = float, default = 0.001 , help = 'Initial learning rate for training.'                   );
    args = parser.parse_args();
    args.kernel_patch_size    = n.array(args.kernel_patch_size.split(',')).astype(int)
    args.kernel_stride_factor = n.array(args.kernel_stride_factor.split(',')).astype(int)
    print('      data_directory: ', args.data_directory);
    print('checkpoint_directory: ', args.checkpoint_directory);
    print('     slice_per_batch: ', args.slice_per_batch);
    print('   kernel_patch_size: ', args.kernel_patch_size);
    print('kernel_stride_factor: ', args.kernel_stride_factor);
    print('              g_ptpt: ', args.g_ptpt);
    print('              g_mrpt: ', args.g_mrpt);
    print('          batch_size: ', args.batch_size);
    print('          nof_epochs: ', args.nof_epochs);
    print('          checkpoint: ', args.checkpoint);
    print('  init_learning_rate: ', args.init_learning_rate);
    main(args.data_directory, args.checkpoint_directory, args.disc_checkpoint_directory, args.slice_per_batch, args.kernel_patch_size, args.kernel_stride_factor, init_learning_rate = args.init_learning_rate, checkpoint = args.checkpoint,   batch_size = args.batch_size, nof_training_epochs = args.nof_epochs, g_mrpt = args.g_mrpt, g_ptpt = args.g_ptpt);
