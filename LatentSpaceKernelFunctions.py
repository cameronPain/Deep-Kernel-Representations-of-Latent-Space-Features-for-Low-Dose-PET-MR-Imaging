
#!/usr/bin/env python3
# Conv2D
# Generate projection data.
#
from tensorflow.keras.layers import Conv2D, Conv3D, UpSampling2D, Concatenate, Input, MaxPooling2D, Activation, BatchNormalization, Dense, add, Dropout, LeakyReLU, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
import tensorflow as tf
import numpy as n
########################################################################################
#
#
#
#    Latent-space kernel functions for low-dose PET MR imaging robust to varying dose reduction
#         author: Cameron Dennis Pain: 20240802
#    affiliation: Monash Biomedical Imaging, Monash University
#          email: cameron.pain@monash.edu.au
#
#   This method represents latent space feature maps using kernel functions derived from MR
#   to provide explicit regularisation for improved performance across a varying range of 
#   dose reduction factors.
#
#
#   This file we define all our trainable layers/model classes.
#   
#    DeepKernelNetwork() takes low-dose PET and MR inputs. 
#
#    At each layer of the DeepKernelNetwork, we calculate kernel functions and code vectors.
#
#    We calculate kernel features and pass these to a decoder branch to recover
#    the target image.
#
########################################################################################

class MR_PT_Feature_Discriminator(tf.keras.Model):
    def __init__(self, name = 'MR PT Feature Discriminator'):
        super().__init__();
        self.d1  = down_block(nof_feature_maps = 8);
        self.d2  = down_block(nof_feature_maps = 16);
        self.d3  = down_block(nof_feature_maps = 32);
        self.d4  = down_block(nof_feature_maps = 64);
        self.d5  = down_block(nof_feature_maps = 64);
        self.fc0 = Dense(16);
        self.bn0 = BatchNormalization();
        self.a0  = Activation('relu');
        self.fc1 = Dense(8);
        self.bn1 = BatchNormalization();
        self.a1  = Activation('relu');
        self.fc2 = Dense(1);
        self.a2  = Activation('sigmoid');
        self.GAP  = GlobalAveragePooling2D();


    def call(self, x):
        mr_feat0, mr_feat1, mr_feat2, pt_feat = x;
        pt_feat = tf.tile(pt_feat, (1,2,2,1));
        print('mr0: ',mr_feat0.shape);
        print('mr1: ',mr_feat1.shape);
        print('mr2: ',mr_feat2.shape);
        print('pt5: ',pt_feat.shape);
        #Try and learn the deep PET features from shallow kernel features. If we can, then the model is encoding 
        #Shallow accessable information into the deep layers. We want to minimise this so that we utilise shallow features as a priority.
        #This will avoid learning valid representations in-distribution which do not translate as well to out-of-distribution
        #As we hypothesise that deep representations are less robust to a shift in input domain. ;
        f,d1  = self.d1(mr_feat0);
        d1    = tf.concat([d1, mr_feat1],axis=-1);
        f,d2  = self.d2(d1);
        d2    = tf.concat([d2, mr_feat2],axis=-1)
        f,d3  = self.d3(d2);
        d3  = tf.concat([d3, pt_feat],axis=-1);
        f,d4  = self.d4(d3);
        f,d5  = self.d5(d4);
        d6  = self.GAP(d5);
        d7  = self.fc0(d6);
        d7  = self.bn0(d7);
        d7  = self.a0(d7);
        d8  = self.fc1(d7);
        d8  = self.bn1(d8);
        d8  = self.a1(d8);
        d9  = self.fc2(d8);
        d9  = self.a2(d9);
        return d9;


class PT_PT_Feature_Discriminator(tf.keras.Model):
    def __init__(self, name = 'MR PT Feature Discriminator', shallow_patch_size= (12,12)):
        super().__init__();

        self.d1  = down_block(nof_feature_maps = 8);
        self.d2  = down_block(nof_feature_maps = 16);
        self.d3  = down_block(nof_feature_maps = 32);
        self.d4  = down_block(nof_feature_maps = 64, pool_size=(3,3));

        self.fc0 = Dense(16);
        self.bn0 = BatchNormalization();
        self.a0  = Activation('relu');
        self.fc1 = Dense(8);
        self.bn1 = BatchNormalization();
        self.a1  = Activation('relu');
        self.fc2 = Dense(1);
        self.bn2 = BatchNormalization();
        self.a2  = Activation('sigmoid');
        self.GAP = GlobalAveragePooling2D();

    def call(self, x):
        #Try and match deep pt feature with each patch from the shallow pt feature.  
        #This will push the pet feature encoder towards encoding information 
        #globally across the image. This will prevent small anatomical regions becoming 
        #encoded which will likely be corrupted at larger dose reduction factors.
        #The MR PT constraint pushes the model towards using shallow layers where possible.
        #The PT PT constraint forces the model to learn a more dose-robust representation in deep layers.
        f, d1 = self.d1(x);
        f, d2 = self.d2(d1);
        f, d3 = self.d3(d2);
        f, d4 = self.d4(d3);

        d4 = self.GAP(d4);
        d4 = self.fc0(d4);
        d4 = self.bn0(d4);
        d4 = self.a0(d4);
        d5 = self.fc1(d4);
        d5 = self.bn1(d5);
        d5 = self.a1(d5);
        d6 = self.fc2(d5);
        d6 = self.bn2(d6);
        d6 = self.a2(d6);
        return d6;

class B_Kernel_Function_Generator(tf.keras.layers.Layer):
    def __init__(self, name = 'Image basis generator', nof_input_channels = 16, nof_conv_blocks = 4, Nb = 4):
        super().__init__();
        if nof_conv_blocks > 1:
            convblocks = [conv_block(nof_input_channels * Nb) for i in range(nof_conv_blocks-1)];
        else:
            convblocks = [];
        convblocks.append(conv_block(Nb*nof_input_channels, activation = None));
        self.output_activation            = Activation('softmax');
        self.convblocks                   = convblocks;
        self.nof_input_channels           = nof_input_channels;
        self.Nb                           = Nb;
        self.nof_conv_blocks              = nof_conv_blocks;        

    def call(self,x):
        #####
        #
        # We want to rearrange it so the channels go along the batch axis.
        #
        #####
        b, h, w, c = x.shape;
        #reshape so the channels go along the batch axis.
        for i in range(self.nof_conv_blocks):
            x = self.convblocks[i](x);
        ib,ih,iw,icib = x.shape;
        x = tf.reshape(x, shape = (-1,h,w,self.nof_input_channels, self.Nb));
        x = self.output_activation(x);
        #We mask these contributions to avoid exploding gradients;
        mask = tf.cast((x > 1e-5),dtype=n.float32)
        x = mask * x; 
        return x;

class Sparse_Kernel_Matrix_Multiplication(tf.keras.layers.Layer):
    def __init__(self, name ='Feature combiner'):
        super().__init__();

    def align_patches(self, images, patch_size, strides):
        px, py          = patch_size;
        sx, sy          = strides;
        lx, ly          = (px//sx), (py//sy);
        NSx,NSy,z,h,w,c = images.shape;
        rollx, rolly    = n.arange( -px//2 + sx//2, px//2 + sx//2, sx ), n.arange( -py//2 + sy//2, py//2 + sy//2, sy );
        rx, ry          = n.meshgrid(rollx,rolly);
        r_idx           = n.arange(lx*ly).reshape((1,lx,ly));
        r               = n.concatenate([r_idx, n.expand_dims(rx,0),n.expand_dims(ry,0)],axis=0)
        r               = tf.reshape(r, shape = (3,lx*ly));
        r               = tf.transpose(r, perm = (1,0));
        rev             = tf.transpose(images, perm  = (2,3,4,5,0,1));
        rev             = tf.reshape(rev  , shape = (z,h,w,c,lx*ly));
        rev             = tf.transpose(rev, perm  = (4,0,1,2,3));
        unrolled_image  = tf.map_fn(lambda i: tf.roll(rev[i[0],:,:,:,:], shift = (i[1],i[2]), axis=(1,2)), r, dtype=n.float32)
        return unrolled_image;

    def call(self,x, patch_size = (8,8), strides = (8,8)):
        mr_features, pet_features, b_features = x;
        nof_cols       = b_features.shape[-2];
        nof_rows       = b_features.shape[-1];
        b,h,w,c        = pet_features.shape
        bb,bh,bw,bc,bf = b_features.shape;
        px,py          = patch_size;
        #############
        #We perform a patch-wise matrix multiplication.
        #############
        sx,sy          = strides;
        Nx,Ny          = h//sx , w//sy;
        b_features     = tf.reshape(b_features, shape = (-1,bh,bw,bc*bf));
        b_patches      = tf.image.extract_patches(b_features  , (1,px,py,1), (1,sx,sy,1), (1,1,1,1), "SAME");
        pet_patches    = tf.image.extract_patches(pet_features, (1,px,py,1), (1,sx,sy,1), (1,1,1,1), "SAME");
        Nx,Ny          = (h//px), (w//py);
        lx, ly         = (px//sx), (py//sy);
        SNx,SNy        = Nx*lx, Ny*ly;
        b_patches      = tf.reshape(b_patches, shape = (-1,SNx,SNy,px,py,bc,bf));
        pet_patches    = tf.reshape(pet_patches, shape = (-1,SNx,SNy,px,py,bc,1));
        diagonal_norm  = tf.math.reduce_sum(b_patches, axis=(3,4),keepdims=True);
        kernel_pet     = tf.math.reduce_sum(pet_patches * b_patches, axis= (3,4),keepdims=True);
        kernelpet_norm = tf.math.divide_no_nan(kernel_pet, diagonal_norm); #(-1,h,w,1,1,c,bf);
        kernelpet_norm = kernelpet_norm * b_patches;
        kernel_pet     = tf.math.reduce_sum( tf.reshape(kernelpet_norm, shape = (-1,SNx,SNy,px,py,bc,bf)), axis = -1, keepdims=False);
        l_sx, l_sy     = n.meshgrid(range(lx), range(ly));
        l_s            = n.concatenate([n.expand_dims(l_sx,0), n.expand_dims(l_sy,0)],0);
        l_s            = tf.reshape(l_s,shape = (2,lx*ly));
        l_s            = tf.transpose(l_s,perm=(1,0))
        kernel_pet_    = tf.map_fn(lambda i: kernel_pet[:,i[0]::lx,i[1]::ly], l_s, dtype=n.float32)
        kernel_pet_    = tf.transpose(kernel_pet_, perm = (1,2,3,4,5,6,0))
        kernel_pet_    = tf.reshape(kernel_pet_, shape = (b, Nx,Ny,px,py,c, lx,ly));
        kernel_pet_    = tf.transpose(kernel_pet_, perm = (0,1,3,2,4,5,6,7));
        kernel_pet_    = tf.reshape(kernel_pet_, shape = (b, Nx*px,Ny*py,c,lx,ly));
        kernel_pet_    = tf.transpose(kernel_pet_, perm = (4,5,0,1,2,3));
        kernel_pet     = self.align_patches(kernel_pet_, patch_size, strides);
        kPET           = tf.math.reduce_mean(kernel_pet, axis=0, keepdims=False);
        return kPET;

class Deep_Kernel_Representation(tf.keras.layers.Layer):
    def __init__(self, name = 'Contrast mixing block', nof_input_channels = 16, Nb = 4,  nof_conv_blocks = 2):
        super().__init__();
        self.b_kernel_function_generator         = B_Kernel_Function_Generator(nof_input_channels = nof_input_channels, Nb = Nb, nof_conv_blocks = nof_conv_blocks);
        self.sparse_kernel_matrix_multiplication = Sparse_Kernel_Matrix_Multiplication();

    def call(self, x, patch_size = (24,24), strides = (1,1)):
        mr_features, pet_features = x;
        b_features                = self.b_kernel_function_generator(mr_features);
        kernel_features           = self.sparse_kernel_matrix_multiplication([mr_features, pet_features, b_features], patch_size = patch_size, strides = strides);
        return kernel_features; 

class DeepKernelNetwork(tf.keras.Model):
    def __init__(self, PET_Dropout_Rate = 0.0, kernel_stride_factor = (4,2,1,1,1,1), kernel_patch_size = (8,4,2,1,1,1), slice_per_batch = 3):
        super().__init__();
        self.patch_size           = kernel_patch_size;
        self.slice_per_batch      = slice_per_batch;
        self.kernel_stride_factor = kernel_stride_factor;

        self.pt_d1    = down_block_average(nof_feature_maps = 8  , dropout_rate = 0.0);
        self.pt_d2    = down_block_average(nof_feature_maps = 16 , dropout_rate = 0.0);
        self.pt_d3    = down_block_average(nof_feature_maps = 32 , dropout_rate = 0.0);
        self.pt_d4    = down_block_average(nof_feature_maps = 64 , dropout_rate = 0.0);
        self.pt_d5    = down_block_average(nof_feature_maps = 128, dropout_rate = 0.0);

        self.mr_d1    = down_block(nof_feature_maps = 8  , dropout_rate = 0.);
        self.mr_d2    = down_block(nof_feature_maps = 16 , dropout_rate = 0.);
        self.mr_d3    = down_block(nof_feature_maps = 32 , dropout_rate = 0.);
        self.mr_d4    = down_block(nof_feature_maps = 64 , dropout_rate = 0.);
        self.mr_d5    = down_block(nof_feature_maps = 128, dropout_rate = 0.);
 
        self.kernel1  = Deep_Kernel_Representation(nof_input_channels = 8   , Nb = 4, nof_conv_blocks = 2);
        self.kernel2  = Deep_Kernel_Representation(nof_input_channels = 16  , Nb = 4, nof_conv_blocks = 2);
        self.kernel3  = Deep_Kernel_Representation(nof_input_channels = 32  , Nb = 4, nof_conv_blocks = 2);
        self.kernel4  = Deep_Kernel_Representation(nof_input_channels = 64  , Nb = 4, nof_conv_blocks = 2);
        self.kernel5  = Deep_Kernel_Representation(nof_input_channels = 128 , Nb = 4, nof_conv_blocks = 2);
        
        self.u0       = double_conv_block(nof_feature_maps=256, dropout_rate = 0.);
        self.u1       = up_block(nof_feature_maps=128, dropout_rate = 0.);
        self.u2       = up_block(nof_feature_maps=64, dropout_rate = 0.);
        self.u3       = up_block(nof_feature_maps=32, dropout_rate = 0.);
        self.u4       = up_block(nof_feature_maps=16, dropout_rate = 0.);
        self.u5       = up_block(nof_feature_maps=8, dropout_rate = 0.);
        self.out_conv = Conv2D(1, (3,3), padding='same',strides=1);
        self.relu     = Activation('relu');

    def call(self,x):
        #Modality 1 is PET
        #Modality 2 is MR;
        #tf.print('');
        pt       = x[:,:,:,:self.slice_per_batch];
        mr       = x[:,:,:,self.slice_per_batch:];

        #Mixer 1
        pt_f1, pt_d1 = self.pt_d1(x);
        mr_f1, mr_d1 = self.mr_d1(mr);
        k_f1         = self.kernel1([mr_d1, pt_d1], patch_size=(self.patch_size[0], self.patch_size[0]),  strides = (self.kernel_stride_factor[0],self.kernel_stride_factor[0]));#
        k_f1_        = k_f1;
        pt_d1        = tf.concat([pt_d1, tf.stop_gradient(mr_d1)],axis=-1);
        k_f1         = tf.concat([k_f1, mr_d1],axis=-1);
         
        #Mixer 2
        pt_f2, pt_d2 = self.pt_d2(pt_d1);
        mr_f2, mr_d2 = self.mr_d2(mr_d1);
        k_f2         = self.kernel2([mr_d2, pt_d2], patch_size=(self.patch_size[1], self.patch_size[1]),  strides =  (self.kernel_stride_factor[1],self.kernel_stride_factor[1]) );#
        k_f2_        = k_f2;
        pt_d2        = tf.concat([pt_d2, tf.stop_gradient(mr_d2)],axis=-1);
        k_f2         = tf.concat([k_f2, mr_d2],axis=-1);

        #Mixer 3
        pt_f3, pt_d3 = self.pt_d3(pt_d2);
        mr_f3, mr_d3 = self.mr_d3(mr_d2);
        k_f3         = self.kernel3([mr_d3, pt_d3], patch_size=(self.patch_size[2], self.patch_size[2]),  strides =  (self.kernel_stride_factor[2],self.kernel_stride_factor[2]));#
        k_f3_        = k_f3;
        pt_d3        = tf.concat([pt_d3, tf.stop_gradient(mr_d3)],axis=-1);
        k_f3         = tf.concat([k_f3, mr_d3],axis=-1);

        #Mixer 4
        pt_f4, pt_d4 = self.pt_d4(pt_d3);
        mr_f4, mr_d4 = self.mr_d4(mr_d3);
        k_f4         = self.kernel4([mr_d4, pt_d4], patch_size=(self.patch_size[3], self.patch_size[3]),  strides =  (self.kernel_stride_factor[3],self.kernel_stride_factor[3]))#
        pt_d4        = tf.concat([pt_d4, tf.stop_gradient(mr_d4)],axis=-1);
        k_f4         = tf.concat([k_f4, mr_d4],axis=-1);

        #Mixer 5
        pt_f5, pt_d5 = self.pt_d5(pt_d4);
        mr_f5, mr_d5 = self.mr_d5(mr_d4);
        k_f5         = self.kernel5([mr_d5,pt_d5], patch_size=(self.patch_size[4], self.patch_size[4]),  strides =  (self.kernel_stride_factor[4],self.kernel_stride_factor[4]))#
        pt_d5        = tf.concat([pt_d5, tf.stop_gradient(mr_d5)],axis=-1);
        k_f5         = tf.concat([k_f5, mr_d5],axis=-1);

        u0     = self.u0(k_f5);
        u1     = self.u1([u0,k_f4]);
        u2     = self.u2([u1,k_f3]);
        u3     = self.u3([u2,k_f2]);
        u4     = self.u4([u3,k_f1]);
        u5     = self.u5([u4,mr_f1]);
        output = self.out_conv(u5);
        output = self.relu(output);
        return output, pt_d1, pt_d3, pt_d5, k_f1_, k_f2_, k_f3_;

class conv_block(tf.keras.layers.Layer):
     def __init__(self, nof_feature_maps, kernel_size=3, strides=1, padding='same', activation = 'relu', dropout = 0.0, activity_regularizer = None):
         super().__init__();
         self.c = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding, activity_regularizer = activity_regularizer);
         self.b = BatchNormalization();
         self.a = Activation(activation);
         self.d = Dropout(dropout);
         self.use_activation = activation;
     def call(self, x):
         x = self.c(x);
         x = self.b(x);
         if self.use_activation:
             x = self.a(x);
         else:
             pass;
         x = self.d(x);
         return x

class down_block_average(tf.keras.layers.Layer):
     def __init__(self, nof_feature_maps = 16, kernel_size=3, strides=1, padding='same', pool_size=(2,2), dropout_rate = 0.0):
         super().__init__();
         self.c1 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b1 = BatchNormalization();
         self.a1 = Activation('relu');
         self.dp1 = Dropout(dropout_rate);
         self.c2 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b2 = BatchNormalization();
         self.a2 = Activation('relu');
         self.dp2 = Dropout(0.0);
         self.d1 = AveragePooling2D(pool_size = pool_size);

     def call(self, x):
         x = self.c1(x);
         x = self.b1(x);
         x = self.a1(x);
         x = self.dp1(x);
         x = self.c2(x);
         x = self.b2(x);
         x = self.a2(x);
         x = self.dp2(x);
         d = self.d1(x);
         return x,d;

class down_block(tf.keras.layers.Layer):
     def __init__(self, nof_feature_maps = 16, kernel_size=3, strides=1, padding='same', pool_size=(2,2), dropout_rate = 0.0):
         super().__init__();
         self.c1 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b1 = BatchNormalization();
         self.a1 = Activation('relu');
         self.dp1 = Dropout(dropout_rate);
         self.c2 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b2 = BatchNormalization();
         self.a2 = Activation('relu');
         self.dp2 = Dropout(0.0);
         self.d1 = MaxPool2D(pool_size = pool_size);
          
     def call(self, x):
         x = self.c1(x);
         x = self.b1(x);
         x = self.a1(x);
         x = self.dp1(x);
         x = self.c2(x);
         x = self.b2(x);
         x = self.a2(x); 
         x = self.dp2(x);
         d = self.d1(x);
         return x,d;

class up_block(tf.keras.layers.Layer):
     def __init__(self, nof_feature_maps = 16, kernel_size=3, strides = 1, padding = 'same', upsample_size=(2,2), dropout_rate = 0.0):
         super().__init__();
         self.u1     = UpSampling2D(upsample_size, interpolation='bilinear');
         self.concat = Concatenate();
         self.c1     = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b1     = BatchNormalization();
         self.a1     = Activation('relu');
         self.dp1    = Dropout(dropout_rate);
         self.c2     = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b2     = BatchNormalization();
         self.a2     = Activation('relu');

     def call(self,x):
         feats, upsamples = x;
         u = self.u1(feats);
         x = self.concat([u,upsamples]);
         x = self.c1(x);
         x = self.b1(x);
         x = self.a1(x);
         x = self.dp1(x);
         x = self.c2(x);
         x = self.b2(x); 
         x = self.a2(x); 
         return x;

class double_conv_block(tf.keras.layers.Layer):
     def __init__(self, nof_feature_maps = 16, kernel_size=3, padding='same', strides = 1, dropout_rate = 0.0):
         super().__init__();
         self.c1 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b1 = BatchNormalization();
         self.a1 = Activation('relu');
         self.dp1 = Dropout(dropout_rate);
         self.c2 = Conv2D(nof_feature_maps, kernel_size, strides = strides, padding=padding);
         self.b2 = BatchNormalization();
         self.a2 = Activation('relu');
         self.dp2 = Dropout(dropout_rate);

     def call(self,x):
         x = self.c1(x);
         x = self.b1(x);
         x = self.a1(x);
         x = self.dp1(x); 
         x = self.c2(x); 
         x = self.b2(x);
         x = self.a2(x);
         x = self.dp2(x);
         return x;
