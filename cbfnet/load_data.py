'''
This file contains utilities for organizing data and loading into a
TF dataset. 

The function get_dataset takes as input filenames of numpy files sorted
by the function organize_filenames.
'''

from nilearn.image import load_img
import numpy as np
import pandas as pd
import tensorflow as tf

def organize_filenames(file):
    ''' Organize filenames stored in "file" txt into sorted
    X, y lists. '''
    
    with open(file) as f:
        names = [line.rstrip('\n') for line in f]
    X_files = sorted([name for name in names if 'asl' in name])
    y_files = list(zip(sorted([name for name in names if 'CBF' in name]),
                  sorted([name for name in names if 'ATT' in name])))
    
    return X_files, y_files

def numpy_from_nifti(in_files, crop):
    ''' Load a numpy array from a list of nifti filenames.'''

    image_list = []

    for file in in_files:
        x, y, z = crop
        image = load_img(file).get_fdata()
    
        if x:
            image = image[x:-x,...]
        if y:
            image = image[:,y:-y,...]
        if z:
            image = image[...,z:-z,:]
    
        image_list.append(image)
    image_array = np.asarray(image_list)
    return image_array

def get_dataset(X, y, batch_size=1, crop_size=[0,0,0], 
                moments=None, mask_num=0, include_all_masked=False,
                shuffle=True):
    ''' Get tf.Dataset from lists of nifti filenames. '''
    
    def mask_plds(image, channels=1, num2remove=0, include_all=False):
        ''' Mask the input PLDs. Option to mask all up to and including
        num2remove, or just the specified num.'''

        images = []
        if include_all:
            for i in num2remove: 
                sample_size = channels-i
                idx = tf.Variable([3,2,4,1,5,0])[:sample_size]
                mask = tf.one_hot(idx, depth=6, on_value=1.0,
                                  off_value=0.0)
                mask = tf.reduce_sum(mask,0)
                image = tf.cast(image, tf.float32)*mask
                images.append(image)
        else:
            sample_size = channels-num2remove
            idx = tf.Variable([3,2,4,1,5,0])[:sample_size]
            mask = tf.one_hot(idx, depth=6, on_value=1.0, 
                              off_value=0.0)
            mask = tf.reduce_sum(mask,0)
            image = tf.cast(image, tf.float32)*mask
            images.append(image)

        print('img_len', len(images))
        return tf.data.Dataset.from_tensor_slices(images)
    
    # Setup tensorflow dataset from generator for memory efficient processing
    X_ds = tf.data.Dataset.from_tensor_slices(
        numpy_from_nifti(X, crop=crop_size))
    y_ds = tf.data.Dataset.from_tensor_slices(
        numpy_from_nifti(y, crop=crop_size))

    # Get scaling info for normalization
    moments = pd.read_csv(moments, index_col=0)

    X_range = moments.loc['Xtrain','range']
    y_range = [moments.loc['y_cbf','range'],moments.loc['y_att','range']]

    # Scale data to approx 0->1 and combine (X,y) into train, test datasets
    X_scale = X_ds.map(lambda x: x/X_range)
    X_ds_mask = X_scale.map(lambda x: mask_plds(x, channels=6, 
                                                num2remove=mask_num,
                                                include_all=include_all_masked))
    def flatten(*x):
        return tf.data.Dataset.from_tensor_slices([i for i in x])
    X_ds_mask = X_ds_mask.flat_map(lambda x: x)
    y_ds_scale = y_ds.map(lambda y: y/y_range)
    if include_all_masked:
        y_ds_scale = y_ds_scale.map(lambda y: [[y,]*len(mask_num)])
        y_ds_scale = y_ds_scale.flat_map(lambda y: tf.data.Dataset.from_tensor_slices(y))
    if shuffle:
        ds = tf.data.Dataset.zip((X_ds_mask,y_ds_scale)).shuffle(10000).batch(batch_size)
    else:
        ds = tf.data.Dataset.zip((X_ds_mask,y_ds_scale)).batch(batch_size)

    return ds
