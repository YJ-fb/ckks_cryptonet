#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from skimage.util import view_as_blocks
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from os import path


class Dataset:
    
    def __init__(self, verbosity = False):
        self.verbosity = verbosity
    
    def load(self, reduction = 1, base_dir = 'storage/datasets/'):
        """ Return the usable mnist dataset """
        
        if (self.saved_after(reduction, base_dir)):
            
            if self.verbosity :
                print("Loading local dataset (R" + str(reduction) + ") ...")
                
            train = np.load(base_dir +'train'+str(reduction)+'.npy')
            train_labels =  np.load(base_dir +'train_labels.npy')
            test = np.load(base_dir +'test'+str(reduction)+'.npy')
            test_labels = np.load(base_dir +'test_labels.npy')
            
            if self.verbosity:
                print("Dataset loaded with success!")
                print("")
                
            return (train, train_labels), (test, test_labels)
        
        if self.verbosity :
            print("Retrieving remote dataset...")
            
        (train, train_labels), (test, test_labels) = mnist.load_data()
        
        if self.verbosity :
            print("Dataset retrieved with success.")
            print("")
        
        train = self.downsample_dataset(train, reduction, True, base_dir +'train'+str(reduction))
        test = self.downsample_dataset(test, reduction, True, base_dir +'test'+str(reduction))
        
        train_labels = utils.to_categorical(train_labels, 10)
        test_labels = utils.to_categorical(test_labels, 10)

        if not self.label_saved():
            
            if self.verbosity :
                print('Saving labels...')
                
            np.save(base_dir +'test_labels', test_labels)
            np.save(base_dir +'train_labels', train_labels)
            
            if self.verbosity :
                print('Labels saved with success.')
        
        return (train, train_labels), (test, test_labels)

        
    
    def label_saved(self, base_dir = 'storage/datasets/'):
        """ Check if labels was saved before """
        
        return (path.exists(base_dir +'train_labels.npy') and 
                 path.exists(base_dir +'test_labels.npy'))
    
    def saved_after(self, reduction, base_dir = 'storage/datasets/'):
        """ Check if dataset was saved before """
        
        return ( path.exists(base_dir +'train'+str(reduction)+'.npy') and
                path.exists(base_dir +'test'+str(reduction)+'.npy') and
                self.label_saved())
    
    def pad_set(self, tpset, pad = (1,0), val = 0):
        result = []
        for i in range(tpset.__len__()):
            padded = np.lib.pad(tpset[i], pad, 'constant', constant_values=val)
            result.append(padded)
        return np.asarray(result)
    
    def downsample_dataset(self, dataset, size = 2, export = True, file = False):
        """ Apply the downsample function to each image in the dataset """
        
        if(export and path.exists(file+'.npy')):
            if self.verbosity :
                print('Loading file: "' + file + '" ...')
                
            return np.load(file+'.npy')
       
        dsampled = []
        
        for i in range(dataset.__len__()):
            dsampled.append(self.downsample(dataset[i],size))
        
        dsampled = self.pad_set(dsampled)
        dsampled = self.reshape(dsampled, dim=dsampled[0].__len__())

        
        if(file and export):
            if self.verbosity :
                print('Saving file: "' + file + '" ...')
                
            np.save(file, dsampled)
        
        return dsampled
    

    def downsample(self, img, size = 2):
        """ Apply the Average Pool (kernel = strides = (size,size)) 
            to the input image """
        
        if size < 2 or size >= img.__len__():
            return img
        
        view = view_as_blocks(img, (size,size))
        flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
        return np.mean(flatten_view, axis=2)
    
    def reshape(self, input_set, last_dimension = 1, astype = 'float32', sub = 255, dim = 28):
        """ Standardize the pixel values from int [0-255] to astype [0-1] """
        
        if last_dimension <= 0:
            if sub <= 1:
                return input_set.reshape(input_set.shape[0],dim,dim).astype(astype)
            else:
                return input_set.reshape(input_set.shape[0],dim,dim).astype(astype)/sub
        else:
            if sub <= 1:
                return input_set.reshape(input_set.shape[0],dim,dim,last_dimension).astype( astype)
            else:
                return input_set.reshape(input_set.shape[0],dim,dim,last_dimension).astype( astype)/sub

