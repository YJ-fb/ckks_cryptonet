#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import utils as tfku
import numpy as np
import os
import glob
import pandas as pd 

import talos
from CryptoNet import mnist_model
from CryptoNet import util

verbosity = True # 全局变量，控制输出的详细程度

class Experiment:
    
    name = None
    params = None
    verbose = False # 控制类实例是否打印详细信息
    
    def __init__(self, name, params, verbose = False):
        # 初始化相关参数
        self.name = name
        self.params = params
        self.verbose = verbose
        
    
    def computeResult(self):
        os.chdir("all_results")
        files = [i for i in glob.glob('*.{}'.format('csv'))]
        final = pd.concat([pd.read_csv(f) for f in files ])
        final.to_csv( "final.csv", index=False, encoding='utf-8-sig')
    
    def plotResults(self, scan_object = None, analyze_file = None):
        analyze_object = None
        
        if(scan_object != None):
            analyze_object = talos.Analyze(scan_object)         
        
        if(analyze_file != None):
            analyze_object = talos.Reporting(analyze_file)
        
        if(analyze_object == None):
            pass
        else:
            print("Results:")
            print(analyze_object.data)
            print("")
            
            print("Rounds:")
            print(analyze_object.rounds())
            print("")
            
            print("Highest accuracy:")
            print(analyze_object.high('val_acc'))
            print("")
            
            print("Lowest (not null) loss:")
            print(analyze_object.low('val_loss'))
            print("")
            
            print("Round with best results (val_acc):")
            print(analyze_object.rounds2high('val_acc'))
            print("")

            best_params = analyze_object.best_params('val_acc', [])
            print("Best parameters (val_acc) rank:")
            print(best_params)
            print("")
            
            print("Best params:")
            print(best_params[0])
            print("")
            
            print("Best parameters (val_loss) rank:")
            print(analyze_object.best_params('val_loss', ['acc', 'loss', 'val_loss']))
            print("")
            
            # line plot
            analyze_object.plot_line('val_acc')
            
            # line plot
            analyze_object.plot_line('val_loss')
            
            # a regression plot for two dimensions 
            analyze_object.plot_regs('val_acc', 'val_loss')
            
            # up to two dimensional kernel density estimator
            analyze_object.plot_kde('val_acc')
            
            # up to two dimensional kernel density estimator
            analyze_object.plot_kde('val_loss')
            
            # a simple histogram)
            analyze_object.plot_hist('val_acc', bins=40)
            
            
            
    def run(self):
        dataset = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

        # standardize the pixel values from 0-255 to 0-1
        train_images = util.reshapeSet(train_images)
        test_images = util.reshapeSet(test_images)
                
        train_labels = tfku.to_categorical(train_labels, 10)
        test_labels = tfku.to_categorical(test_labels, 10)
                
        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        
        if(self.verbose):
            print("Train set size: %s" % str(train_images.shape))
            print("Train set labels size: %s" % str(train_labels.shape))
            print("Test set size: %s" % str(test_images.shape))
            print("Test set labels size: %s" % str(test_labels.shape))
            print("")
            print("First train object <%s>" % train_labels[0])
            print("First test object <%s>" % test_labels[0])
            print("")
        
        return talos.Scan(
            train_images, train_labels, 
            model=mnist_model, params=self.params,
            x_val = test_images, y_val = test_labels,
            experiment_name = self.name)
        
        
lr = []
for i in range(10):
    lr.append(0.01)

p = {'last_activation': ['softmax'],
          'optimizer': ['SGD'],
          'loss': ['categorical_crossentropy'],
          'batch_size': [200],
          'epochs': [50],
          'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
          'learning_rate': lr}

exp = Experiment(verbose = True, params = p, name = 'DenseDropout01')
#exp.plotResults(exp.run())
#exp.plotResults(None, 'BestEvaluation03/113019213503.csv')
#exp.computeResult()
exp.plotResults(None, "all_results/final.csv")

