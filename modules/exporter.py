#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marzio Monticelli (1459333)
"""

from os import path
from . import model as md
from tensorflow import keras



class Exporter:
    
    def __init__(self, verbosity = False):
        self.verbosity = verbosity
        
    
    def load(self, base_dir = 'storage/models/', model_name = 'model'):
        model_path = base_dir+model_name+".hdf5"
        
        if path.exists(model_path):
            
            if(self.verbosity) :
                print("Loading saved model "+model_path+"...")
                
            model = keras.models.load_model(model_path)
            
            
            if(self.verbosity) :
                print("Loaded with success")
                
            return model
        
        else:
            
            return False
    
    def export(self, model, accuracy_filter = 98, 
               test = None, test_labels = None, 
               base_dir = 'storage/models/', model_name = 'model'):
        
        model_path = base_dir+model_name+".hdf5"
        
        if(accuracy_filter > 0):
            m = md.Model()
            score_acc_val = m.getAccuracy(model, test, test_labels)
            
            if self.verbosity :
                print("Model accuracy: " + str(score_acc_val))
                
            if score_acc_val >= accuracy_filter :
                model.save(model_path)
                
                if self.verbosity: 
                    print("Model " + model_path + " exported with success.")
                    
            elif (self.verbosity) : 
                print("Model not exported. Accuracy ("+str(score_acc_val)+"%) lower than " +
                      str(accuracy_filter) + "%.")
        else:
            model.save(model_path)
            
            if self.verbosity: 
                    print("Model " + model_path + " exported with success.")
    
    def exportBestOf(self, train, train_labels, test, test_labels, params,
                     base_dir = 'storage/models/', model_name = 'model',
                     accuracy_filter = 98,
                     num_test = 10
                     ):
        
        if num_test > 1 :
            print("")
            print("================================================================")
            print("Saving the best model in " + str(num_test) + " runs...")
            print("================================================================")
            
            m = md.Model()
            
            model = self.load(base_dir, model_name)
            if model == False:
                (h, model) = m.fit(train, train_labels, test, test_labels,params)
                self.export(model,-1, None, None, base_dir, model_name)
            
            score_acc_val = m.getAccuracy(model, test, test_labels)
            print("Model accuracy: " + str(score_acc_val))
            
            for i in range(num_test):
                step = i+1
                if self.verbosity:
                    print("")
                    print("Step " + str(step) + "/" + str(num_test) + 
                          " (" +str((step*100)//num_test) + "%" +")" )
                    print("")
                    
                (h,model) = m.fit(train, train_labels, test, test_labels,params)
                
                saved_model = self.load(base_dir, model_name)
                saved_score_acc_val = m.getAccuracy(saved_model,test, test_labels)

                self.export(model, saved_score_acc_val, test, test_labels, base_dir, model_name)
            
            print("")
            print("================================================================")
            print("Process completed !")
            print("================================================================")
            
        else:
           print("Error. Use Exporter.export instead.")
    