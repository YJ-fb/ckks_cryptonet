#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .model import Model
import timeit

class EncodeNet:
    
    def __init__(self,test, test_labels, model, n, t, precision, verbosity=False):
        self.test = test
        self.test_labels = test_labels
        self.model = model
        self.n = n
        self.precision = precision
        self.prec = pow(10, precision)
        self.evaluate_start = None
        self.evaluate_end = None
        self.verbosity = verbosity
        
        self.t = t
        
        print("Using precision: ", self.prec)
        print("Using T: ", self.t)
        print("")
        
    def evaluate(self):
        
        self.evaluate_start = timeit.default_timer()
        
        inpt = self.preprocess_input()
        cnv = self.convolution(15,3,2, inpt)
        d1 = self.dense1((7,7,5), cnv)
        d2 = self.dense2(d1)
        fc = self.fully_connected(d2)
        
        accuracy = self.predict(self.test_labels, fc)
        
        m = Model()
        acc = m.getAccuracy(self.model, self.test, self.test_labels, self.n)
        print("Original Accuracy: " + str(acc) + "%")
        print("")
        
        if(acc == accuracy):
            print("EncodeNet and Keras models give the same accuracy (about "+
                  str(round(accuracy))+"%)")
        else:
            print("[ERR] EncodeNet evaluation fails. The difference with "+
                  "Keras model is about " + str(round(acc-accuracy)) + "%")
    
        
    # =========================================================================
    #  ENCODING FOR HE
    # =========================================================================
    def truncate_to(self, val, precision):
        tr = "{0:."+str(precision)+"f}"
        return float(tr.format(val))
    
    def truncate_tensor(self, tensor, precision):
        ish = tensor.shape
        tensor = tensor.flatten()
        res = []
        for el in tensor:
            res.append(self.truncate_to(el, precision))
        return tensor.reshape(ish)
        
        
    def check_encode(self,val):
        t = round(self.t/2)
        t +=1
        if val >= t or val <= -t:
            return False
        return True
    
    def check_encode_tensor(self, tensor):
        tensor = tensor.flatten()
        t2 = round(self.t/2)
        for el in tensor:
            if not self.check_encode(el):
                raise Exception("Value ("+str(el)+") must be in [-"+str(t2)+", +"+str(t2)+"] interval")
    
    def encode(self, val):
        val = round(val * self.prec)
        if self.check_encode(val):
            return val
        else:
            t2 = round(self.t/2)
            raise Exception("Value ("+str(val)+") must be in [-"+str(t2)+", +"+str(t2)+"] interval")
    
            
    def encode_array(self, arr):
        ret = []
        for el in arr:
            ret.append(self.encode(el))
        return ret
    
    def encode_tensor(self, tensor):
        input_shape = tensor.shape
        tensor = tensor.flatten()
        tensor = np.array(self.encode_array(tensor))
        tensor = tensor.reshape(input_shape)
        return self.truncate_tensor(tensor, self.precision)
    
    
    # =========================================================================
    #  GETTING NN W. AND B.
    # =========================================================================
        
    def get_conv(self):
        layer = self.model.layers[0]
        w = layer.get_weights()[0]
        
        ew =  np.empty(shape=(3,3,5))
        for i in range(3):
            for x in range(3):
                for y in range(5):
                    ew[i, x, y] = w[i,x,0,y]
                    
        b = layer.get_weights()[1]
        
        return ew,b 
    
    def get_dense1(self):
        layer = self.model.layers[2]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w,b
    
    def get_dense2(self):
        layer = self.model.layers[3]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w,b
    
    def get_fully(self):
        layer = self.model.layers[4]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w,b
    
    # =========================================================================
    #  INPUT
    # =========================================================================
    
    def preprocess_input(self):
        input_dim, dim, dim1, indx = self.test.shape
        pre_flat = self.test.flatten()
        pixel_arr_dim = dim*dim1
        
        print("Processing input...")
        print("Input shape: ", self.test.shape)
        
        arr = []
            
        for x in range(pre_flat.__len__()):
            if x < pixel_arr_dim:
                arr.append([pre_flat[x]])
            else:
                pi = (x % pixel_arr_dim)
                arr[pi].append(pre_flat[x])
        
        arr = np.array(arr)
        print("Output shape:" , arr.shape)
        print("")
        
        arr = self.encode_tensor(arr)
        
        return arr
    
    # =========================================================================
    #  CONVOLUTION 
    # =========================================================================
    
    def _get_conv_window_(self, size, kernel):
        res = []
        x = 0
        for i in range(kernel):
            x = size * i
            for j in range(kernel):
                res.append(x+j)
        return res
    
    def _get_conv_indexes_(self, pixel, size, kernel, stride, padding = 0):
        res = []
        output_size = int(((size - kernel + (2 * padding))/stride)) + 1
        x = pixel
        for i in range(output_size):
            x = pixel + ((size * stride) * i)
            for j in range(output_size):
                res.append(x)
                x += stride
        return res
       
    def get_conv_map(self, size, kernel, stride):
        window = self._get_conv_window_(size,kernel)
        conv_map = []
        for i in range(window.__len__()):
            conv_map.append(self._get_conv_indexes_(window[i], size, kernel, 
                                                    stride))
        return np.array(conv_map)
    
    def get_conv_windows(self, size, kernel, stride):
        cm = self.get_conv_map(size,kernel,stride)
        windows = []
        for i in range(cm.shape[1]):
            w = []
            for j in range(cm.shape[0]):
                w.append(cm[j,i])
            windows.append(w)
        
        return np.array(windows)
    
    def get_map(self, el):
        return [el] * self.n
    
    def convolution(self, size, kernel, stride, inpt):
        w,b = self.get_conv()
        
        w = self.encode_tensor(w)
        b = self.encode_tensor(b)
        
        cw = self.get_conv_windows(size,kernel,stride)

        w = w.reshape((w.shape[0]*w.shape[1], w.shape[2]))
        fw = np.empty(shape=(w.shape[1], w.shape[0]))
        
        out_shape = (cw.shape[0], fw.shape[0], self.n)
        
        print("Computing convolution...")
        print("Input shape:", inpt.shape)
        print("Output shape: ", out_shape)
        print("")

        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                fw[j,i]  = w[i,j]

        res = []

        i = 0
        for f in fw:
            for y in range(cw.shape[0]):
                local_sum = []
                for k in range(cw.shape[1]):
                    encw = self.get_map(f[k])
                    el = np.multiply(inpt[cw[y,k]], encw)
                    if(len(local_sum) == 0):
                        local_sum = el
                    else:
                        local_sum = np.add(local_sum, el)
                local_sum = np.add(local_sum, b[i])
                res.append(local_sum)
            i+=1

        res = np.array(res)
    
        
        lr = []
        for i in range(res.shape[0]):
            if(lr.__len__() < cw.shape[0]):
                lr.append([res[i]])
            else:
                inx = i % cw.shape[0]
                lr[inx].append(res[i])
        
        out = np.array(lr)
        # self.check_encode_tensor(out)
        return out
    
    
    # =========================================================================
    #  DENSE 1
    # =========================================================================
    
    def dense1(self, input_shape, out_conv):
        
        w,b = self.get_dense1()
        
        w = self.encode_tensor(w)
        b = self.encode_tensor(b)
        
        per = input_shape[0] * input_shape[1]
        filters = input_shape[2]
        flat = per * filters
        
            
        if flat != w.shape[0] :
            raise Exception("Input shape " + str(input_shape) + 
                            " is not compatible with preprocessed input " + 
                            str(w.shape))
            
        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights "+
                            str(w.shape) +" and biases "+ str(b.shape) +
                            " are incopatible.")
        
        out = np.empty(shape=(w.shape[1], self.n))
        
        
        print("Computing first dense...")
        print("Input shape: ", out_conv.shape)
        print("Output shape:" , out.shape)
        print("")
                    
        
        for x in range(w.shape[1]):
            local_sum = []
            for i in range(per):
                for j in range(filters):
                    # fname = out_conv + "/" + str(i) + "_filter" + str(j)
                    row = ((i*filters) + j)
                
                    encw = self.get_map(w[row][x])
                
                    el = np.multiply(out_conv[i,j], encw)
                
                    if(len(local_sum) == 0):
                       local_sum = el
                    else:
                        local_sum = np.add(local_sum, el)
                    
            enc_b = self.get_map(b[x])
            ts = np.add(local_sum, enc_b)
            ts = np.square(ts)
            out[x] = ts
        
        # self.check_encode_tensor(out)
        return out
            
    
    # =========================================================================
    #  DENSE 2
    # =========================================================================
    
    def dense2(self, d1):
        
        w,b = self.get_dense2()
        
        w = self.encode_tensor(w)
        b = self.encode_tensor(b)
        
        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights "+
                            str(w.shape) +" and biases "+ str(b.shape) +
                            "are incopatible.")
 
        out = np.empty(shape=(w.shape[1], self.n))     
        
        print("Computing second dense...")
        print("Input shape: ", d1.shape)
        print("Output shape:" , out.shape)
        print("")
        
        for x in range(w.shape[1]):
            local_sum = [] 
            for i in range(w.shape[0]):
                # fname = input_folder + "/square_" + str(i)
                encw = self.get_map(w[i][x])
                el = np.multiply(d1[i], encw)
                
                if(len(local_sum) == 0):
                    local_sum = el
                else:
                    local_sum = np.add(local_sum, el)
            
            enc_b = self.get_map(b[x])
            ts = np.add(local_sum, enc_b)
            ts = np.square(ts)
            out[x] = ts
            
        # self.check_encode_tensor(out)
        return out
    
    # =========================================================================
    #  FULLY CONNECTED
    # =========================================================================
    
    def fully_connected(self, d2):
        w,b = self.get_fully()
        
        w = self.encode_tensor(w)
        b = self.encode_tensor(b)
        
        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights "+
                            str(w.shape) +" and biases "+ str(b.shape) +
                            "are incopatible.")
        
        out = np.empty(shape=(w.shape[1], self.n))
        
        print("Computing fully connected...")
        print("Input shape: ", d2.shape)
        print("Output shape:" , out.shape)
        print("")
        
        for x in range(w.shape[1]):
            local_sum = [] 
            for i in range(w.shape[0]):
                # fname = input_folder + "/square_" + str(i)
                encw = self.get_map(w[i][x])
                el = np.multiply(d2[i], encw)
                
                if(len(local_sum) == 0):
                    local_sum = el
                else:
                    local_sum = np.add(local_sum, el)
            
            enc_b = self.get_map(b[x])
            ts = np.add(local_sum, enc_b)
            out[x] = ts
            
        # self.check_encode_tensor(out)    
        return out
    
    
    # =========================================================================
    #  PREDICT / EVALUATE
    # =========================================================================
    
    def predict(self, test_labels, fc):
        
        test_labels = test_labels[:self.n]
        
        self.evaluate_end = timeit.default_timer()
        evt = round(self.evaluate_end - self.evaluate_start)
        
        print(str(fc.shape[1]) + "/" + str(fc.shape[1]) + 
              " [==============================] - " + str(evt)+"s")
        
        el = []
        
        for i in range(test_labels.shape[1]):
            # file = out_folder + "/fc_"+str(i)    
            if(el.__len__() <= i):
                el.append([])
                    
            for j in range(fc[i].__len__()):
                if(el.__len__() <= j):
                    el.append([fc[i][j]])
                else:
                    el[j].append(fc[i][j])
                    
        
        el = np.array(el) 
        
        print("================================")
        print(el[0])
        
        pos = 0
        
        for i in range(el.shape[0]):
            mp = np.argmax(el[i])
            ml = np.argmax(test_labels[i])
            if(mp == ml):
                pos+=1
                    
        acc = (pos/self.n) * 100
        print("Model Accuracy: " + str(acc) + "% ("+str(pos)+"/"+
              str(test_labels.shape[0])+")")
        print("")
        
        return acc
    
    
    # =========================================================================
    #  PREDICTED OUTPUT PER LAYER (FOR DEBUG PRUPOSE)
    # =========================================================================
    
    def get_conv_out(self):
        lm = self.model
        lm.outputs = [lm.layers[0].output]
        ye = lm.predict(self.test)
        return ye
    
    def get_dense1_out(self):
        lm = self.model
        lm.outputs = [lm.layers[2].output]
        ye = lm.predict(self.test)
        return ye
    
    def get_dense2_out(self):
        lm = self.model
        lm.outputs = [lm.layers[3].output]
        ye = lm.predict(self.test)
        return ye
    
    def get_fully_out(self):
        lm = self.model
        lm.outputs = [lm.layers[4].output]
        ye = lm.predict(self.test)
        return ye