#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce
import timeit
import numpy as np

from .encryptednet import EncryptedNet

class Cryptonet:
    
    def __init__(self, 
            test, test_labels,
            model,
            p_moduli = [], # Plaintext modulus. All operations are modulo t. (t)
            coeff_modulus = 8192, # Coefficient modulus (n)
            precision = 2,
            verbosity = False
    ):
    """Initialize the Cryptonet class with necessary parameters.

                Args:
                    test (numpy.ndarray): Test dataset.
                    test_labels (numpy.ndarray): Labels for the test dataset.
                    model (keras.Model or similar): The neural network model to evaluate.
                    p_moduli (list, optional): List of plaintext moduli. Defaults to [].
                    coeff_modulus (int, optional): Coefficient modulus value. Defaults to 8192.
                    precision (int, optional): Precision for floating-point calculations. Defaults to 2.
                    verbosity (bool, optional): Flag to control verbose output. Defaults to False.
    """
        self.verbosity = verbosity
        self.p_moduli = p_moduli
        self.n = coeff_modulus
        self.precision = precision
        self.test = test
        self.test_labels = test_labels
        self.model = model
        
        self.encryptors = []
        for i in range(p_moduli.__len__()):
           self.encryptors.append(EncryptedNet(test, test_labels, model, coeff_modulus, p_moduli[i], precision, False))
    
       
    def evaluate(self):
        for i in range(self.encryptors.__len__()):
            self.encryptors[i].evaluate(False)
            
        self.predict()
            
    def predict(self):
        if self.verbosity:
            print("Computing Prediction")
            print("==================================")
        
        results = []
        for i in range(self.encryptors.__len__()):
            results.append(self.encryptors[i].get_results())
        
        results = np.array(results)
        
        cn_pred = self.crt_inverse(results, self.p_moduli)
        cn_pred = np.argmax(cn_pred, axis=1)
        l_pred = np.argmax(self.test_labels, axis=1)
        
        pos = 0
        neg = 0
        for i in range(cn_pred.shape[0]):
            if(l_pred[i] == cn_pred[i]):
                pos +=1
            else:
                neg +=1
        
        tot = pos + neg
        print("Total predictions: " + str(tot))
        print("Positive predictions: " + str(pos))
        print("Negative predictions: " + str(neg))
        print("===============================================")
        acc = (pos/tot) * 100
        loss = (neg/tot) * 100
        print("Model Accurancy: " + str(acc) + "%")
        print("Model Loss: " + str(loss) + "%")
            
            
    def get_product(self, arr):
        prod = 1
        for i in range(arr.__len__()):
            prod = prod * arr[i]
        return prod
    
    def crt_inverse(self, arr, p_moduli):
        tprod = self.get_product(p_moduli)
        tprod2 = tprod // 2 
        res = []
        for i in range(arr.shape[1]):
            res.append([])
            for j in range(10):
                res[i].append([])
                val = self.crt([arr[0,i,j],arr[1,i,j]] ,p_moduli)
                if(val > tprod2):
                    res[i][j] = val - tprod
                else:
                    res[i][j] = val
        return np.array(res)
    
    def crt(self, arr, p_moduli):

        t_prod = []
        t_coef = []
        tprod = self.get_product(p_moduli)
        
        for i in range(p_moduli.__len__()):
            t_prod.append(tprod // p_moduli[i])
            t_coef.append(self.mul_inv(t_prod[i], p_moduli[i]))
            
        res = 0
        
        for i in range(arr.__len__()):
            res += arr[i] * t_coef[i] * t_prod[i]
        
        return res % tprod
    
    def mul_inv(self, a, b):
        b0 = b
        x0, x1 = 0, 1
        if b == 1: return 1
        while a > 1:
            q = a // b
            a, b = b, a%b
            x0, x1 = x1 - q * x0, x0
        if x1 < 0: x1 += b0
        return x1
