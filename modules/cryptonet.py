#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce
import timeit
import numpy as np
import pickle

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

    def train(self, epochs=10):
        """Train the model using the provided training data and save the trained model.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        """
        if self.verbosity:
            print("Training the model...")
        
        # Assuming that 'model' has a fit method compatible with Keras models
        self.model.fit(self.test, self.test_labels, epochs=epochs)

        if self.verbosity:
            print("Model trained successfully.")
            print("Saving the trained model...")

        # Save the trained model to file (implement saving logic here)

    def load_trained_model(self, filepath):
        """Load a pre-trained model from a file.

        Args:
            filepath (str): Path to the saved model file.
        """
        # Implement loading logic here
        pass
       
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
    def analyze_predictions(self):
        """Analyze prediction results and provide insights."""
        if self.verbosity:
            print("Analyzing predictions...")
    
        # Assuming predict() has been called and cn_pred is available
        cn_pred = np.argmax(self.crt_inverse([encryptor.get_results() for encryptor in self.encryptors], self.p_moduli), axis=1)
        l_pred = np.argmax(self.test_labels, axis=1)

        # Calculate per-class accuracy
        classes = range(10)  # Assuming 10 classes for MNIST dataset
        class_accuracy = {cls: {'correct': 0, 'total': 0} for cls in classes}
        
        for true_label, pred_label in zip(l_pred, cn_pred):
        class_accuracy[true_label]['total'] += 1
        if true_label == pred_label:
            class_accuracy[true_label]['correct'] += 1
    
        # Print per-class accuracy
        print("Class-wise Accuracy:")
        for cls in classes:
            total = class_accuracy[cls]['total']
            correct = class_accuracy[cls]['correct']
            acc = (correct / total) * 100 if total > 0 else 0
            print(f"Class {cls}: {acc:.2f}% ({correct}/{total})")
    
        # Identify most common misclassifications
        misclassifications = {}
        for true_label, pred_label in zip(l_pred, cn_pred):
            if true_label != pred_label:
                key = (true_label, pred_label)
                misclassifications[key] = misclassifications.get(key, 0) + 1
    
        sorted_misclassifications = sorted(misclassifications.items(), key=lambda item: item[1], reverse=True)[:5]
        print("\nTop 5 Most Common Misclassifications:")
        for (true_cls, pred_cls), count in sorted_misclassifications:
            print(f"True Class {true_cls} -> Predicted as {pred_cls}: {count} times")


    def save_encrypted_model(self, filepath):
        """Save the current state of the encrypted model to a file.

        Args:
            filepath (str): Path where the encrypted model will be saved.
        """
        if self.verbosity:
            print(f"Saving encrypted model to {filepath}...")

        with open(filepath, 'wb') as f:
            # Save necessary attributes
            state = {
                'p_moduli': self.p_moduli,
                'coeff_modulus': self.n,
                'precision': self.precision,
                'encryptors': [encryptor.save_state() for encryptor in self.encryptors],
                # Add other necessary states here
            }
            pickle.dump(state, f)

        if self.verbosity:
            print("Encrypted model saved successfully.")
