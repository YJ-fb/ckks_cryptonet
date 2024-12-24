#./usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals
from .plainnet import PlainNet
from .encodenet import EncodeNet
from .encryptednet import EncryptedNet
from .exporter import Exporter
from .dataset import Dataset



class Debug:
    
    def __init__(self, pmoduli, n, precision, model_name, verbosity=False):
        """ Initialize the deubug class """
        self.exporter = Exporter(verbosity = verbosity)
        self.dataset = Dataset(verbosity = verbosity)
        self.verbosity = verbosity
        self.model_name = model_name
        self.tlist = pmoduli
        self.n = n
        self.precision = precision
    
    
    def test_plain_net(self):
        """ Evaluate the plain version of the neural network comparing
            the result with the pre-trained model """
        plainnet = self.buildPlainNet()
        acc = plainnet.evaluate()
        print("PlainNet acc:", acc)
    
    
    def test_encoded_net(self, t_index = 0):
        """ Evaluate the encoded version of the neural network comparing
        the result with the pre-trained model """
        encodenet = self.buildEncodeNet(t_index)
        acc = encodenet.evaluate()
        print("EncodeNet acc:", acc)
        
        
    def test_encrypted_net(self, t_index = 0):
        """ Evaluate the encrypted version of the neural network comparing
        the result with the pre-trained model """
        encodenet = self.buildEncryptedNet(t_index)
        acc = encodenet.evaluate(True)
        print("EncodeNet acc:", acc)
     
        
    def buildPlainNet(self):
        """ Build the plain version of the Neural Network """
        m,t,tl = self.getParams()
        return PlainNet(t,tl,m, self.n, self.verbosity)    
    
    
    def buildEncodeNet(self, t_index = 0):
        """ Build the encoded version of the Neural Network """
        m,t,tl = self.getParams()
        return EncodeNet(t,tl, m, self.n, self.tlist[t_index], self.precision,
                         self.verbosity)
    
    def buildEncryptedNet(self, t_index = 0):
        """ Build the encrypted version of the Neural Network """
        m,t,tl = self.getParams()
        return EncryptedNet(t,tl, m, self.n, self.tlist[t_index], self.precision,
                         self.verbosity)
    
    def getParams(self):
        """ Return zipped model, test set and relative labels """
        model = self.exporter.load(model_name = self.model_name)
        (train, train_labels), (test, test_labels) = self.dataset.load(2)
        test_labels = test_labels[:self.n]
        test = test[:self.n]
        
        return model,test, test_labels
    