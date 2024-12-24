#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from os import path
import timeit
import tenseal as ts
from tenseal.sealapi import CKKSEncoder, Encryptor

from modules.model import Model
from modules.util import createDir


class EncryptedNet:

    def __init__(self, test, test_labels, model,
                 n, t, precision, verbosity=False,
                 base_dir="storage/contexts/",
                 preprocess_dir="storage/layers/preprocessed/"):

        self.test = test
        self.test_labels = test_labels
        self.model = model
        self.n = n
        self.t = t
        self.precision = precision
        self.verbosity = verbosity

        self.prec = pow(10, precision)
        self.evaluate_start = None
        self.evaluate_end = None

        self.base_dir = base_dir
        self.preprocess_dir = preprocess_dir

        self.ctxt_dir = base_dir + "ctx_" + str(t) + "_" + str(n)
        self.enclayers_dir = self.ctxt_dir + "/layers/precision_" + str(precision)
        self.keys_dir = self.ctxt_dir + "/keys"

        self.generate_context()

        self.input_folder = self.enclayers_dir + "/input"
        self.conv_folder = self.enclayers_dir + "/conv"
        self.dense1_folder = self.enclayers_dir + "/dense1"
        self.dense2_folder = self.enclayers_dir + "/dense2"
        self.dense3_folder = self.enclayers_dir + "/dense3"
        self.out_folder = self.enclayers_dir + "/output"

        createDir(self.enclayers_dir)
        createDir(self.input_folder)
        createDir(self.conv_folder)
        createDir(self.dense1_folder)
        createDir(self.dense2_folder)
        createDir(self.dense3_folder)
        createDir(self.out_folder)

    def evaluate(self, get_acc=False):

        self.evaluate_start = timeit.default_timer()
        if self.verbosity:
            print("==============================================================")
            print("CLIENT")
            print("==============================================================")
        self.preprocess_input()

        if self.verbosity:
            print("==============================================================")
            print("SERVICE PROVIDER")
            print("==============================================================")

        self.convolution(15, 3, 2, (225,))
        self.dense1((7, 7, 5), (49, 5))
        self.dense2((100,))
        self.fully_connected((10,))

        accuracy = self.predict(self.test_labels, (10,), get_acc)

        if get_acc:
            m = Model()
            acc = m.getAccuracy(self.model, self.test, self.test_labels, self.n)
            print("Original Accuracy: " + str(acc) + "%")
            print("")

            if (acc == accuracy):
                print("EncryptedNet and Keras models give the same accuracy (about " +
                      str(round(accuracy)) + "%)")
            else:
                print("[ERR] EncryptedNet evaluation fails. The difference with " +
                      "Keras model is about " + str(round(acc - accuracy)) + "%")

    # ==========================================================================
    #  UTILITY FUNCTIONS
    # ==========================================================================

    def generate_context(self):
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_degree=4096, coeff_mod_bit_sizes=[60, 40, 40, 40])
        self.context.global_scale = 2 ** 40
        self.encoder = CKKSEncoder(self.context)
        self.encryptor = Encryptor(self.context)

    def store(self, arr, folder):
        arr = arr.flatten()

        if self.is_stored_before(folder):
            createDir(folder, True)

        for i in range(arr.__len__()):
            fname = folder + "/enc_" + str(i)
            arr[i].save(fname)

    def retrieve(self, folder, shape):
        if not self.is_stored_before(folder):
            raise Exception("Required files not found in folder " + folder)

        to_populate = np.empty(shape=shape)
        to_populate = to_populate.flatten()
        l = []

        for i in range(to_populate.__len__()):
            p = ts.plain_tensor(self.context)
            fname = folder + "/enc_" + str(i)
            if not path.exists(fname):
                raise Exception("File ", fname, "not exists")
            p.load(fname)
            l.append(p)

        return np.reshape(l, shape)

    def get_results(self):
        out_file = self.out_folder + "/results.npy"
        if not path.exists(out_file):
            raise Exception("Impossible to retrieve the result. Evaluation needed.")

        res = np.load(out_file)
        return res

    def is_stored_before(self, folder):
        fname = folder + "/enc_0"
        return path.exists(fname)

    # ==========================================================================
    #  ENCODING FOR HE
    # ==========================================================================
    def truncate_to(self, val, precision):
        tr = "{0:." + str(precision) + "f}"
        return float(tr.format(val))

    def truncate_tensor(self, tensor, precision):
        ish = tensor.shape
        tensor = tensor.flatten()
        res = []
        for el in tensor:
            res.append(self.truncate_to(el, precision))
        return np.array(res).reshape(ish)

    def check_encode(self, val):
        t = round(self.t / 2)
        t += 1
        if val >= t or val <= -t:
            return False
        return True

    def check_encode_tensor(self, tensor):
        tensor = tensor.flatten()
        t2 = round(self.t / 2)
        for el in tensor:
            if not self.check_encode(el):
                raise Exception("Value (" + str(el) + ") must be in [-" + str(t2) + ", +" + str(t2) + "] interval")

    def encode(self, val):
        val = round(val * self.prec)
        if self.check_encode(val):
            return int(val)
        else:
            t2 = round(self.t / 2)
            raise Exception("Value (" + str(val) + ") must be in [-" + str(t2) + ", +" + str(t2) + "] interval")

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

    def get_map(self, el):
        tns = [el] * self.n
        return self.encryptor.encrypt(self.encoder.encode(tns))

    # ==========================================================================
    #  GETTING NN W. AND B.
    # ==========================================================================

    def get_conv(self):
        layer = self.model.layers[0]
        w = layer.get_weights()[0]

        ew = np.empty(shape=(3, 3, 5))
        for i in range(3):
            for x in range(3):
                for y in range(5):
                    ew[i, x, y] = w[i, x, 0, y]

        b = layer.get_weights()[1]

        return ew, b

    def get_dense1(self):
        layer = self.model.layers[2]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w, b

    def get_dense2(self):
        layer = self.model.layers[3]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w, b

    def get_fully(self):
        layer = self.model.layers[4]
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        return w, b
    # =========================================================================
    #  INPUT
    # =========================================================================
    
    def preprocess_input(self):
        input_dim, dim, dim1, indx = self.test.shape
        pre_flat = self.test.flatten()
        pixel_arr_dim = dim*dim1
        
        if self.is_stored_before(self.input_folder):
            print("Input layer preprocessed before. You can found it in " + self.input_folder + " folder")
            return None
        
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
        print("Encoding input...")

        arr = self.encode_tensor(arr)
        
        print("Encrypting input...")
        
        out = []
        for el in arr:
            encoded = self.encoder.encode(el)
            encrypted = self.encryptor.encrypt(encoded)
            out.append(encrypted)

        out = np.array(out)
        self.store(out, self.input_folder)

        out = None
        
    
    # =========================================================================
    #  CONVOLUTION 
    # =========================================================================

    def _get_conv_window_(self, size, kernel):
        res = []
        x = 0
        for i in range(kernel):
            x = size * i
            for j in range(kernel):
                res.append(x + j)
        return res

    def _get_conv_indexes_(self, pixel, size, kernel, stride, padding=0):
        res = []
        output_size = int(((size - kernel + (2 * padding)) / stride)) + 1
        x = pixel
        for i in range(output_size):
            x = pixel + ((size * stride) * i)
            for j in range(output_size):
                res.append(x)
                x += stride
        return res

    def get_conv_map(self, size, kernel, stride):
        window = self._get_conv_window_(size, kernel)
        conv_map = []
        for i in range(window.__len__()):
            conv_map.append(self._get_conv_indexes_(window[i], size, kernel, stride))
        return np.array(conv_map)

    def get_conv_windows(self, size, kernel, stride):
        cm = self.get_conv_map(size, kernel, stride)
        windows = []
        for i in range(cm.shape[1]):
            w = []
            for j in range(cm.shape[0]):
                w.append(cm[j, i])
            windows.append(w)

        return np.array(windows)

    def convolution(self, size, kernel, stride, shape):

        if self.is_stored_before(self.conv_folder):
            print("Convolution layer processed before. You can found it in " + self.conv_folder + " folder")
            return None

        w, b = self.get_conv()

        w = self.encode_tensor(w)
        b = self.encode_tensor(b)

        cw = self.get_conv_windows(size, kernel, stride)

        w = w.reshape((w.shape[0] * w.shape[1], w.shape[2]))
        fw = np.empty(shape=(w.shape[1], w.shape[0]))

        out_shape = (cw.shape[0], fw.shape[0])

        inpt = self.retrieve(self.conv_folder, shape)

        print("Computing convolution...")
        print("Input shape:", inpt.shape)
        print("Output shape: ", out_shape)
        print("")

        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                fw[j, i] = w[i, j]

        res = []

        i = 0
        for f in fw:
            encb = self.get_map(b[i])
            for y in range(cw.shape[0]):
                local_sum = None
                for k in range(cw.shape[1]):
                    encw = self.get_map(f[k])
                    el = self.encryptor.multiply(inpt[cw[y, k]], encw)
                    if (local_sum == None):
                        local_sum = el
                    else:
                        local_sum = self.encryptor.add(local_sum, el)
                local_sum = self.encryptor.add(local_sum, encb)
                res.append(local_sum)
            i += 1

        res = np.array(res)

        lr = []
        for i in range(res.shape[0]):
            if (lr.__len__() < cw.shape[0]):
                lr.append([res[i]])
            else:
                inx = i % cw.shape[0]
                lr[inx].append(res[i])

        out = np.array(lr)
        self.store(out, self.conv_folder)
        out = None

    # ==========================================================================
    #  DENSE 1
    # =========================================================================

    def dense1(self, input_shape, shape):

        if self.is_stored_before(self.dense1_folder):
            print("First Dense layer processed before. You can found it in " + self.dense1_folder + " folder")
            return None

        w, b = self.get_dense1()

        w = self.encode_tensor(w)
        b = self.encode_tensor(b)

        per = input_shape[0] * input_shape[1]
        filters = input_shape[2]
        flat = per * filters

        if flat != w.shape[0]:
            raise Exception("Input shape " + str(input_shape) +
                            " is not compatible with preprocessed input " +
                            str(w.shape))

        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights " +
                            str(w.shape) + " and biases " + str(b.shape) +
                            " are incompatible.")

        out = []

        out_conv = self.retrieve(self.conv_folder, shape)

        print("Computing first dense...")
        print("Input shape: ", out_conv.shape)
        print("Output shape:", w.shape[1])
        print("")

        for x in range(w.shape[1]):
            local_sum = None
            for i in range(per):
                for j in range(filters):
                    row = ((i * filters) + j)

                    encw = self.get_map(w[row][x])
                    el = self.encryptor.multiply(out_conv[i, j], encw)

                    if (local_sum == None):
                        local_sum = el
                    else:
                        local_sum = self.encryptor.add(local_sum, el)

            enc_b = self.get_map(b[x])
            ts = self.encryptor.add(local_sum, enc_b)
            ts = self.encryptor.square(ts)
            out.append(ts)

        out = np.array(out)
        self.store(out, self.dense1_folder)
        out = None
            
    
    # =========================================================================
    #  DENSE 2
    # =========================================================================

    def dense2(self, shape):

        if self.is_stored_before(self.dense2_folder):
            print("Second Dense layer processed before. You can found it in " + self.dense2_folder + " folder")
            return None

        w, b = self.get_dense2()

        w = self.encode_tensor(w)
        b = self.encode_tensor(b)

        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights " +
                            str(w.shape) + " and biases " + str(b.shape) +
                            "are incompatible.")

        d1 = self.retrieve(self.dense1_folder, shape)

        print("Computing second dense...")
        print("Input shape: ", d1.shape)
        print("Output shape:", w.shape[1])
        print("")

        ind = 0

        for x in range(w.shape[1]):
            local_sum = None
            for i in range(w.shape[0]):
                encw = self.get_map(w[i][x])
                el = self.encryptor.multiply(d1[i], encw)

                if (local_sum == None):
                    local_sum = el
                else:
                    local_sum = self.encryptor.add(local_sum, el)

            enc_b = self.get_map(b[x])
            ts = self.encryptor.add(local_sum, enc_b, True)
            ts = self.encryptor.square(ts)
            fname = self.dense2_folder + "/enc_" + str(ind)
            ts.save(fname)
            local_sum = None
            ts = None
            ind += 1

        # SAVED BEFORE (memory issue)
        # out = np.array(out)
        # self.store(out, self.dense2_folder)
        # out = None
    
    # =========================================================================
    #  FULLY CONNECTED
    # =========================================================================
    
    def fully_connected(self, shape):
        
        if self.is_stored_before(self.dense3_folder):
            print("Third Dense layer processed before. You can found it in " + self.dense3_folder + " folder")
            return None
        
        w,b = self.get_fully()
        
        w = self.encode_tensor(w)
        b = self.encode_tensor(b)
        
        if w.shape[1] != b.shape[0]:
            raise Exception("Preprocessed weights "+
                            str(w.shape) +" and biases "+ str(b.shape) +
                            "are incopatible.")

        d2 = self.retrieve(self.dense2_folder, shape)
        
        print("Computing fully connected...")
        print("Input shape: ", d2.shape)
        print("Output shape:" , w.shape[1])
        print("")
        
        ind = 0
        
        for x in range(w.shape[1]):
            local_sum = None 
            for i in range(w.shape[0]):
                encw = self.get_map(w[i][x])
                el = self.encryptor.multiply(d2[i], encw)

                if local_sum is None:
                    local_sum = el
                else:
                    local_sum = self.encryptor.add(local_sum, el)

            enc_b = self.get_map(b[x])
            ts = self.encryptor.add(local_sum, enc_b)
            # out.append(ts)
            fname = self.dense3_folder + "/enc_"+str(ind)
            ts.save(fname)
            local_sum = None
            ts = None
            ind += 1
            
        # self.check_encode_tensor(out)    
        # out = np.append(out)
        # self.store(out, self.dense3_folder)
        # out = None
    
    
    # =========================================================================
    #  PREDICT / EVALUATE
    # =========================================================================

    def predict(self, test_labels, shape, get_evaluation=False):

        if path.exists(self.out_folder + "/results.npy"):
            print("Output processed before. You can found it in " + self.out_folder + "/results folder")
            return None
        else:
            print("Elaborating output...")

        fc = self.retrieve(self.dense3_folder, shape)

        out = []
        for el in fc:
            decrypted = self.encryptor.decrypt(el)
            decoded = self.encoder.decode(decrypted)
            out.append(decoded)

        fc = np.array(out)

        self.evaluate_end = timeit.default_timer()
        evt = round(self.evaluate_end - self.evaluate_start)

        print(str(fc.shape[1]) + "/" + str(fc.shape[1]) +
              " [==============================] - " + str(evt) + "s")

        el = []

        for i in range(test_labels.shape[0]):
            if len(el) <= i:
                el.append([])

            for j in range(len(fc[i])):
                if len(el[i]) <= j:
                    el[i].append(fc[i][j])
                else:
                    el[i][j].append(fc[i][j])

        el = np.array(el)

        print("El shape:", el.shape)

        np.save(self.out_folder + "/results.npy", el)

        if get_evaluation:
            print("================================")
            print(el[0])

            pos = 0

            for i in range(el.shape[0]):
                mp = np.argmax(el[i])
                ml = np.argmax(test_labels[i])
                if mp == ml:
                    pos += 1

            acc = (pos / self.n) * 100
            print("Model Accuracy: " + str(acc) + "% (" + str(pos) + "/" + str(test_labels.shape[0]) + ")")
            print("")

            return acc

        # ==========================================================================
        #  PREDICTED OUTPUT PER LAYER (FOR DEBUG PURPOSE)
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
