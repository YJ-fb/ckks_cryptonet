#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from os import path
import timeit
import numpy as np
import tenseal as ts
from tenseal.sealapi import CKKSEncoder, Encryptor

from .util import createDir


class Encryption:

    def __init__(self,
                 verbosity=False,
                 p_modulus=1964769281,  # Plaintext modulus. All operations are modulo p. (t)
                 coeff_modulus=8192,  # Coefficient modulus (n)
                 batching=True,  # Set to true to enable batching
                 poly_base=2,  # Polynomial base (x)
                 security_level=128,  # Security level equivalent in AES. 128 or 192. (10 || 12 rounds)
                 int_digits=64,  # Truncated positions for integer part.
                 frac_digits=32,  # Truncated positions for fractional part.,
                 relin_keys_count=2,  # The number of relinKeys will be generated/restored
                 relin_bitcount=16,  # [1,60] bigger is faster but noiser
                 relin_size=4,  # |cxtx| = K+1 ==> size at least K-1
                 base_dir="storage/contexts/",
                 preprocess_dir="storage/layers/preprocessed/",
                 precision=4
                 ):

        self.verbosity = verbosity
        self.precision = precision

        self.t = p_modulus
        self.n = coeff_modulus
        self.batching = batching
        self.pbase = poly_base
        self.security = security_level
        self.idig = int_digits
        self.fdig = frac_digits
        self.relin_bits = relin_bitcount
        self.relin_size = relin_size

        # Setup TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.n,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.encoder = CKKSEncoder(self.context)
        self.encryptor = Encryptor(self.context)

        # Required directories
        self.preprocess_dir = preprocess_dir
        self.base_dir = base_dir
        self.ctxt_dir = base_dir + "ctx_" + str(self.t) + "_" + str(self.n)
        self.enclayers_dir = self.ctxt_dir + "/layers/precision_" + str(self.precision)
        self.keys_dir = self.ctxt_dir + "/keys"

        createDir(self.enclayers_dir)

        if self.verbosity:
            print("Created with success with the following parameters:")
            self.context_info()

    def context_info(self):
        """ Print the local context information """
        print("")
        print("Context parameters")
        print("============================")
        print("Batch encoding: " + str(self.context.is_batched()))
        print("Polynomial base: " + str(self.context.plain_modulus()))
        print("Frac digits: " + str(self.context.log_n()))
        print("Int digits: " + str(self.context.log_m()))
        print("Plaintext coeff (m): " + str(self.context.m()))
        print("Slots fitting in a ctxt: " + str(self.context.num_slots()))
        print("Plaintext modulus (p): " + str(self.context.plain_modulus()))
        print("Security level (AES): " + str(self.context.security_level()))
        print("")
        print("")

    def get_map(self, el):
        """ Encode and encrypt a single element """
        pt = ts.plain_tensor([el], [1])
        enc = self.encryptor.encrypt(pt)
        return enc

    def _enc_arr_(self, arr, file_name=None):
        """ Encode and encrypt an array """
        pt = ts.plain_tensor(arr, [len(arr)])
        enc = self.encryptor.encrypt(pt)
        if file_name:
            enc.save(file_name)
        return enc

    def _decode_(self, to_decode, t, precision):
        """ Decode the value encoded with _encode_ """
        t2 = t / 2
        if to_decode > t2:
            return (to_decode - t) / pow(10, precision)
        else:
            return to_decode / pow(10, precision)

    def decode_tensor(self, tensor, t, precision):
        ret = []
        for i in range(tensor.__len__()):
            ret.append(self._decode_(tensor[i], t, precision))

        return np.array(ret)

    def convolution(self, size, kernel, stride):
        if self.verbosity:
            print("Computing Convolution")
            print("==================================")

        conv_folder = self.enclayers_dir + "/conv"
        pre_conv = conv_folder + "/pre"
        out_conv = conv_folder + "/output"

        if not path.exists(conv_folder):
            createDir(conv_folder)

        conv_w = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_0_conv2d_3.npy"
        conv_b = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_bias_0_conv2d_3.npy"

        if path.exists(pre_conv):
            print("(Pre)processed before. You can found it in " +
                  pre_conv + " folder.")

        elif not path.exists(conv_w):

            print("Convolution weights need to be preprocessed before (with precision " +
                  str(self.precision) + ").")
            print("")
        else:
            createDir(pre_conv)

            filters = np.load(conv_w)

            start = timeit.default_timer()

            fshape = filters.shape
            f = filters.reshape((fshape[0] * fshape[1], fshape[2]))
            conv_map = self.get_conv_map(size, kernel, stride)

            if (conv_map.shape[0] != f.shape[0]):
                raise Exception("Convolution map and filter shapes must match.")

            if self.verbosity:
                print("Convolution: output preprocessing...")
                print("0%")

            for x in range(f.shape[0]):
                for y in range(f.shape[1]):
                    w_filter = self.get_map(f[x, y])
                    for k in range(conv_map.shape[1]):
                        enc_pixel = self.getEncryptedPixel(conv_map[x, k])
                        # computing |self.n| dot products at time
                        res = enc_pixel * w_filter  # 使用Tenseal的乘法
                        f_name = pre_conv + "/pixel" + str(conv_map[x, k]) + "_filter" + str(y)
                        res.save(f_name)
                if self.verbosity:
                    perc = int(((x + 1) / f.shape[0]) * 100)
                    print(str(perc) + "% (" + str(x + 1) + "/" + str(f.shape[0]) + ")")

            stop = timeit.default_timer()

            if self.verbosity:
                print("Convolution: output preprocessed in " + str(stop - start) + " s.")

        if path.exists(out_conv):

            print("Processed before. You can found it in " +
                  out_conv + " folder.")
            print("")

        elif not path.exists(conv_b):

            print("Convolution biases need to be preprocessed before (with precision " +
                  str(self.precision) + ").")
            print("")

        else:
            createDir(out_conv)

            biases = np.load(conv_b)

            start = timeit.default_timer()

            bshape = biases.shape
            windows = self.get_conv_windows(size, kernel, stride)
            wshape = windows.shape

            if self.verbosity:
                print("Convolution: output processing...")
                print("0%")

            for x in range(bshape[0]):
                encoded_bias = self.get_map(biases[x])
                for y in range(wshape[0]):
                    local_sum = None
                    for k in range(wshape[1]):
                        f_name = pre_conv + "/pixel" + str(windows[y, k]) + "_filter" + str(x)
                        p = ts.ckks_tensor(self.context)
                        p.load(f_name)
                        if (local_sum == None):
                            local_sum = p
                        else:
                            local_sum = local_sum + p  # 使用Tenseal的加法

                    local_sum = local_sum + encoded_bias  # 使用Tenseal的加法
                    file_name = out_conv + "/" + str(y) + "_filter" + str(x)
                    local_sum.save(file_name)

                if self.verbosity:
                    perc = int(((x + 1) / bshape[0]) * 100)
                    print(str(perc) + "% (" + str(x + 1) + "/" + str(bshape[0]) + ")")

            stop = timeit.default_timer()

            if self.verbosity:
                print("Convolution: output processed in " + str(stop - start) + " s.")
                print("")

        return out_conv

    def _get_conv_window_(self, size, kernel):
        """ Get the indices relative to the first convolutional window. """
        res = []
        x = 0
        for i in range(kernel):
            x = size * i
            for j in range(kernel):
                res.append(x + j)
        return res

    def _get_conv_indexes_(self, pixel, size, kernel, stride, padding=0):
        """ Slide the given index in the flatten volume returning all the indexes
        to which the same convolution filter must be applied. """
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
        """ Return the convolutional map of the input volume (given its width)
            according to the element index in its flatten version. """
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

    def getEncryptedPixel(self, index):
        """ Load an encrypted pixel from the input layer. """
        pixel_file = self.enclayers_dir + "/input/pixel_" + str(index) + ".seal"
        enc_pixel = ts.ckks_tensor(self.context)
        enc_pixel.load(pixel_file)
        return enc_pixel

    def preprocess(self, model, test_set):
        """ Start the preprocessing of the NN input and weights. """
        self._pre_process_input_(model, test_set)
        self._pre_process_layers_(model)

    def _pre_process_input_(self, model, test_set):
        """ Preprocess (encode) the input and save it in the preprocess directory. """
        input_size, input_dim, input_dim1, el_index = test_set.shape
        if input_size < self.n:
            raise Exception(
                "Too small input set. It must be at least " + str(self.n) + " len. " + str(input_size) + " given")

        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"
        createDir(base_dir, False)

        if path.exists(base_dir + 'pre_input.npy'):
            print("Input layer encoded before. You can found it in " + base_dir + 'pre_input.npy')
            return

        encoded_input = np.empty(shape=(input_size, input_dim, input_dim1), dtype=np.float32)

        if self.verbosity:
            print("Processing input...")
            print("Input shape: " + str(test_set.shape))
            print("Precision: " + str(self.precision))

        for i in range(input_size):
            for x in range(input_dim):
                for y in range(input_dim1):
                    encoded_input[i, x, y] = self._encode_(test_set[i, x, y, 0].item(), self.t, self.precision)

        if self.verbosity:
            print("Saving preprocessed input...")

        np.save(base_dir + "pre_input", encoded_input)

        if self.verbosity:
            print("Preprocessed input saved.")

    def _pre_process_layers_(self, model):
        """ Preprocess (encode) NN weights and biases. """
        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"
        createDir(base_dir, False)

        for i in range(model.layers.__len__()):
            self._pre_process_layer_(model.layers[i], i)

    def _pre_process_layer_(self, layer, index=0):
        """ Preprocess a single layer. """
        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"

        if self.verbosity:
            print("Processing the " + str(index) + "_" + str(layer.name) + " layer...")

        if path.exists(base_dir + "pre_" + str(index) + "_" + str(layer.name) + ".npy"):
            print("Layer preprocessed before. You can found it in " + base_dir + " folder.")
            return

        weights, biases = layer.get_weights()

        encoded_weights = np.empty(shape=weights.shape, dtype=np.float32)
        encoded_biases = np.empty(shape=biases.shape, dtype=np.float32)

        if self.verbosity:
            print("Weights tensor shape:  " + str(weights.shape))
            print("Biases tensor shape:  " + str(biases.shape))
            print("")

        for i in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                encoded_weights[i, x] = self._encode_(weights[i, x].item(), self.t, self.precision)

        for i in range(biases.shape[0]):
            encoded_biases[i] = self._encode_(biases[i].item(), self.t, self.precision)

        if self.verbosity:
            print("Weights and biases encoded with success.")

        np.save(base_dir + 'pre_' + str(index) + "_" + str(layer.name), encoded_weights)
        np.save(base_dir + 'pre_bias_' + str(index) + "_" + str(layer.name), encoded_biases)

        if self.verbosity:
            print("Layer " + str(layer.name) + "_" + str(index) + " weights and biases saved.")
            print("Layer preprocessing ends with success.")

    def dense1(self, input_shape):
        if self.verbosity:
            print("Computing First Dense (square)")
            print("==================================")

        dense_folder = self.enclayers_dir + "/dense1"
        out_folder = dense_folder + "/output"

        conv_folder = self.enclayers_dir + "/conv"
        out_conv = conv_folder + "/output"

        wfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_2_dense_9.npy"
        bfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_bias_2_dense_9.npy"

        if not path.exists(dense_folder):
            createDir(dense_folder)

        if path.exists(out_folder):
            print("Processed before. You can found it in " + out_folder + " folder.")
            print("")

        elif not path.exists(wfile) or not path.exists(bfile):
            raise Exception(
                "First dense layer weights and biases need to be preprocessed before (with precision " + str(
                    self.precision) + ").")

        elif not path.exists(out_conv):
            raise Exception("Convolution output required. Please run Encryption.convolution(...) before.")

        else:
            createDir(out_folder)

            w = np.load(wfile)
            b = np.load(bfile)

            start = timeit.default_timer()

            per = input_shape[0] * input_shape[1]
            filters = input_shape[2]

            flat = per * filters

            if flat != w.shape[0]:
                raise Exception(
                    "Input shape " + str(input_shape) + " is not compatible with preprocessed input " + str(w.shape))

            if w.shape[1] != b.shape[0]:
                raise Exception(
                    "Preprocessed weights " + str(w.shape) + " and biases " + str(b.shape) + " are incompatible.")

            if self.verbosity:
                print("First Dense: output processing...")
                print("0%")

            for x in range(w.shape[1]):
                local_sum = None
                for i in range(per):
                    for j in range(filters):
                        fname = out_conv + "/" + str(i) + "_filter" + str(j)
                        p = ts.ckks_tensor(self.context)
                        p.load(fname)
                        row = (i * filters + j)
                        encw = self.get_map(w[row][x])

                        el = p * encw  # 使用Tenseal的乘法

                        if local_sum is None:
                            local_sum = el
                        else:
                            local_sum = local_sum + el  # 使用Tenseal的加法

                enc_b = self.get_map(b[x])
                ts = local_sum + enc_b  # 使用Tenseal的加法
                ts = ts.square()  # 使用Tenseal的平方
                out_name = out_folder + "/square_" + str(x)
                ts.save(out_name)

                if self.verbosity:
                    perc = int(((x + 1) / w.shape[1]) * 100)
                    print(str(perc)+"% (" + str(x+1) + "/" + str(w.shape[1]) + ")")

                stop = timeit.default_timer()

                if self.verbosity:
                    print("First Dense: output processed in " + str(stop-start) + " s.")
                    print("")
                
    # =========================================================================
    # SECOND DENSE LAYER 
    # -------------------------------------------------------------------------
    # It is computed given the output files from first dense layer and the
    # weights (filters) and biases preprocessed from the model
    # =========================================================================         
    def dense2(self):
        if self.verbosity:
            print("Computing Second Dense (square)")
            print("==================================")

        input_folder = self.enclayers_dir + "/dense1/output"

        dense_folder = self.enclayers_dir + "/dense2"
        out_folder = dense_folder + "/output"

        wfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_3_dense_10.npy"
        bfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_bias_3_dense_10.npy"

        if not path.exists(dense_folder):
            createDir(dense_folder)

        if path.exists(out_folder):
            print("Processed before. You can found it in " +
                  out_folder + " folder.")
            print("")

        elif not path.exists(wfile) or not path.exists(bfile):
            raise Exception("Second dense layer weights and biases need to be preprocessed before (with precision " +
                            str(self.precision) + ").")

        elif not path.exists(input_folder):
            raise Exception("First dense output required. Please run Encryption.dense1(...) before.")

        else:
            createDir(out_folder)

            w = np.load(wfile)
            b = np.load(bfile)

            if w.shape[1] != b.shape[0]:
                raise Exception("Preprocessed weights " +
                                str(w.shape) + " and biases " + str(b.shape) +
                                "are incompatible.")

            if self.verbosity:
                print("Second Dense: output processing...")
                print("0%")

            start = timeit.default_timer()

            for x in range(w.shape[1]):
                local_sum = None
                for i in range(w.shape[0]):
                    fname = input_folder + "/square_" + str(i)
                    p = ts.ckks_tensor(self.context)
                    p.load(fname)
                    encw = self.get_map(w[i][x])
                    el = p * encw  # 使用Tenseal的乘法

                    if local_sum is None:
                        local_sum = el
                    else:
                        local_sum = local_sum + el  # 使用Tenseal的加法

                enc_b = self.get_map(b[x])
                ts = local_sum + enc_b  # 使用Tenseal的加法
                ts = ts.square()  # 使用Tenseal的平方
                out_name = out_folder + "/square_" + str(x)
                ts.save(out_name)

                if self.verbosity:
                    perc = int(((x + 1) / w.shape[1]) * 100)
                    print(str(perc) + "% (" + str(x + 1) + "/" + str(w.shape[1]) + ")")

            stop = timeit.default_timer()
            if self.verbosity:
                print("Second Dense: output processed in " + str(stop - start) + " s.")
                print("")

    def fully_connected(self):
        if self.verbosity:
            print("Computing Fully Connected")
            print("==================================")

        input_folder = self.enclayers_dir + "/dense2/output"

        fc_folder = self.enclayers_dir + "/fullyconnected"
        out_folder = fc_folder + "/output"

        wfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_4_dense_11.npy"
        bfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_bias_4_dense_11.npy"

        if not path.exists(fc_folder):
            createDir(fc_folder)

        if path.exists(out_folder):
            print("Processed before. You can found it in " +
                  out_folder + " folder.")
            print("")

        elif not path.exists(wfile) or not path.exists(bfile):
            raise Exception("Fully connected layer weights and biases need to be preprocessed before (with precision " +
                            str(self.precision) + ").")

        elif not path.exists(input_folder):
            raise Exception("Second dense output required. Please run Encryption.dense2(...) before.")

        else:
            createDir(out_folder)

            w = np.load(wfile)
            b = np.load(bfile)

            if w.shape[1] != b.shape[0]:
                raise Exception("Preprocessed weights " +
                                str(w.shape) + " and biases " + str(b.shape) +
                                "are incompatible.")

            if self.verbosity:
                print("Fully Connected: output processing...")
                print("0%")

            start = timeit.default_timer()

            for x in range(w.shape[1]):
                local_sum = None
                for i in range(w.shape[0]):
                    fname = input_folder + "/square_" + str(i)
                    p = ts.ckks_tensor(self.context)
                    p.load(fname)
                    encw = self.get_map(w[i][x])
                    el = p * encw  # 使用Tenseal的乘法

                    if local_sum is None:
                        local_sum = el
                    else:
                        local_sum = local_sum + el  # 使用Tenseal的加法

                enc_b = self.get_map(b[x])
                ts = local_sum + enc_b  # 使用Tenseal的加法
                out_name = out_folder + "/fc_" + str(x)
                ts.save(out_name)

                if self.verbosity:
                    perc = int(((x + 1) / w.shape[1]) * 100)
                    print(str(perc) + "% (" + str(x + 1) + "/" + str(w.shape[1]) + ")")

            stop = timeit.default_timer()
            if self.verbosity:
                print("Fully Connected: output processed in " + str(stop - start) + " s.")
                print("")


    # =========================================================================
    # FULLY CONNECTED LAYER 
    # -------------------------------------------------------------------------
    # It is computed given the output files from the second dense layer and the
    # weights (filters) and biases preprocessed from the model
    # =========================================================================         

    def fully_connected(self):
        if self.verbosity:
            print("Computing Fully Connected")
            print("==================================")

        input_folder = self.enclayers_dir + "/dense2/output"

        fc_folder = self.enclayers_dir + "/fullyconnected"
        out_folder = fc_folder + "/output"

        wfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_4_dense_11.npy"
        bfile = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_bias_4_dense_11.npy"

        if not path.exists(fc_folder):
            createDir(fc_folder)

        if path.exists(out_folder):
            print("Processed before. You can found it in " +
                  out_folder + " folder.")
            print("")

        elif not path.exists(wfile) or not path.exists(bfile):
            raise Exception("Fully connected layer weights and biases need to be preprocessed before (with precision " +
                            str(self.precision) + ").")

        elif not path.exists(input_folder):
            raise Exception("Second dense output required. Please run Encryption.dense2(...) before.")

        else:
            createDir(out_folder)

            w = np.load(wfile)
            b = np.load(bfile)

            if w.shape[1] != b.shape[0]:
                raise Exception("Preprocessed weights " +
                                str(w.shape) + " and biases " + str(b.shape) +
                                "are incompatible.")

            if self.verbosity:
                print("Fully Connected: output processing...")
                print("0%")

            start = timeit.default_timer()

            for x in range(w.shape[1]):
                local_sum = None
                for i in range(w.shape[0]):
                    fname = input_folder + "/square_" + str(i)
                    p = ts.ckks_tensor(self.context)
                    p.load(fname)
                    encw = self.get_map(w[i][x])
                    el = p * encw  # 使用Tenseal的乘法

                    if local_sum is None:
                        local_sum = el
                    else:
                        local_sum = local_sum + el  # 使用Tenseal的加法

                enc_b = self.get_map(b[x])
                ts = local_sum + enc_b  # 使用Tenseal的加法
                out_name = out_folder + "/fc_" + str(x)
                ts.save(out_name)

                if self.verbosity:
                    perc = int(((x + 1) / w.shape[1]) * 100)
                    print(str(perc) + "% (" + str(x + 1) + "/" + str(w.shape[1]) + ")")

            stop = timeit.default_timer()
            if self.verbosity:
                print("Fully Connected: output processed in " + str(stop - start) + " s.")
                print("")

    def get_results(self, test_labels):
        dense_folder = self.enclayers_dir + "/fullyconnected/output"
        el = []

        for i in range(test_labels.shape[1]):
            file = dense_folder + "/fc_" + str(i)
            enc = ts.ckks_tensor(self.context)
            enc.load(file)
            dec = enc.decrypt()  # 使用Tenseal的解密
            dec = self.decode_tensor(dec, self.t, self.precision)

            if len(el) <= i:
                el.append([])

            for j in range(dec.__len__()):
                if len(el[i]) <= j:
                    el[i].append(dec[j])
                else:
                    el[i][j].append(dec[j])

        el = np.array(el)
        return el

    def predict(self, test_labels):
        if self.verbosity:
            print("Computing Prediction")
            print("==================================")

        fc_folder = self.enclayers_dir + "/fullyconnected"
        out_folder = fc_folder + "/output"

        if not path.exists(out_folder):
            raise Exception("You need to compute the fully connected layer before.")

        print(test_labels[0])
        # Only q predictions are done simultaneously
        # for i in range(self.n)

        el = []

        start = timeit.default_timer()

        for i in range(test_labels.shape[1]):
            file = out_folder + "/fc_" + str(i)
            enc = ts.ckks_tensor(self.context)
            enc.load(file)
            dec = enc.decrypt()  # 使用Tenseal的解密
            dec = self.decode_tensor(dec, self.t, self.precision)

            if len(el) <= i:
                el.append([])

            for j in range(dec.__len__()):
                if len(el[i]) <= j:
                    el[i].append(dec[j])
                else:
                    el[i][j].append(dec[j])

        el = np.array(el)
        print(el.shape)
        print(el[0])
        pos = 0

        for i in range(el.shape[0]):
            mp = np.argmax(el[i])
            ml = np.argmax(test_labels[i])
            if mp == ml:
                pos += 1

        stop = timeit.default_timer()
        print("Computation time: " + str(stop - start) + " s.")
        print("Positive prediction: " + str(pos))
        print("Negative prediction: " + str(self.n - pos))
        acc = (pos / self.n) * 100
        print("Model Accuracy: " + str(acc) + "%")

    def _encode_(self, to_encode, t, precision):
        """ Check encode for the given value:
            Admitted intervals:
            + : [0, t/2]
            - : [(t/2)+1, t] ==> [-((t/2)+1), 0]

            Ex:
                positive: [0,982384640] ==> [0,982384640] ==> [0, t/2]
                negative: [-982384640, 0] ==> [982384641, 1964769281] ==> [(t/2)+1, t]
        """
        precision = pow(10, precision)
        val = round((to_encode * precision))
        t2 = t / 2

        if val < 0:
            minval = -(t2 + 1)
            if val < minval:
                raise Exception("The value to encode (" +
                                str(val) + ") is smaller than -((t/2)+1) = " +
                                str(minval))
            else:
                return (t + val)
        else:
            if val > t2:
                raise Exception("The value to encode (" +
                                str(val) + ") is larger than t/2 = " + str(t2))
            else:
                return val

    def _decode_(self, to_decode, t, precision):
        """ Decode the value encoded with _encode_ """
        t2 = t / 2
        if to_decode > t2:
            return (to_decode - t) / pow(10, precision)
        else:
            return to_decode / pow(10, precision)

    def decode_tensor(self, tensor, t, precision):
        ret = []
        for i in range(tensor.__len__()):
            ret.append(self._decode_(tensor[i], t, precision))

        return np.array(ret)

    def encrypt_input(self, get_result=False):
        """ Encrypt the input layer generating one file per
        encrypted pixel index """

        pre_input_file = self.preprocess_dir + "precision_" + str(self.precision) + "/pre_input.npy"

        if not path.exists(pre_input_file):
            raise Exception("Preprocessed input not found in " + pre_input_file +
                            " please run Encryption.preprocess before.")

        input_folder = self.enclayers_dir + "/input"

        if path.exists(input_folder):
            print("Input layer encrypted before. You can found it in: " + input_folder)
            if not get_result:
                return None

        createDir(input_folder)
        pre_input = np.load(self.preprocess_dir + "precision_" +
                            str(self.precision) + "/pre_input.npy")

        if self.verbosity:
            print("")
            print("Encrypting (preprocessed) input layer with shape " +
                  str(pre_input.shape) + "...")

        input_dim, dim, dim1 = pre_input.shape
        pre_flat = pre_input.flatten()
        arr = []
        pixel_arr_dim = dim * dim1

        for x in range(pre_flat.__len__()):
            if x < pixel_arr_dim:
                arr.append([pre_flat[x]])
            else:
                arr[(x % pixel_arr_dim)].append(pre_flat[x])

        arr = np.array(arr)

        enc = []
        for i in range(arr.shape[0]):
            fname = input_folder + '/pixel_' + str(i) + ".seal"
            enc.append(self._enc_arr_(arr[i], fname))

        if self.verbosity:
            print("Input layer encrypted with success in " +
                  str(enc.__len__())+" files (one per pixel)")

        return np.array(enc)

    def getEncryptedPixel(self, index):
        pixel_file = self.enclayers_dir + "/input/pixel_" + str(index) + ".seal"

        enc_pixel = ts.ckks_tensor(self.context)
        enc_pixel.load(pixel_file)
        return enc_pixel

    def encode_arr(self, arr):
        pt = ts.plain_tensor(arr, [len(arr)])

        enc = self.encryptor.encrypt(pt)
        return enc

    def enc_arr(self, arr, file_name=None):
        pt = ts.plain_tensor(arr, [len(arr)])

        enc = self.encryptor.encrypt(pt)
        if file_name:
            enc.save(file_name)
        return enc

    def preprocess(self, model, test_set):
        """ Start the preprocessing of the NN input and weights """

        self.pre_process_input(model, test_set)
        self.pre_process_layers(model)

    def pre_process_input(self, model, test_set):
        """ Preprocess (encode) the input and save it in the preprocess directory """

        input_size, input_dim, input_dim1, el_index = test_set.shape
        if input_size < self.n:
            raise Exception("Too small input set. It must be at least " + str(self.n) + " len. " + str(input_size) + " given")

        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"
        createDir(base_dir, False)

        if path.exists(base_dir + 'pre_input.npy'):
            print("")
            print("Input layer encoded before. You can found it in " +
                  base_dir + 'pre_input.npy')
            print("")

        else:
            encoded_input = np.empty(shape=(input_size, input_dim, input_dim1), dtype=np.float32)

            if self.verbosity:
                print("")
                print("Processing input...")
                print("=====================================================")
                print("Input shape: " + str(test_set.shape))
                print("Precision: " + str(self.precision))
                print("")

        for i in range(input_size):
            for x in range(input_dim):
                for y in range(input_dim1):
                    encoded_input[i, x, y] = self._encode_(test_set[i, x, y, 0].item(), self.t, self.precision)

        if self.verbosity:
            print("Saving preprocessed input...")

        print("Input shape: " + str(encoded_input.shape))

        np.save(base_dir + "pre_input", encoded_input)

        if self.verbosity:
            print("Preprocessed input saved.")

            encoded_input = None

    def pre_process_layers(self, model):
        """ Preprocess (encode) NN weights and biases """

        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"
        createDir(base_dir, False)
        for i in range(model.layers.__len__()):
            self._pre_process_layer_(model.layers[i], i)

    def pre_process_layer(self, layer, index=0):
        """ Preprocess a single layer """

        base_dir = self.preprocess_dir + "precision_" + str(self.precision) + "/"
        if self.verbosity:
            print("")
            print("Processing the " + str(index) + "_" + str(layer.name) + " layer...")
            print("=========================================================")

        if path.exists(base_dir + "pre_" + str(index) + "_" + str(layer.name) + ".npy"):
            print("Layer preprocessed before. You can found it in " + base_dir + " folder.")
            return

        else:
            weights, biases = layer.get_weights()

            encoded_weights = np.empty(shape=weights.shape, dtype=np.float32)
            encoded_biases = np.empty(shape=biases.shape, dtype=np.float32)

            if self.verbosity:
                print("Weights tensor shape:  " + str(weights.shape))
                print("Biases tensor shape:  " + str(biases.shape))
                print("")

            for i in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    encoded_weights[i, x] = self._encode_(weights[i, x].item(), self.t, self.precision)

            for i in range(biases.shape[0]):
                encoded_biases[i] = self._encode_(biases[i].item(), self.t, self.precision)

            if self.verbosity:
                print("Weights and biases encoded with success.")

            np.save(base_dir + 'pre_' + str(index) + "_" + str(layer.name), encoded_weights)
            np.save(base_dir + 'pre_bias_' + str(index) + "_" + str(layer.name), encoded_biases)

            if self.verbosity:
                print("Layer " + str(layer.name) + "_" + str(index) + " weights and biases saved.")
                print("")
                print("Layer precomputation ends with success.")
                print("")
    
