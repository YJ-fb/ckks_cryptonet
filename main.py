#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from modules.dataset import Dataset
from modules.model import Model
from modules.exporter import Exporter
from modules.cryptonet import Cryptonet
from modules.debug import Debug
import os
import numpy as np

# =============================================================================
# TRAINING parms
# =============================================================================

params = {
    'kernel_size': (3, 3),
    'strides': (2, 2),
    'last_activation': 'softmax',
    'optimizer': 'SGD',
    'loss': 'categorical_crossentropy',
    'batch_size': 200,
    'epochs': 50,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'nesterov': False,
    'use_dropout': False
}

# =============================================================================
# HE parms
# =============================================================================
p_moduli = [1099511922689, 1099512004609]  # list of t
coeff_mod = 8192  # n
precision = 2  # encoding precision

# =============================================================================
# GLOBAL parms
# =============================================================================
verbosity = True  # enable code verbosity
debug = False  # enable debug mode
debug_plain = False  # test the plain version
debug_encoded = False  # test the encoded version
debug_encrypted = True  # test the encrypted version

model_path = 'storage/models/model.h5'  # 模型保存路径

if debug:

    debug = Debug(p_moduli, coeff_mod, precision, "model", verbosity)

    if debug_plain:
        debug.test_plain_net()

    if debug_encoded:
        debug.test_encoded_net(0)

    if debug_encrypted:
        debug.test_encrypted_net(1)
else:
    ds = Dataset(verbosity=verbosity)
    (train, train_labels), (test, test_labels) = ds.load(2)

    # 创建 Model 实例
    model_instance = Model(input_shape=(15, 15, 1), params=params)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        if verbosity:
            print("Model not found. Training a new model...")

        # 训练模型
        history, trained_model = model_instance.fit(train, train_labels, test, test_labels)

        # 保存模型
        trained_model.save(model_path)
        if verbosity:
            print("Model trained and saved to", model_path)
    else:
        # 加载已有模型
        model_instance.load(model_path)

    exp = Exporter(verbosity=verbosity)
    # exp.exportBestOf(train, train_labels, test, test_labels, params, model_name="model15", num_test=10)

    test = test[:coeff_mod]
    test_labels = test_labels[:coeff_mod]

    cn = Cryptonet(test, test_labels, model_instance.model, p_moduli, coeff_mod, precision, True)
    cn.evaluate()

    m = Model()
    acc = model_instance.getAccuracy(test, test_labels)
    print("Original Accuracy: " + str(acc) + "%")