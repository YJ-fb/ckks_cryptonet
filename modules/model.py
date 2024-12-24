from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras import utils
import os

def square_activation(x):
    return x ** 2

utils.get_custom_objects().update({'square_activation': Activation(square_activation)})

class Model:

    def __init__(self, input_shape=None, params=None):
        self.input_shape = input_shape
        self.params = params or {
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'last_activation': 'softmax',
            'optimizer': 'SGD',
            'loss': 'categorical_crossentropy',
            'batch_size': 200,
            'epochs': 50,
            'dropout': 0.2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'nesterov': False,
            'use_dropout': True
        }
        self.model = None

    def getAccuracy(self, x_val, y_val, batch_size=None):
        if self.model is not None:
            score = self.model.evaluate(x_val, y_val, verbose=1, batch_size=batch_size)
            return score[1] * 100
        else:
            raise ValueError("Model has not been trained or loaded yet.")

    def fit(self, x_train, y_train, x_val, y_val):
        """ 训练模型 """

        print("")
        print("Model params:")
        print("")
        print("Input shape: " + str(self.input_shape))
        print("")
        print(self.params)
        print("")

        # 验证输入数据形状
        if x_train.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input data shape {x_train.shape[1:]} does not match expected input shape {self.input_shape}")

        # 构建模型
        inputs = Input(shape=self.input_shape)

        x = Conv2D(5, kernel_size=self.params['kernel_size'], strides=self.params['strides'], padding='same',
                   activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(100, activation=square_activation)(x)

        if self.params['use_dropout']:
            x = Dropout(self.params['dropout'])(x)

        x = Dense(10, activation=square_activation)(x)
        outputs = Dense(10, activation=self.params['last_activation'])(x)

        self.model = keras.models.Model(inputs=inputs, outputs=outputs)

        sgd = keras.optimizers.SGD(
            learning_rate=self.params['learning_rate'],
            momentum=self.params['momentum'],
            nesterov=self.params['nesterov']
        )

        self.model.compile(optimizer=sgd,
                           loss=self.params['loss'],
                           metrics=['accuracy'])

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=1
        )

        return history, self.model

    def save(self, filepath):
        if self.model is not None:
            self.model.save(filepath)
        else:
            raise ValueError("Model has not been trained or loaded yet.")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath, custom_objects={'square_activation': square_activation})
        else:
            raise FileNotFoundError(f"Model file {filepath} does not exist.")