# 基于Python的支持ckks加密的cryptonet

## 简介
本项目是一个基于ckks的用于实现和测试深度学习模型加密推理的Python库，它是在[CNN](URL "https://github.com/poloclub/cnn-explainer")的基础上进行的扩展与改进。参考论文 [CryptoNet](URL "https://proceedings.mlr.press/v48/gilad-bachrach16.pdf")将CKKS同态加密与CNN神经网络结合起来，本项目旨在简化CKKS同态加密（HE）技术的应用，支持训练神经网络模型，并能够在不泄露数据隐私的情况下执行加密推理。

## 特性
- 继承自CNN神经网络：保留了原始神经网络的核心功能和结构。
- 增强的功能：增加了CKKS同态加密更多的调试选项、模型导出功能以及更灵活的参数配置。
- 用户友好：提供了更加清晰的API文档和使用示例，便于新用户快速上手。
- 持续改进：我们致力于不断优化性能并添加新的特性以满足不同的需求。

## 环境
- Python 3.9版本
- Tensorflow 2.18.0 版本
- tenseal 0.3.15 版本

## 用法
### 第一步：安装下载该项目

点击右上方下载该项目，将其保存到本地，运用pycharm等编程软件打开。
此时在文件目录下引用库函数，例如：

    from modules.model import Model

### 第二步：例子，使用modules库中的函数进行加密

#### 首先需要初始化参数
在开始之前，你需要设置训练参数和同态加密参数。例如：

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

    p_moduli = [1099511922689, 1099512004609]  # 模数列表
    coeff_mod = 8192  # 系数模量
    precision = 2  # 编码精度

#### 数据集准备
使用 Dataset 类来加载和预处理数据集：

    ds = Dataset(verbosity=True)
    (train, train_labels), (test, test_labels) = ds.load()

#### 模型训练
创建 Model 实例并训练模型。如果存在已保存的模型，则会自动加载。

    model_instance = Model(input_shape=(15, 15, 1), params=params)

    if not os.path.exists(model_path):
        history, trained_model = model_instance.fit(train, train_labels, test, test_labels)
        trained_model.save(model_path)
    else:
        model_instance.load(model_path)

#### 同态加密评估
使用 Cryptonet 类来进行加密推理评估：

    cn = Cryptonet(test, test_labels, model_instance.model, p_moduli, coeff_mod, precision, True)
    cn.evaluate()

#### 获取原始准确性
最后，可以通过以下方式获取未加密状态下的模型准确性：

    acc = model_instance.getAccuracy(test, test_labels)
    print("Original Accuracy: " + str(acc) + "%")

## 调试
如果你希望启用调试模式，请将全局变量 debug 设置为 True，然后可以根据需要设置 debug_plain, debug_encoded, 和 debug_encrypted 来分别测试不同阶段的模型行为。

## 原始项目对比与改进
相比于原始的CryptoNets项目，我们在以下几个方面进行了改进：

- 首先是运用了不同的编程语言Python，便于用户使用
- 引入了更直观的模块化设计，使得代码更容易维护和扩展。
- 提供了更丰富的接口和功能，如模型导出、更详细的日志输出等。
- 对API进行了优化，以便更好地适应现代机器学习工作流。
- 增加了详细的文档和支持材料，帮助开发者更快地上手使用。

## 贡献
欢迎贡献代码！请先阅读贡献指南。
