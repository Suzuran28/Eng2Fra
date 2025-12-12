# Eng2Fra



基于带有注意力机制的GRU模型实现英译法

> 数据来源于官方

**注：该模型输出并非纯正法文**

## 使用方式

确保有 pytorch 环境

- 安装`tqdm`模块和`matplotlib`模块
- 运行 `main.py` 通过命令行使用(可自行修改)
- 安装`pyside6`模块，运行`translator_UI.py`使用带有UI界面的翻译器

## 训练模型

如果你想要进行训练，可以在`eng2fra.py`中修改以下参数

- `MAX_LENGTH `：模型可预测的最大长度，默认为`10`，该选项会影响训练时间，但影响较小

- `train_model`中的`epochs` 和 `lr`，默认分别为`10`和`1e-4`，该选项会影响训练时间，影响较大

- 在`get_data`方法中，`lines[:16000]`为用的数据量，修改`:`后的值修改训练数据量，该选项会影响训练时间，影响非常大

- 该模型的数据处理中，去除了大部分法语格式，你可以修改`unicodeToAscii`，`normalizeEng`和`normalizeFra`方法来自定义预处理

- 方法`format_french_sentence`用来格式化输出，可以自定义修改

- 方法`test_use_seq2seq`可以用来测试模型，`shows`用来控制输出测试的数量，默认为`3`

  > 注：该方法取数据的 6000+shows 来进行测试，若数据不足6000，请自行修改259~260处的数据

- 方法`use_seq2seq`需要传入一个经过预处理的英文句子，会返回一个模型预测结果

  > 例：print(use_seq2seq(normalizeEng("Thank you.")))

- 默认会进行损失绘图，如果不需要，可以注释掉 253~259 的内容

- 每次训练都会基于旧模型(如果存在)来进行训练，如果不需要可以注释掉 223~229 的内容

- `eng2fra.py`中自带`get_resource_path`方法，可以使用`pyinstaller`对其打包，此内容会影响到后面路径的调用，请勿删除或修改

