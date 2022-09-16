# 项目介绍

本项目是个人学习Transformer时的一些学习笔记和博客的汇总。目前内容有：

1. `AnnotatedTransformer.ipynb`：万字逐行解析与实现Transformer，并进行德译英实战
2. `nn.Transformer_demo.ipynb`: Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解
3. `attention_tutorial.md`：层层剖析，让你彻底搞懂Self-Attention、MultiHead-Attention和Masked-Attention的机制和原理
4. `en_to_zh_demo.ipynb`：Pytorch实战：基于nn.Transformer实现机器翻译（英译汉）
5. `Hugging Face快速入门`：讲解了Hugging Face模型和数据集两个重要部分的使用方法
6. `bert_classification_demo.ipynb`：Pytorch实战：基于BERT实现文本隐喻分类（Kaggle入门题目）
7. `bert_pytorch_implement.ipynb`：BERT源码实现与解读(Pytorch)

如有错误的地方，欢迎指出。

我的博客地址为：https://blog.csdn.net/zhaohongfei_358



# Transformer源码解读

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/chaotic-transformer-tutorials/blob/master/AnnotatedTransformer.ipynb)

`AnnotatedTransformer.ipynb` 文件是对Transformer源码的一些解释和注释扩展。源码来源于项目[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) 。 并在该项目的基础上删除了一些增加了许多详细的注释，并且删除了一些与理解Transformer无关的代码（例如并行计算）。

本篇博客为：https://blog.csdn.net/zhaohongfei_358/article/details/126085246

# Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/chaotic-transformer-tutorials/blob/master/nn.Transformer_demo.ipynb)

`nn.Transformer_demo.ipynb` 文件详细讲解了Pytorch中nn.Transformer的使用，并从黑盒角度讲解了Transformer的使用

本篇博客为：https://blog.csdn.net/zhaohongfei_358/article/details/126019181

# 层层剖析，让你彻底搞懂Self-Attention、MultiHead-Attention和Masked-Attention的机制和原理

博客地址为：https://blog.csdn.net/zhaohongfei_358/article/details/122861751

# Pytorch实战：基于nn.Transformer实现机器翻译（英译汉）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/chaotic-transformer-tutorials/blob/master/en_to_zh_demo.ipynb)

博客地址：https://blog.csdn.net/zhaohongfei_358/article/details/126175328

# Hugging Face快速入门

博客地址:https://blog.csdn.net/zhaohongfei_358/article/details/126224199

# Pytorch实战：基于BERT实现文本隐喻分类（Kaggle入门题目）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/chaotic-transformer-tutorials/blob/master/bert_classification_demo.ipynb)

使用Bert做一个简单的二分类问题。

博客地址：https://blog.csdn.net/zhaohongfei_358/article/details/126426855

# BERT源码实现与解读(Pytorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/chaotic-transformer-tutorials/blob/master/bert_pytorch_implement.ipynb)

使用nn.Transformer构建BERT模型，并使用样例样本使用MLM任务和NSP任务训练BERT

博客地址：https://blog.csdn.net/zhaohongfei_358/article/details/126426855
