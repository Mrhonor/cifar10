# 尝试使用变种的DenseNet实现cifar10数据集的识别
---
>[论文引用DenseNet](https://arxiv.org/pdf/1608.06993.pdf)

---

## 模型的详情
***注：本库下包含多个不同微调的模型，模型的详情仅为部分模型情况***

**input - 5x5 conv - 2x2 max_pool - (bn - relu - 1x1 conv - bn - relu - 3x3 conv) x 3 - 4x4 avg_pool - bn - softmax**

该模型是不完全的DenseNet, 去掉了多个Dense Block, 做的比较好的分类准确率仅为78%左右

---
## 该库的文件体系
+ ./ 
   + Dense_inference.py 包含了普遍模型前向传播的定义
   + Dense_train.py 包含了训练的操作和模型的保存
   + Dense_eval.py 加载模型，计算交叉验证的准确率
   + DenseFunc.py 包含所需要的加载数据集的部分函数
   + Dense_test.py 加载模型，不断显示图片可视化测试分类成果
   + model/
      + model*/ 对应不同的特殊模型
        + Dense_inference.py 该特殊模型的前向传播定义
        + Dense_train.py 
        + Dense_eval.py 
        + DenseFunc.py 
        + readme.md 该特殊模型的一些详情，准确率的情况
        + model/ 保存训练好的模型结构和参数
        + graph/ 保存tensorboard流程图
+ ../cifar-10-batches-py/ 数据集，数据集应当放在与该库外相同目录的地方



