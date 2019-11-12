### 10000次迭代后准确率不再提升, 交叉验证准确率为67.6%，训练集准确率74.4%
#### 最后一层不使用softmax
一开始没有使用softmax的原因为在计算softmax的时候，没有进行批量归一化（line140）：
一开始的代码
```
    softmax_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.0))
    softmax = tf.nn.bias_add(tf.matmul(reshaped, softmax_weights), softmax_biases)
    # softmax = batch_norm(softmax) 这里一开始没有加
    output = tf.nn.softmax(softmax)
```
由于参数初始化的原因，一开始计算出不同类别的数相差很大，softmax得到的结果为要么无线接近于1，要么无限接近于0，无法训练。
且一开始没有注意到这个问题，就将softmax层去掉进行训练了