import tensorflow as tf
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# print(data_batch1.shape)
# print(labels_batch1[0])
# plt.imshow(data_batch1[0].transpose(1,2,0))
# plt.axis("off")
# plt.show()
DECAY = 0.99

def batch_norm(input, train, NUM, type):
    if type==1:
        pop_mean = tf.get_variable("pop_mean1", [NUM], trainable=False, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable("pop_var1", [NUM], trainable=False, initializer=tf.constant_initializer(1.0))
    else:
        pop_mean = tf.get_variable("pop_mean2", [NUM], trainable=False, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable("pop_var2", [NUM], trainable=False, initializer=tf.constant_initializer(1.0))

    if train==False:
        axis = list(range(len(input.get_shape())-1))
        mean, var = tf.nn.moments(input, axis)

        train_mean = tf.assign(pop_mean, pop_mean*DECAY + mean*(1 - DECAY))
        train_var = tf.assign(pop_var, pop_var*DECAY + var*(1 - DECAY))
        with tf.control_dependencies([train_mean, train_var]):
            bn = tf.nn.batch_normalization(input, mean, var, 0, 1, 1e-3)
    else:
        bn = tf.nn.batch_normalization(input, pop_mean, pop_var, 0, 1, 1e-3)
    return bn

    
def onehot(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



class cifar10:
    def __init__(self):
        batch1 = unpickle("../../../cifar-10-batches-py/data_batch_1")
        self.data_batch1 = batch1[b'data'] 
        self.data_batch1 = self.data_batch1.reshape((10000,3,32,32))
        self.labels_batch1 = batch1.get(b'labels')

        batch2 = unpickle("../../../cifar-10-batches-py/data_batch_2")
        self.data_batch2 = batch2[b'data'] 
        self.data_batch2 = self.data_batch2.reshape((10000,3,32,32))
        self.labels_batch2 = batch2.get(b'labels')

        batch3 = unpickle("../../../cifar-10-batches-py/data_batch_3")
        self.data_batch3 = batch3[b'data'] 
        self.data_batch3 = self.data_batch3.reshape((10000,3,32,32))
        self.labels_batch3 = batch3.get(b'labels')

        batch4 = unpickle("../../../cifar-10-batches-py/data_batch_4")
        self.data_batch4 = batch4[b'data'] 
        self.data_batch4 = self.data_batch4.reshape((10000,3,32,32))
        self.labels_batch4 = batch4.get(b'labels')

        batch5 = unpickle("../../../cifar-10-batches-py/data_batch_5")
        self.data_batch5 = batch5[b'data'] 
        self.data_batch5 = self.data_batch5.reshape((10000,3,32,32))
        self.labels_batch5 = batch5.get(b'labels')

        test_batch = unpickle("../../../cifar-10-batches-py/test_batch")
        self.test_data = test_batch[b'data']
        self.test_data = self.test_data.reshape((10000,3,32,32)) 
        self.labels_test = test_batch.get(b'labels')

    def get_next_batch(self, BATCH_SIZE, i):
        num = i*BATCH_SIZE % 40000
        if num <= 10000-BATCH_SIZE:
            x = self.data_batch1[num:num+BATCH_SIZE]
            y = onehot(self.labels_batch1[num:num+BATCH_SIZE], 10)
            return x, y
        if num > 10000-BATCH_SIZE and num <= 20000-BATCH_SIZE:
            num = num % 10000
            x = self.data_batch2[num:num+BATCH_SIZE]
            y = onehot(self.labels_batch2[num:num+BATCH_SIZE], 10)
            return x, y
        if num > 20000-BATCH_SIZE and num <= 30000-BATCH_SIZE:
            num = num % 10000
            x = self.data_batch3[num:num+BATCH_SIZE]
            y = onehot(self.labels_batch3[num:num+BATCH_SIZE], 10)
            return x, y
        if num > 30000-BATCH_SIZE and num <= 40000-BATCH_SIZE:
            num = num % 10000
            x = self.data_batch4[num:num+BATCH_SIZE]
            y = onehot(self.labels_batch4[num:num+BATCH_SIZE], 10)
            return x, y    
    
    def get_val_batch(self, num1, num2):
        if num1 >= 10000 or num2 >= 10000 or num1 > num2:
            return None
        
        x = self.data_batch5[num1:num2]
        y = onehot(self.labels_batch5[num1:num2], 10)
        return x, y

    def get_test_batch(self, num1, num2):
        if num1 >= 10000 or num2 >= 10000 or num1 > num2:
            return None
        
        x = self.test_data[num1:num2]
        y = onehot(self.labels_test[num1:num2], 10)
        return x, y