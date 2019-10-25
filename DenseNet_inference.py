import tensorflow as tf
from DenseNetFunc import *

batch1 = unpickle("../cifar-10-batches-py/data_batch_1")
data_batch1 = batch1[b'data'] 
data_batch1 = data_batch1.reshape((32,32,3,10000))
labels_batch1 = batch1.get(b'labels')
print(data_batch1.shape)
# print(labels_batch1)

