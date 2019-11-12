import tensorflow as tf
from DenseNetFunc import *
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
'''
label:
      0 airplane
      1 automobile
      2 bird
      3 cat
      4 deer
      5 dog
      6 frog
      7 horse
      8 ship
      9 truck
'''

CONV1_SIZE = 3
NUM_CHANNELS = 3
NUM_LABELS = 10
CONV1_DEEP = 32

CONV2_SIZE = 3
CONV2_DEEP = 32

CONV3_SIZE = 3
CONV3_DEEP = 64

CONV4_SIZE = 3
CONV4_DEEP = 128

CONV5_SIZE = 3
CONV5_DEEP = 168

DECAY = 0.9

def inference(input_tensor, train, regularizer):
      with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(
                  tf.nn.conv2d(input_tensor, conv1_weights, strides = [1,1,1,1], padding='SAME'), 
                  conv1_biases
            )
            if train != None:
                  conv1 = tf.nn.dropout(conv1, 0.5)
            conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

      #bn - relu - 3x3 conv
      with tf.variable_scope('layer2-conv2'):
            conv2_bn = batch_norm(conv1, decay=DECAY, updates_collections=None, is_training=train)
            conv2_relu = tf.nn.relu(conv2_bn)

            conv2_weights = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(
                  tf.nn.conv2d(conv2_relu, conv2_weights, strides=[1,1,1,1], padding='SAME'),
                  conv2_biases
            )
            if train != None:
                  conv2 = tf.nn.dropout(conv2, 0.5)


      with tf.variable_scope('layer3-conv3'):
            conv3_bn = batch_norm(conv2, decay=DECAY, updates_collections=None, is_training=train)
            conv3_relu = tf.nn.relu(conv3_bn)

            conv3_weights = tf.get_variable("weights", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("biases", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.bias_add(
                  tf.nn.conv2d(conv3_relu, conv3_weights, strides=[1,1,1,1], padding='SAME'),
                  conv3_biases
            )
            if train != None:
                  conv3 = tf.nn.dropout(conv3, 0.5)

            conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
      with tf.variable_scope('layer4-conv4'):
            conv4_bn = batch_norm(conv3, decay=DECAY, updates_collections=None, is_training=train)
            conv4_relu = tf.nn.relu(conv4_bn)

            conv4_weights = tf.get_variable("weights", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("biases", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.bias_add(
                  tf.nn.conv2d(conv4_relu, conv4_weights, strides=[1,1,1,1], padding='SAME'),
                  conv4_biases
            )
            if train != None:
                  conv4 = tf.nn.dropout(conv4, 0.5)

            conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

      with tf.variable_scope('layer5-conv5'):
            conv5_bn = batch_norm(conv4, decay=DECAY, updates_collections=None, is_training=train)
            conv5_relu = tf.nn.relu(conv5_bn)

            conv5_weights = tf.get_variable("weights", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_biases = tf.get_variable("biases", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
            conv5 = tf.nn.bias_add(
                  tf.nn.conv2d(conv5_relu, conv5_weights, strides=[1,1,1,1], padding='SAME'),
                  conv5_biases
            )
            if train != None:
                  conv5 = tf.nn.dropout(conv5, 0.5)

      with tf.variable_scope('layer6-global_avg_pool'):
            pool2 = tf.nn.avg_pool(conv5, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

      pool_shape = pool2.get_shape().as_list()
      nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
      reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

      with tf.variable_scope('layer7-softmax'):
            softmax_weights = tf.get_variable("weights", [nodes, NUM_LABELS], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            if regularizer != None:
                  tf.add_to_collection('losses', regularizer(softmax_weights))

            softmax_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.0))

            softmax = tf.nn.bias_add(tf.matmul(reshaped, softmax_weights), softmax_biases)

            # pop_mean = tf.get_variable("pop_mean", [NUM_LABELS], trainable=False, initializer=tf.constant_initializer(0.0))
            # pop_var = tf.get_variable("pop_var", [NUM_LABELS], trainable=False, initializer=tf.constant_initializer(1.0))
            # softmax = batch_norm(softmax, train, pop_mean, pop_var)

            output = tf.nn.softmax(softmax)
            
      return output