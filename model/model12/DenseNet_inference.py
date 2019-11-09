import tensorflow as tf
from DenseNetFunc import *
import matplotlib.pyplot as plt
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

CONV2_DEEP_BOTTLENECK = CONV1_DEEP / 2
CONV2_SIZE = 3
CONV2_DEEP = 32

CONV3_DEEP_BOTTLENECK = (CONV1_DEEP + CONV2_DEEP) / 2
CONV3_SIZE = 3
CONV3_DEEP = 64

CONV4_DEEP_BOTTLENECK = (CONV1_DEEP + CONV2_DEEP + CONV3_DEEP) / 2
CONV4_SIZE = 3
CONV4_DEEP = 128


def inference(input_tensor, train, regularizer):
      # 5x5x32 conv
      with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(
                  tf.nn.conv2d(input_tensor, conv1_weights, strides = [1,1,1,1], padding='SAME'), 
                  conv1_biases
            )
            # conv1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

      # bn - relu - 1x1 conv - bn - relu - 3x3 conv
      with tf.variable_scope('layer2-conv2'):
            conv2_bn1 = batch_norm(conv1)
            conv2_relu1 = tf.nn.relu(conv2_bn1)

            conv2_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP, CONV2_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases1 = tf.get_variable("biases1", [CONV2_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv2_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv2_relu1, conv2_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv2_biases1
            )

            conv2_bn2 = batch_norm(conv2_bottleneck)
            conv2_relu2 = tf.nn.relu(conv2_bn2)

            conv2_weights2 = tf.get_variable("weights2", [CONV2_SIZE, CONV2_SIZE, CONV2_DEEP_BOTTLENECK, CONV2_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases2 = tf.get_variable("biases2", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(
                  tf.nn.conv2d(conv2_relu2, conv2_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv2_biases2
            )
            if train != None:
                  conv2 = tf.nn.dropout(conv2, 0.5)

      with tf.variable_scope('layer3-conv3'):
            conv3_concat = tf.concat([conv1, conv2], 3)
            conv3_bn1 = batch_norm(conv3_concat)
            conv3_relu1 = tf.nn.relu(conv3_bn1)

            conv3_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP+CONV2_DEEP, CONV3_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases1 = tf.get_variable("biases1", [CONV3_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv3_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv3_relu1, conv3_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv3_biases1
            )

            conv3_bn2 = batch_norm(conv3_bottleneck)
            conv3_relu2 = tf.nn.relu(conv3_bn2)

            conv3_weights2 = tf.get_variable("weights2", [CONV3_SIZE, CONV3_SIZE, CONV3_DEEP_BOTTLENECK, CONV3_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases2 = tf.get_variable("biases2", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.bias_add(
                  tf.nn.conv2d(conv3_relu2, conv3_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv3_biases2
            )
            if train != None:
                  conv3 = tf.nn.dropout(conv3, 0.5)
    
      with tf.variable_scope('layer4-conv4'):
            conv4_concat = tf.concat([conv1, conv2, conv3], 3)
            conv4_bn1 = batch_norm(conv4_concat)
            conv4_relu1 = tf.nn.relu(conv4_bn1)

            conv4_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP+CONV2_DEEP+CONV3_DEEP, CONV4_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases1 = tf.get_variable("biases1", [CONV4_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv4_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv4_relu1, conv4_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv4_biases1
            )

            conv4_bn2 = batch_norm(conv4_bottleneck)
            conv4_relu2 = tf.nn.relu(conv4_bn2)

            conv4_weights2 = tf.get_variable("weights2", [CONV4_SIZE, CONV4_SIZE, CONV4_DEEP_BOTTLENECK, CONV4_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases2 = tf.get_variable("biases2", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.bias_add(
                  tf.nn.conv2d(conv4_relu2, conv4_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv4_biases2
            )
            if train != None:
                  conv4 = tf.nn.dropout(conv4, 0.5)

      with tf.variable_scope('layer4-avg_pool'):
            concat = tf.concat([conv1, conv2, conv3, conv4], 3)
            
            pool2 = tf.nn.max_pool(concat, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

      pool_shape = pool2.get_shape().as_list()
      nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
      reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

      with tf.variable_scope('layer5-softmax'):
            softmax_weights = tf.get_variable("weights", [nodes, NUM_LABELS], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            if regularizer != None:
                  tf.add_to_collection('losses', regularizer(softmax_weights))

            softmax_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.0))

            softmax = tf.nn.bias_add(tf.matmul(reshaped, softmax_weights), softmax_biases)

            softmax = batch_norm(softmax)

            output = tf.nn.softmax(softmax)
            
      return output