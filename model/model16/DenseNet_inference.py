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

CONV1_SIZE = 5
NUM_CHANNELS = 3
NUM_LABELS = 10
CONV1_DEEP = 32
K = 32  # growt rate

CONV2_DEEP_BOTTLENECK = 4 * K
CONV2_SIZE = 3
CONV2_DEEP = K

CONV3_DEEP_BOTTLENECK = 4 * K
CONV3_SIZE = 3
CONV3_DEEP = K

CONV4_DEEP_BOTTLENECK = 4 * K
CONV4_SIZE = 3
CONV4_DEEP = K

CONV5_DEEP_BOTTLENECK = 4 * K
CONV5_SIZE = 3
CONV5_DEEP = K

CONV6_DEEP_BOTTLENECK = 4 * K
CONV6_SIZE = 3
CONV6_DEEP = K

CONV7_DEEP_BOTTLENECK = 4 * K
CONV7_SIZE = 3
CONV7_DEEP = K

CONV8_DEEP_BOTTLENECK = 4 * K
CONV8_SIZE = 3
CONV8_DEEP = K

CONV9_DEEP_BOTTLENECK = (CONV1_DEEP + CONV2_DEEP + CONV3_DEEP + CONV4_DEEP + CONV5_DEEP + CONV6_DEEP + CONV7_DEEP + CONV8_DEEP) / 2

CONV_DEEP_BOTTLENECK = 4 * K
CONV_SIZE = 3
CONV_DEEP = K

def inference(input_tensor, train, regularizer):
      # 3x3x32 conv
      with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(
                  tf.nn.conv2d(input_tensor, conv1_weights, strides = [1,1,1,1], padding='SAME'), 
                  conv1_biases
            )
            conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

      # bn - relu - 1x1 conv - bn - relu - 3x3 conv
      with tf.variable_scope('layer2-conv2'):
            conv2_bn1 = batch_norm(conv1, train, CONV_DEEP, 1)
            conv2_relu1 = tf.nn.relu(conv2_bn1)

            conv2_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP, CONV2_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases1 = tf.get_variable("biases1", [CONV2_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv2_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv2_relu1, conv2_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv2_biases1
            )

            conv2_bn2 = batch_norm(conv2_bottleneck, train, CONV_DEEP*4, 2)
            conv2_relu2 = tf.nn.relu(conv2_bn2)

            conv2_weights2 = tf.get_variable("weights2", [CONV2_SIZE, CONV2_SIZE, CONV2_DEEP_BOTTLENECK, CONV2_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases2 = tf.get_variable("biases2", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(
                  tf.nn.conv2d(conv2_relu2, conv2_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv2_biases2
            )
            if train == True:
                  conv2 = tf.nn.dropout(conv2, 0.5)

      with tf.variable_scope('layer3-conv3'):
            conv3_concat = tf.concat([conv1, conv2], 3)
            conv3_bn1 = batch_norm(conv3_concat, train, CONV_DEEP*2, 1)
            conv3_relu1 = tf.nn.relu(conv3_bn1)

            conv3_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP+CONV2_DEEP, CONV3_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases1 = tf.get_variable("biases1", [CONV3_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv3_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv3_relu1, conv3_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv3_biases1
            )

            conv3_bn2 = batch_norm(conv3_bottleneck, train, CONV_DEEP*4, 2)
            conv3_relu2 = tf.nn.relu(conv3_bn2)

            conv3_weights2 = tf.get_variable("weights2", [CONV3_SIZE, CONV3_SIZE, CONV3_DEEP_BOTTLENECK, CONV3_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases2 = tf.get_variable("biases2", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.bias_add(
                  tf.nn.conv2d(conv3_relu2, conv3_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv3_biases2
            )
            if train == True:
                  conv3 = tf.nn.dropout(conv3, 0.5)
    
      with tf.variable_scope('layer4-conv4'):
            conv4_concat = tf.concat([conv1, conv2, conv3], 3)
            conv4_bn1 = batch_norm(conv4_concat, train, CONV_DEEP*3, 1)
            conv4_relu1 = tf.nn.relu(conv4_bn1)

            conv4_weights1 = tf.get_variable("weights1", [1, 1, CONV1_DEEP+CONV2_DEEP+CONV3_DEEP, CONV4_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases1 = tf.get_variable("biases1", [CONV4_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv4_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv4_relu1, conv4_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv4_biases1
            )

            conv4_bn2 = batch_norm(conv4_bottleneck, train, CONV_DEEP*4, 2)
            conv4_relu2 = tf.nn.relu(conv4_bn2)

            conv4_weights2 = tf.get_variable("weights2", [CONV4_SIZE, CONV4_SIZE, CONV4_DEEP_BOTTLENECK, CONV4_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases2 = tf.get_variable("biases2", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.bias_add(
                  tf.nn.conv2d(conv4_relu2, conv4_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv4_biases2
            )
            if train == True:
                  conv4 = tf.nn.dropout(conv4, 0.5)

      with tf.variable_scope('layer5-conv5'):
            conv5_concat = tf.concat([conv1, conv2, conv3, conv4], 3)
            conv5_bn1 = batch_norm(conv5_concat, train, CONV_DEEP*4, 1)
            conv5_relu1 = tf.nn.relu(conv5_bn1)

            conv5_weights1 = tf.get_variable("weights1", [1, 1, K*4, CONV5_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_biases1 = tf.get_variable("biases1", [CONV5_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv5_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv5_relu1, conv5_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv5_biases1
            )

            conv5_bn2 = batch_norm(conv5_bottleneck, train, CONV_DEEP*4, 2)
            conv5_relu2 = tf.nn.relu(conv5_bn2)

            conv5_weights2 = tf.get_variable("weights2", [CONV5_SIZE, CONV5_SIZE, CONV5_DEEP_BOTTLENECK, CONV5_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_biases2 = tf.get_variable("biases2", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
            conv5 = tf.nn.bias_add(
                  tf.nn.conv2d(conv5_relu2, conv5_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv5_biases2
            )
            if train == True:
                  conv5 = tf.nn.dropout(conv5, 0.5)

      with tf.variable_scope('layer6-conv6'):
            conv6_concat = tf.concat([conv1, conv2, conv3, conv4, conv5], 3)
            conv6_bn1 = batch_norm(conv6_concat, train, CONV_DEEP*5, 1)
            conv6_relu1 = tf.nn.relu(conv6_bn1)

            conv6_weights1 = tf.get_variable("weights1", [1, 1, K*5, CONV6_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv6_biases1 = tf.get_variable("biases1", [CONV6_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv6_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv6_relu1, conv6_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv6_biases1
            )

            conv6_bn2 = batch_norm(conv6_bottleneck, train, CONV_DEEP*4, 2)
            conv6_relu2 = tf.nn.relu(conv6_bn2)

            conv6_weights2 = tf.get_variable("weights2", [CONV6_SIZE, CONV6_SIZE, CONV6_DEEP_BOTTLENECK, CONV6_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv6_biases2 = tf.get_variable("biases2", [CONV6_DEEP], initializer=tf.constant_initializer(0.0))
            conv6 = tf.nn.bias_add(
                  tf.nn.conv2d(conv6_relu2, conv6_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv6_biases2
            )
            if train == True:
                  conv6 = tf.nn.dropout(conv6, 0.5)

      with tf.variable_scope('layer7-conv7'):
            conv7_concat = tf.concat([conv1, conv2, conv3, conv4, conv5, conv6], 3)
            conv7_bn1 = batch_norm(conv7_concat, train, CONV_DEEP*6, 1)
            conv7_relu1 = tf.nn.relu(conv7_bn1)

            conv7_weights1 = tf.get_variable("weights1", [1, 1, K*6, CONV7_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv7_biases1 = tf.get_variable("biases1", [CONV7_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv7_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv7_relu1, conv7_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv7_biases1
            )

            conv7_bn2 = batch_norm(conv7_bottleneck, train, CONV_DEEP*4, 2)
            conv7_relu2 = tf.nn.relu(conv7_bn2)

            conv7_weights2 = tf.get_variable("weights2", [CONV7_SIZE, CONV7_SIZE, CONV7_DEEP_BOTTLENECK, CONV7_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv7_biases2 = tf.get_variable("biases2", [CONV7_DEEP], initializer=tf.constant_initializer(0.0))
            conv7 = tf.nn.bias_add(
                  tf.nn.conv2d(conv7_relu2, conv7_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv7_biases2
            )
            if train == True:
                  conv7 = tf.nn.dropout(conv7, 0.5)

      with tf.variable_scope('layer8-conv8'):
            conv8_concat = tf.concat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 3)
            conv8_bn1 = batch_norm(conv8_concat, train, CONV_DEEP*7, 1)
            conv8_relu1 = tf.nn.relu(conv8_bn1)

            conv8_weights1 = tf.get_variable("weights1", [1, 1, K*7, CONV8_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv8_biases1 = tf.get_variable("biases1", [CONV8_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv8_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv8_relu1, conv8_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv8_biases1
            )

            conv8_bn2 = batch_norm(conv8_bottleneck, train, CONV_DEEP*4, 2)
            conv8_relu2 = tf.nn.relu(conv8_bn2)

            conv8_weights2 = tf.get_variable("weights2", [CONV8_SIZE, CONV8_SIZE, CONV8_DEEP_BOTTLENECK, CONV8_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv8_biases2 = tf.get_variable("biases2", [CONV8_DEEP], initializer=tf.constant_initializer(0.0))
            conv8 = tf.nn.bias_add(
                  tf.nn.conv2d(conv8_relu2, conv8_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv8_biases2
            )
            if train == True:
                  conv8 = tf.nn.dropout(conv8, 0.5)

      with tf.variable_scope('layer9-transition'):
            conv9_concat = tf.concat([conv8_concat, conv8], 3)
            conv9_bn1 = batch_norm(conv9_concat, train, K*8, 1)
            conv9_relu1 = tf.nn.relu(conv9_bn1)

            conv9_weights1 = tf.get_variable("weights1", [1, 1, CONV9_DEEP_BOTTLENECK*2, CONV9_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv9_biases1 = tf.get_variable("biases1", [CONV9_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv9_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv9_relu1, conv9_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv9_biases1
            )            
            pool2 = tf.nn.avg_pool(conv9_concat, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

      with tf.variable_scope('layer10-conv10'):
            conv10_bn1 = batch_norm(pool2, train, CONV9_DEEP_BOTTLENECK*2, 1)
            conv10_relu1 = tf.nn.relu(conv10_bn1)

            conv10_weights1 = tf.get_variable("weights1", [1, 1, CONV9_DEEP_BOTTLENECK*2, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv10_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv10_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv10_relu1, conv10_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv10_biases1
            )

            conv10_bn2 = batch_norm(conv10_bottleneck, train, CONV_DEEP*4, 2)
            conv10_relu2 = tf.nn.relu(conv10_bn2)

            conv10_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv10_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv10 = tf.nn.bias_add(
                  tf.nn.conv2d(conv10_relu2, conv10_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv10_biases2
            )
            if train == True:
                  conv10 = tf.nn.dropout(conv10, 0.5)

      with tf.variable_scope('layer11-conv11'):
            conv11_bn1 = batch_norm(conv10, train, CONV_DEEP, 1)
            conv11_relu1 = tf.nn.relu(conv11_bn1)

            conv11_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv11_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv11_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv11_relu1, conv11_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv11_biases1
            )

            conv11_bn2 = batch_norm(conv11_bottleneck, train, CONV_DEEP*4, 2)
            conv11_relu2 = tf.nn.relu(conv11_bn2)

            conv11_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv11_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv11 = tf.nn.bias_add(
                  tf.nn.conv2d(conv11_relu2, conv11_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv11_biases2
            )
            if train == True:
                  conv11 = tf.nn.dropout(conv11, 0.5)

      with tf.variable_scope('layer12-conv12'):
            conv12_concat = tf.concat([conv10, conv11], 3)
            conv12_bn1 = batch_norm(conv12_concat, train, CONV_DEEP*2, 1)
            conv12_relu1 = tf.nn.relu(conv12_bn1)

            conv12_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP*2, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv12_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv12_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv12_relu1, conv12_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv12_biases1
            )

            conv12_bn2 = batch_norm(conv12_bottleneck, train, CONV_DEEP*4, 2)
            conv12_relu2 = tf.nn.relu(conv12_bn2)

            conv12_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv12_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv12 = tf.nn.bias_add(
                  tf.nn.conv2d(conv12_relu2, conv12_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv12_biases2
            )
            if train == True:
                  conv12 = tf.nn.dropout(conv12, 0.5)

      with tf.variable_scope('layer13-conv13'):
            conv13_concat = tf.concat([conv10, conv11, conv12], 3)
            conv13_bn1 = batch_norm(conv13_concat, train, CONV_DEEP*3, 1)
            conv13_relu1 = tf.nn.relu(conv13_bn1)

            conv13_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP*3, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv13_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv13_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv13_relu1, conv13_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv13_biases1
            )

            conv13_bn2 = batch_norm(conv13_bottleneck, train, CONV_DEEP*4, 2)
            conv13_relu2 = tf.nn.relu(conv13_bn2)

            conv13_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv13_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv13 = tf.nn.bias_add(
                  tf.nn.conv2d(conv13_relu2, conv13_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv13_biases2
            )
            if train == True:
                  conv13 = tf.nn.dropout(conv13, 0.5)

      with tf.variable_scope('layer14-conv14'):
            conv14_concat = tf.concat([conv10, conv11, conv12, conv13], 3)
            conv14_bn1 = batch_norm(conv14_concat, train, CONV_DEEP*4, 1)
            conv14_relu1 = tf.nn.relu(conv14_bn1)

            conv14_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP*4, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv14_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv14_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv14_relu1, conv14_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv14_biases1
            )

            conv14_bn2 = batch_norm(conv14_bottleneck, train, CONV_DEEP*4, 2)
            conv14_relu2 = tf.nn.relu(conv14_bn2)

            conv14_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv14_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv14 = tf.nn.bias_add(
                  tf.nn.conv2d(conv14_relu2, conv14_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv14_biases2
            )
            if train == True:
                  conv14 = tf.nn.dropout(conv14, 0.5)

      with tf.variable_scope('layer15-conv15'):
            conv15_concat = tf.concat([conv10, conv11, conv12, conv13, conv14], 3)
            conv15_bn1 = batch_norm(conv15_concat, train, CONV_DEEP*5, 1)
            conv15_relu1 = tf.nn.relu(conv15_bn1)

            conv15_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP*5, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv15_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv15_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv15_relu1, conv15_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv15_biases1
            )

            conv15_bn2 = batch_norm(conv15_bottleneck, train, CONV_DEEP_BOTTLENECK, 2)
            conv15_relu2 = tf.nn.relu(conv15_bn2)

            conv15_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv15_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv15 = tf.nn.bias_add(
                  tf.nn.conv2d(conv15_relu2, conv15_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv15_biases2
            )
            if train == True:
                  conv15 = tf.nn.dropout(conv15, 0.5)

      with tf.variable_scope('layer16-conv16'):
            conv16_concat = tf.concat([conv10, conv11, conv12, conv13, conv14, conv15], 3)
            conv16_bn1 = batch_norm(conv16_concat, train, CONV_DEEP*6, 1)
            conv16_relu1 = tf.nn.relu(conv16_bn1)

            conv16_weights1 = tf.get_variable("weights1", [1, 1, CONV_DEEP*6, CONV_DEEP_BOTTLENECK], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv16_biases1 = tf.get_variable("biases1", [CONV_DEEP_BOTTLENECK], initializer=tf.constant_initializer(0.0))
            conv16_bottleneck = tf.nn.bias_add(
                  tf.nn.conv2d(conv16_relu1, conv16_weights1, strides=[1,1,1,1], padding='SAME'),
                  conv16_biases1
            )

            conv16_bn2 = batch_norm(conv16_bottleneck, train, CONV_DEEP_BOTTLENECK, 2)
            conv16_relu2 = tf.nn.relu(conv16_bn2)

            conv16_weights2 = tf.get_variable("weights2", [CONV_SIZE, CONV_SIZE, CONV_DEEP_BOTTLENECK, CONV_DEEP], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv16_biases2 = tf.get_variable("biases2", [CONV_DEEP], initializer=tf.constant_initializer(0.0))
            conv16 = tf.nn.bias_add(
                  tf.nn.conv2d(conv16_relu2, conv16_weights2, strides=[1,1,1,1], padding='SAME'),
                  conv16_biases2
            )
            if train == True:
                  conv16 = tf.nn.dropout(conv16, 0.5)

      with tf.variable_scope('layer17-global_avg_pool'):
            global_concat = tf.concat([conv16_concat, conv16], 3)
            
            pool3 = tf.nn.avg_pool(global_concat, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

      pool_shape = pool3.get_shape().as_list()
      nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
      reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

      with tf.variable_scope('layer5-softmax'):
            softmax_weights = tf.get_variable("weights", [nodes, NUM_LABELS], 
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            if regularizer != None:
                  tf.add_to_collection('losses', regularizer(softmax_weights))

            softmax_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.0))

            softmax = tf.nn.bias_add(tf.matmul(reshaped, softmax_weights), softmax_biases)

            softmax = batch_norm(softmax, train, NUM_LABELS, 1)

            output = tf.nn.softmax(softmax)
            
      return output