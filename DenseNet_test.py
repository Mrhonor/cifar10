import tensorflow as tf
import matplotlib.pyplot as plt
import DenseNet_inference
import DenseNet_train
import DenseNetFunc
import msvcrt
import numpy as np

BATCH_SIZE = 500
BEST_MODE_SAVE_PATH = "model/model13/model29900.ckpt"

def evaluate(dataset):
    with tf.Graph().as_default() as g:
        x  = tf.placeholder(tf.float32, 
            [BATCH_SIZE, 
            DenseNet_train.IMAGE_SIZE,
            DenseNet_train.IMAGE_SIZE,
            DenseNet_inference.NUM_CHANNELS],

            name="x-input")
        y_labels = tf.placeholder(tf.float32, [None, DenseNet_inference.NUM_LABELS], name='y-input')


        y = DenseNet_inference.inference(x, None, None)


        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, BEST_MODE_SAVE_PATH)
            i = 0
            xs, ys = list(dataset.get_test_batch(0, BATCH_SIZE))
            xs = xs.transpose(0,2,3,1)
            outcome = sess.run(y, feed_dict={x:xs})
            while True:
                # print("输出结果为：label: %d"%(tf.argmax(outcome, 1).eval()))
                print(outcome[i])
                print("模型给出的label：%d"%(np.argmax(outcome[i])))
                print("真正的标签为 : %d"%(np.argmax(ys[i])))
                plt.imshow(xs[i])
                plt.axis("off")
                plt.show()
                if ord(msvcrt.getch()) in [68, 100]:
                    break
                i = i + 1

            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    cifar = DenseNetFunc.cifar10()
    evaluate(cifar)


if __name__ == '__main__':
    tf.app.run()