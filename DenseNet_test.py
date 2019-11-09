import tensorflow as tf
import matplotlib.pyplot as plt
import DenseNet_inference
import DenseNet_train
import DenseNetFunc
import msvcrt
import numpy as np

BATCH_SIZE = 500

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

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(DenseNet_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # xs, ys = list(dataset.get_val_batch(1000, 1000+BATCH_SIZE))
                # xs = xs.transpose(0,2,3,1)
                # accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_labels:ys})
                # print("accuracy = %g" %(accuracy_score))
                # print(sess.run(y, feed_dict={x:xs}))
                i = 0
                xs, ys = list(dataset.get_val_batch(0, BATCH_SIZE))
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