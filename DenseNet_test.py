import tensorflow as tf
import matplotlib.pyplot as plt
import DenseNet_inference
import DenseNet_train
import DenseNetFunc


def evaluate(dataset):
    with tf.Graph().as_default() as g:
        x  = tf.placeholder(tf.float32, 
            [1, 
            DenseNet_train.IMAGE_SIZE,
            DenseNet_train.IMAGE_SIZE,
            DenseNet_inference.NUM_CHANNELS],

            name="x-input")
        y_labels = tf.placeholder(tf.float32, [None, DenseNet_inference.NUM_LABELS], name='y-input')


        y = DenseNet_inference.inference(x, None, None)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(DenseNet_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                i = 0
                while True:
                    xs, ys = list(dataset.get_val_batch(i, i))
                    xs = xs.transpose(0,2,3,1)
                    outcome = sess.run(y, feed_dict={x:xs})
                    print("输出结果为：label: %d"%(tf.argmax(outcome, 1))
                    print(outcome)
                    plt.imshow(xs)
                    plt.axis("off")
                    plt.show()


            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    cifar = DenseNetFunc.cifar10()
    evaluate(cifar)


if __name__ == '__main__':
    tf.app.run()