import tensorflow as tf
import DenseNet_inference
import DenseNet_train
import DenseNetFunc

BATCH_SIZE = 50
MODEL_SAVE_PATH = DenseNet_train.MODEL_SAVE_PATH
GRAPH_SAVE_PATH ="graph/"

def evaluate(dataset):
    with tf.Graph().as_default() as g:
        x  = tf.placeholder(tf.float32, 
            [BATCH_SIZE, 
            DenseNet_train.IMAGE_SIZE,
            DenseNet_train.IMAGE_SIZE,
            DenseNet_inference.NUM_CHANNELS],

            name="x-input")
        y_labels = tf.placeholder(tf.float32, [None, DenseNet_inference.NUM_LABELS], name='y-input')


        y = DenseNet_inference.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(var_list=tf.global_variables())

        with tf.Session() as sess:
            saver.restore(sess, MODEL_SAVE_PATH + "model29900.ckpt")
            summary_write = tf.summary.FileWriter(GRAPH_SAVE_PATH, tf.get_default_graph())
            i = 0
            acc = 0
            while i*BATCH_SIZE < 10000 - BATCH_SIZE:
                xs, ys = list(dataset.get_val_batch(i*BATCH_SIZE, i*BATCH_SIZE + BATCH_SIZE))
                # xs, ys = list(dataset.get_next_batch(BATCH_SIZE, i))
                xs = xs.transpose(0,2,3,1)
                accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_labels:ys})
                i = i + 1
                acc = acc + accuracy_score
            acc = acc / i
            with tf.variable_scope('layer5-softmax', reuse=True):
                pop_mean = tf.get_variable("pop_mean1", [DenseNet_inference.NUM_LABELS], trainable=False, initializer=tf.constant_initializer(0.0))
                print(sess.run(pop_mean))
            print("acc: %f"%(acc))


def main(argv=None):
    cifar = DenseNetFunc.cifar10()
    evaluate(cifar)


if __name__ == '__main__':
    tf.app.run()