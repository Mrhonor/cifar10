import tensorflow as tf
import DenseNet_inference
import DenseNet_train
import DenseNetFunc

BATCH_SIZE = 9999

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
                xs, ys = list(dataset.get_val_batch(0, BATCH_SIZE))
                xs = xs.transpose(0,2,3,1)
                accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_labels:ys})
                print("accuracy = %g" %(accuracy_score))

            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    cifar = DenseNetFunc.cifar10()
    evaluate(cifar)


if __name__ == '__main__':
    tf.app.run()