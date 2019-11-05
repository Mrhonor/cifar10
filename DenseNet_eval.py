import tensorflow as tf
import DenseNet_inference
import DenseNet_train
import DenseNetFunc

BATCH_SIZE = 10

def evaluate(dataset):
    with tf.Graph().as_default() as g:
        x  = tf.placeholder(tf.float32, 
            [BATCH_SIZE, 
            DenseNet_inference.IMAGE_SIZE,
            DenseNet_inference.IMAGE_SIZE,
            DenseNet_inference.NUM_CHANNELS],

            name="x-input")
        y_labels = tf.placeholder(tf.float32, [None, DenseNet_inference.OUTPUT_NODE], name='y-input')

        #validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        y = DenseNet_inference.inference(x, None, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(DenseNet_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("%s step, accuracy = %g" %(global_step, accuracy_score))

            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()