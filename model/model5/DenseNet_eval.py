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


        y = DenseNet_inference.inference(x, None, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, MODEL_SAVE_PATH + "model28000.ckpt")
            summary_write = tf.summary.FileWriter(GRAPH_SAVE_PATH, tf.get_default_graph())
            i = 0
            acc1 = 0
            acc2 = 0
            acc3 = 0
            while i*BATCH_SIZE < 10000 - BATCH_SIZE:
                x1, y1 = list(dataset.get_test_batch(i*BATCH_SIZE, i*BATCH_SIZE + BATCH_SIZE)) #测试集准确率
                x2, y2 = list(dataset.get_val_batch(i*BATCH_SIZE, i*BATCH_SIZE + BATCH_SIZE)) #验证集准确率
                x3, y3 = list(dataset.get_next_batch(BATCH_SIZE, i)) #训练集准确率
                x1 = x1.transpose(0,2,3,1)
                x2 = x2.transpose(0,2,3,1)
                x3 = x3.transpose(0,2,3,1)
                accuracy_score1 = sess.run(accuracy, feed_dict={x:x1, y_labels:y1})
                accuracy_score2 = sess.run(accuracy, feed_dict={x:x2, y_labels:y2})
                accuracy_score3 = sess.run(accuracy, feed_dict={x:x3, y_labels:y3})
                i = i + 1
                acc1 = acc1 + accuracy_score1
                acc2 = acc2 + accuracy_score2
                acc3 = acc3 + accuracy_score3
            acc1 = acc1 / i
            acc2 = acc2 / i
            acc3 = acc3 / i
            print("测试集准确率: %f\n验证集准确率：%f\n训练集准确率: %f\n"%(acc1, acc2, acc3))


def main(argv=None):
    cifar = DenseNetFunc.cifar10()
    evaluate(cifar)


if __name__ == '__main__':
    tf.app.run()