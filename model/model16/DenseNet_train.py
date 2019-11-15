import tensorflow as tf
import DenseNet_inference
import os

BATCH_SIZE = 50
IMAGE_SIZE = 32
REGULARAZTION_RATE   = 0.0001
LEARN_RATE = 0.001
TRAIN_STEP = 30000
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model"

def train(dataset):
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        DenseNet_inference.NUM_CHANNELS],
        name="x-input"
    )
    y_labels = tf.placeholder(
        tf.float32,
        [None,
        DenseNet_inference.NUM_LABELS],
        name='labels-input'
    )

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = DenseNet_inference.inference(x, True, regularizer)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_labels)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    saver = tf.train.Saver(var_list=tf.global_variables())

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAIN_STEP):
            xs, ys = list(dataset.get_next_batch(BATCH_SIZE, i))
            xs = xs.transpose(0,2,3,1)
            acc, step, loss_value = sess.run([accuracy, train_step, loss], feed_dict={x:xs, y_labels:ys})

            if i % 100 == 0:
                print("%d step, loss : %g, accuracy : %f"%(i, loss_value, acc))
                # print(ys)
                # print(sess.run(y, feed_dict={x:xs}))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+str(i)+".ckpt"))

        xs, ys = list(dataset.get_next_batch(BATCH_SIZE, 0))
        xs = xs.transpose(0,2,3,1)
        acc, step, loss_value = sess.run([accuracy, train_step, loss], feed_dict={x:xs, y_labels:ys})
        print("finish train, loss: %g, accracy: %f"%(loss_value, acc))

def main(argv=None):
    cifar10 = DenseNet_inference.cifar10()
    train(cifar10)

if __name__ == "__main__":
    tf.app.run()






