import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def batch_norm_wrapper(inputs, phase_train=None, decay=0.99):
    epsilon = 1e-5
    out_dim = inputs.get_shape()[-1]
    scale = tf.Variable(tf.ones([out_dim]))
    beta = tf.Variable(tf.zeros([out_dim]))
    pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
    pop_var = tf.Variable(tf.ones([out_dim]), trainable=False)
    if phase_train == None:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
    rank = len(inputs.get_shape())
    axes = list(range(rank - 1))
    batch_mean, batch_var = tf.nn.moments(inputs, axes)
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale,
                                             epsilon)

    def average():
        train_mean = pop_mean.assign(ema.average(batch_mean))
        train_var = pop_var.assign(ema.average(batch_var))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon)

    return tf.cond(phase_train, update, average)

def train(fine_tune):
    is_training = True
    with tf.Graph().as_default():
        phase_train = tf.placeholder(tf.bool) if is_training else None
        x = tf.placeholder("float", [None, 108, 108, 1], name="input")
        y = tf.placeholder("float", [None, 10])

        W_conv1 = weight_variable([5, 5, 1, 16])
        h_conv1 = tf.nn.relu(batch_norm_wrapper(conv2d(x, W_conv1), phase_train))
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 16, 8])
        h_conv2 = tf.nn.relu(batch_norm_wrapper(conv2d(h_pool1, W_conv2), phase_train))
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([5, 5, 8, 8])
        h_conv3 = tf.nn.relu(batch_norm_wrapper(conv2d(h_pool2, W_conv3), phase_train))
        h_pool3 = max_pool_3x3(h_conv3)

        W_fc1 = weight_variable([9 * 9 * 8, 64])
        h_pool_flat = tf.reshape(h_pool3, [-1, 9 * 9 * 8])
        h_fc1 = tf.nn.relu(batch_norm_wrapper(tf.matmul(h_pool_flat, W_fc1), phase_train))

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([64, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="output")

        cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            if is_training:
                if fine_tune:
                    last_model = "./model/cnn-model-99"
                    print("load " + last_model)
                    saver.restore(sess, last_model)
                all_x = np.load("./npy/x.npy")
                all_y = np.load("./npy/y.npy")
                all_imgs = np.shape(all_x)[0]
                train_indices = np.random.choice(all_imgs, round(all_imgs * 0.9), replace=False)
                test_indices = np.array(list(set(range(all_imgs)) - set(train_indices)))
                x_train = all_x[train_indices]
                x_test = all_x[test_indices]
                y_train = all_y[train_indices]
                y_test = all_y[test_indices]

                num_epoch = 100
                num_data = x_train.shape[0]
                batch_size = 256
                for epoch in range(num_epoch):
                    for idx in range(0, num_data, batch_size):
                        rand_index = np.random.choice(num_data, size=batch_size)
                        batch_x = x_train[rand_index]
                        batch_y = y_train[rand_index]
                        _, loss = sess.run([train_step, cross_entropy],
                                           feed_dict={phase_train: True, x: batch_x, y: batch_y, keep_prob: 0.5})
                        print("epoch %d, images %d, loss %g" % (epoch, idx, loss / batch_size))
                    test_accuracy, test_loss = sess.run([accuracy, cross_entropy],
                                                        feed_dict={phase_train: False, x: x_test, y: y_test,
                                                                   keep_prob: 1.0})
                    print("epoch %d, test accuracy %g" % (epoch, test_accuracy))
                    saver.save(sess, "./model/" + 'cnn-model', global_step=epoch)
                    # CSV out
                    import csv

                    f = open('output.csv', 'a')
                    writer = csv.writer(f, lineterminator='\n')
                    csvlist = []
                    csvlist.append(epoch)
                    csvlist.append(test_accuracy)
                    csvlist.append(test_loss / x_test.shape[0])
                    writer.writerow(csvlist)
                    f.close()
            else:
                last_model = "./model/cnn-model-199"
                print("load " + last_model)
                saver.restore(sess, last_model)


                def pred(im):
                    im = im.astype('float32') / 255.
                    prediction = tf.argmax(y_conv, 1)
                    num, prob = sess.run([prediction, y_conv], feed_dict={x: im, keep_prob: 1.0})
                    error = False
                    for i in range(4):
                        pred_prob = prob[i, num[i]]
                        if (pred_prob < 0.30):
                            print(pred_prob)
                            error = True
                    return error, num

if __name__=="__main__":
    train(False)