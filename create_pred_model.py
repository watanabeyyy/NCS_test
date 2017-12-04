import tensorflow as tf
from train import weight_variable, bias_variable
from train import conv2d, max_pool_2x2, max_pool_3x3
from train import batch_norm_wrapper


def predict(input):
    is_training = False
    phase_train = tf.placeholder(tf.bool) if is_training else None

    W_conv1 = weight_variable([5, 5, 1, 16])
    h_conv1 = tf.nn.relu(batch_norm_wrapper(conv2d(input, W_conv1), phase_train))
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 16, 8])
    h_conv2 = tf.nn.relu(batch_norm_wrapper(conv2d(h_pool1, W_conv2), phase_train))
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, 8, 8])
    h_conv3 = tf.nn.relu(batch_norm_wrapper(conv2d(h_pool2, W_conv3), phase_train))
    h_pool3 = max_pool_3x3(h_conv3)

    W_fc1 = weight_variable([9 * 9 * 8, 64])
    h_pool_flat = tf.reshape(h_pool3, [1, 9 * 9 * 8])
    h_fc1 = tf.nn.relu(batch_norm_wrapper(tf.matmul(h_pool_flat, W_fc1), phase_train))

    keep_prob = 1.0
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([64, 10])
    b_fc2 = bias_variable([10])
    output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="output")

    return output


def run(name):
    with tf.Graph().as_default():
        input = tf.placeholder("float", [1, 108, 108, 1], name="input")
        output = predict(input)

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            last_model = "./model/cnn-model-99"
            saver.restore(sess, last_model)
            saver.save(sess, name)


run('./model/pred_model')
