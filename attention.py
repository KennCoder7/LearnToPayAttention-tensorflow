from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import numpy as np


class Attention(Layer):
    def __init__(self, local_shape, global_shape, method):
        super(Attention, self).__init__()
        self.__method = method

        self.__lcf_channel = local_shape[2]
        self.__lcf_H = local_shape[0]
        self.__lcf_W = local_shape[1]
        self.__glf_channel = global_shape

        self.__project = Dense(self.__lcf_channel, use_bias=False)
        self.__pc = Dense(1, use_bias=False)

    def __call__(self, inputs, *args, **kwargs):
        lcf = inputs[0]
        gbf = inputs[1]
        gbf_pj = self.__project(gbf) if self.__glf_channel != self.__lcf_channel else gbf   # (bs, c)
        lcf_rs = tf.reshape(lcf, [-1, self.__lcf_H * self.__lcf_W, self.__lcf_channel])     # (bs, h*w, c)
        if self.__method == 'dp':
            c = tf.squeeze(tf.matmul(lcf_rs, tf.expand_dims(gbf_pj, -1)), -1)
            # (bs, h*w, c) matmul (bs, c, 1) -> (bs, h*w, 1) -> (bs, h*w)
        elif self.__method == 'pc':
            add = tf.add(lcf_rs, tf.expand_dims(gbf_pj, -2))   # (bs, h*w, c)
            c = tf.squeeze(self.__pc(add), -1)  # (bs, h*w)
        else:
            c = None
            lcf_rs = None
            print('No such method', self.__method)
            exit(0)
        a = tf.nn.softmax(c, 1)     # (bs, h*w)
        ga = tf.squeeze(tf.matmul(tf.transpose(lcf_rs, [0, 2, 1]), tf.expand_dims(a, -1)), -1)
        # (bs, c, h*w) matmul (bs, h*w, 1) -> (bs, c)
        return ga, tf.reshape(a, [-1, self.__lcf_H, self.__lcf_W])


