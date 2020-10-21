import os
from tensorflow.keras import Input, Model, optimizers, datasets
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Layer
from tensorflow.keras import utils
from attention import Attention
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, padding='same', pooling=False):
        super(ConvBlock, self).__init__()
        self.__filters = filters
        self.__ks = kernel_size
        self.__padding = padding
        self.__pooling = pooling

    def __call__(self, inputs, *args, **kwargs):
        _x = Conv2D(self.__filters, self.__ks, padding=self.__padding)(inputs)
        _x = BatchNormalization()(_x)
        _x = Activation('relu')(_x)
        if self.__pooling:
            _x = MaxPooling2D()(_x)
        return _x


def main(output_path=r'D:\pyproject\data\CIFAR10-tensorflow'):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data(r'D:\pyproject\data\cifar-10-batches-py')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)
    epochs = 30
    batch_size = 128
    n_classes = 10
    model_path_loss = output_path + r"\vgg-att.h5"
    save_model_loss = ModelCheckpoint(model_path_loss, monitor='val_loss', save_best_only=True, verbose=2)

    _inputs = Input(shape=(32, 32, 3), name='input')
    _layer = ConvBlock(64, 3)(_inputs)
    _layer = ConvBlock(128, 3)(_layer)
    c1 = ConvBlock(256, 3, pooling=True)(_layer)
    c2 = ConvBlock(512, 3, pooling=True)(c1)
    c3 = ConvBlock(512, 3, pooling=True)(c2)
    _layer = ConvBlock(512, 3, pooling=True)(c3)
    _layer = ConvBlock(512, 3, pooling=True)(_layer)
    _layer = Flatten()(_layer)
    _g = Dense(512, activation='relu')(_layer)
    _outputs = Dense(n_classes, activation='softmax')(_g)
    vgg = Model(_inputs, _outputs)
    vgg.load_weights(output_path + r"\vgg.h5")

    att1_f, att1_map = Attention((16, 16, 256), 512, method='pc')([c1, _g])
    att2_f, att2_map = Attention((8, 8, 512), 512, method='pc')([c2, _g])
    att3_f, att3_map = Attention((4, 4, 512), 512, method='pc')([c3, _g])
    f_concat = tf.concat([att1_f, att2_f, att3_f], axis=1)
    _out = Dense(n_classes, 'softmax')(f_concat)
    model = Model(_inputs, _out)
    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              verbose=1,
              callbacks=[save_model_loss])


def visualization(nums=5):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data(r'D:\pyproject\data\cifar-10-batches-py')
    x_test = x_test / 255.0
    n_classes = 10

    _inputs = Input(shape=(32, 32, 3), name='input')
    _layer = ConvBlock(64, 3)(_inputs)
    _layer = ConvBlock(128, 3)(_layer)
    c1 = ConvBlock(256, 3, pooling=True)(_layer)
    c2 = ConvBlock(512, 3, pooling=True)(c1)
    c3 = ConvBlock(512, 3, pooling=True)(c2)
    _layer = ConvBlock(512, 3, pooling=True)(c3)
    _layer = ConvBlock(512, 3, pooling=True)(_layer)
    _layer = Flatten()(_layer)
    _g = Dense(512, activation='relu')(_layer)
    att1_f, att1_map = Attention((16, 16, 256), 512, method='pc')([c1, _g])
    att2_f, att2_map = Attention((8, 8, 512), 512, method='pc')([c2, _g])
    att3_f, att3_map = Attention((4, 4, 512), 512, method='pc')([c3, _g])
    f_concat = tf.concat([att1_f, att2_f, att3_f], axis=1)
    _out = Dense(n_classes, 'softmax')(f_concat)
    model = Model(_inputs, [_out, att1_map, att2_map, att3_map])
    model.load_weights(r'D:\pyproject\data\CIFAR10-tensorflow\vgg-att.h5')
    # model.compile()
    index_lst = np.arange(len(x_test))
    np.random.shuffle(index_lst)
    x_test = x_test[index_lst]
    x_test = x_test[0:nums]
    y_test = y_test[index_lst]
    y_test = y_test[0:nums]
    pred, att1_map, att2_map, att3_map, = model.predict(x_test)
    pred = tf.argmax(pred, axis=1)
    att1_map = tf.squeeze(tf.image.resize(tf.expand_dims(att1_map, -1), (32, 32)), -1)
    att2_map = tf.squeeze(tf.image.resize(tf.expand_dims(att2_map, -1), (32, 32)), -1)
    att3_map = tf.squeeze(tf.image.resize(tf.expand_dims(att3_map, -1), (32, 32)), -1)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(nums):
        img = x_test[i]
        plt.subplot(141)
        plt.imshow(img)
        plt.title('Img')
        plt.subplot(142)
        plt.imshow(img)
        plt.imshow(att1_map[i], alpha=0.4, cmap='rainbow')
        plt.title('att1_map')
        plt.subplot(143)
        plt.imshow(img)
        plt.imshow(att2_map[i], alpha=0.4, cmap='rainbow')
        plt.title('att2_map')
        plt.subplot(144)
        plt.imshow(img)
        plt.imshow(att3_map[i], alpha=0.4, cmap='rainbow')
        plt.title('att3_map')
        plt.suptitle(f'Prediction={classes[pred[i]]} True={classes[y_test[i][0]]}')
        plt.show()


if __name__ == '__main__':
    # main()
    visualization()