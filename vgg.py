import os
from tensorflow.keras import Input, Model, optimizers, datasets
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Layer
from tensorflow.keras import utils

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
    model_path_loss = output_path + r"\vgg.h5"
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
    _layer = Dense(512, activation='relu')(_layer)
    _outputs = Dense(n_classes, activation='softmax')(_layer)
    model = Model(_inputs, _outputs)

    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              verbose=1,
              callbacks=[save_model_loss])


def reload(_path=r'D:\pyproject\data\CIFAR10-tensorflow'):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data(r'D:\pyproject\data\cifar-10-batches-py')
    x_test = x_test / 255.0
    y_test = utils.to_categorical(y_test, num_classes=10)
    n_classes = 10
    model_path = _path + r"\vgg.h5"

    _inputs = Input(shape=(32, 32, 3), name='input')
    _layer = ConvBlock(64, 3)(_inputs)
    _layer = ConvBlock(128, 3)(_layer)
    c1 = ConvBlock(256, 3, pooling=True)(_layer)
    c2 = ConvBlock(512, 3, pooling=True)(c1)
    c3 = ConvBlock(512, 3, pooling=True)(c2)
    _layer = ConvBlock(512, 3, pooling=True)(c3)
    _layer = ConvBlock(512, 3, pooling=True)(_layer)
    _layer = Flatten()(_layer)
    _layer = Dense(512, activation='relu')(_layer)
    _outputs = Dense(n_classes, activation='softmax')(_layer)
    model = Model(_inputs, _outputs)
    model.load_weights(model_path)
    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.evaluate(x_test, y_test)
    # y = model.predict(x_test[:5])
    # print(y.shape)


if __name__ == '__main__':
    # main()
    reload()