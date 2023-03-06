from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow import keras


def res_cnn(vocab_total_len, cov_dim):
    input_tensor = Input(shape=(vocab_total_len, cov_dim))
    x = Convolution1D(filters=cov_dim, kernel_size=3, padding='same', activation='elu')(input_tensor)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=cov_dim, kernel_size=3, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = layers.add([input_tensor, x])
    output = Activation('elu')(x)
    #output = BatchNormalization()(x)
    res_1d = keras.Model(inputs=[input_tensor], outputs=[output])
    return res_1d

def Discriminator(vocab_de_len, vocab_en_len, vocab_de_size, vocab_en_size, embedding_dim, cov_dim):
    input_tensor1 = Input(shape=(vocab_de_len, vocab_de_size))
    input_tensor2 = Input(shape=(vocab_en_len, vocab_en_size))
    x1 = Dense(vocab_de_size, activation='elu')(input_tensor1)
    x1 = Convolution1D(embedding_dim, 2, padding='same')(x1)

    x2 = Dense(vocab_en_size, activation='elu')(input_tensor2)
    x2 = Convolution1D(embedding_dim, 2, padding='same')(x2)
    tensor_combine = Concatenate(axis=1)([x1,x2])
    #
    x = Dense(vocab_de_len + vocab_en_len, activation='elu')(tensor_combine)
    x = Convolution1D(cov_dim, 2, padding='same')(x)
    x = res_cnn(vocab_de_len + vocab_en_len, cov_dim)(x)
    x = res_cnn(vocab_de_len + vocab_en_len, cov_dim)(x)
    x = res_cnn(vocab_de_len + vocab_en_len, cov_dim)(x)
    x = res_cnn(vocab_de_len + vocab_en_len, cov_dim)(x)
    x = res_cnn(vocab_de_len + vocab_en_len, cov_dim)(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu')(x)
    x = Dense(256, activation='elu')(x)
    x = Dense(1)(x)
    return keras.Model(inputs=[input_tensor1,input_tensor2], outputs=[x])