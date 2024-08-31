import tensorflow as tf
from tensorflow.keras import layers, models

class Encoder:
    def __init__(self, input_shape, num_filters, kernel_size, strides, activation='relu'):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.encoder_model = self._build_encoder()

    def _build_encoder(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(self.num_filters, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(self.num_filters * 2, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(self.num_filters * 4, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        encoded = layers.Dense(128, activation=self.activation)(x)

        return models.Model(inputs, encoded)

    def forward(self, x):
        return self.encoder_model(x)

# Exemplo de uso:
# input_shape = (128, 128, 3)  # Exemplo de entrada: imagem RGB 128x128
# num_filters = 32  # Número de filtros nas camadas convolucionais
# kernel_size = (3, 3)  # Tamanho do kernel
# strides = (2, 2)  # Passos da convolução

# encoder = Encoder(input_shape, num_filters, kernel_size, strides)
# encoder.summary()

# Exemplo de entrada de imagem
# input_image = tf.random.normal([1, 128, 128, 3])
# encoded_output = encoder.forward(input_image)
# print("Encoded output shape:", encoded_output.shape)