import tensorflow as tf
from tensorflow.keras import layers, models

class Decoder:
    def __init__(self, encoded_shape=(128,), original_shape=(200, 200, 3), num_filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'):
        self.encoded_shape = encoded_shape
        self.original_shape = original_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.decoder_model = self._build_decoder()

    def _build_decoder(self):
        inputs = layers.Input(shape=self.encoded_shape)
        
        # Camada Densa inicial que reconstrói a forma de entrada das camadas convolucionais
        x = layers.Dense(self.num_filters * 4 * (self.original_shape[0] // 8) * (self.original_shape[1] // 8), activation=self.activation)(inputs)
        x = layers.Reshape((self.original_shape[0] // 8, self.original_shape[1] // 8, self.num_filters * 4))(x)
        
        # Camadas Conv2DTranspose para aumentar as dimensões espaciais
        x = layers.Conv2DTranspose(self.num_filters * 2, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(self.num_filters, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(self.num_filters // 2, self.kernel_size, strides=self.strides, activation=self.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Camada final para reconstruir a imagem
        x = layers.Conv2DTranspose(self.original_shape[2], self.kernel_size, strides=(1, 1), activation='sigmoid', padding='same')(x)
        
        decoded = layers.Reshape(self.original_shape)(x)
        return models.Model(inputs, decoded)

    def forward(self, x):
        return self.decoder_model(x)

# Exemplo de uso:
# encoded_shape = (256,)  # Exemplo de entrada: vetor codificado de tamanho 256
# original_shape = (128, 128, 3)  # Forma original da imagem RGB 128x128
# num_filters = 32  # Número de filtros nas camadas convolucionais
# kernel_size = (3, 3)  # Tamanho do kernel
# strides = (2, 2)  # Passos da convolução

# decoder = Decoder(encoded_shape, original_shape, num_filters, kernel_size, strides)
# decoder.summary()

# Exemplo de entrada codificada
# encoded_input = tf.random.normal([1, 256])
# decoded_output = decoder.forward(encoded_input)
# print("Decoded output shape:", decoded_output.shape)