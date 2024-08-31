import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import array_to_img
from Encoder import Encoder
from Decoder import Decoder
from DataSource import DataSource

class AutoEncoder:
    def __init__(self, input_shape=(200, 200, 3), encoded_shape=(128,), num_filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'):
        self.input_shape = input_shape
        self.encoded_shape = encoded_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

        # Inicializa Encoder e Decoder
        self.encoder = Encoder(input_shape, num_filters, kernel_size, strides, activation)
        self.decoder = Decoder(encoded_shape, input_shape, num_filters, kernel_size, strides, activation)
        
        # Constrói o modelo completo do AutoEncoder
        self.autoencoder_model = self._build_autoencoder()

    def _build_autoencoder(self):
        inputs = layers.Input(shape=self.input_shape)
        encoded = self.encoder.forward(inputs)
        print("Encoded output shape:", encoded.shape)  # Verificação da forma codificada
        decoded = self.decoder.forward(encoded)
        autoencoder_model = models.Model(inputs, decoded)
        autoencoder_model.compile(optimizer='adam', loss='mse')
        return autoencoder_model

    def trainOnPath(self, directory_path, batch_size=32, epochs=10, validation_split=0.2):
        data_source = DataSource(target_size=self.input_shape[:2])
        images, labels, class_names = data_source.inputImagesFromPath(directory_path)
        history = self.autoencoder_model.fit(images, images, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        return history

    def generateImageFromMatrix(self, matrix):
        decoded_image = self.decoder.forward(matrix)
        return decoded_image

    def saveGeneratedImage(self, image_array, filename='generated_image.png'):
        image = array_to_img(image_array)
        save_dir = 'data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)
        print(f"Image saved at: {save_path}")
        
    def save_model(self, path):
        self.autoencoder_model.save(path)
        
    def load_model(self, path):
        self.autoencoder_model = tf.keras.models.load_model(path)
        
    def summary(self):
        self.autoencoder_model.summary()

        # Definir a função que recria o modelo

# Criar uma instância do AutoEncoder
autoencoder = AutoEncoder()
autoencoder.summary()

# Treinar o modelo se necessário ou carregar um modelo salvo
try:
    # Se já tem um modelo salvo, carregue-o
    autoencoder.load_model('./data/autoEncoder.keras')
    print("Modelo carregado com sucesso.")
except Exception as e:
    # Se não tem, treine e salve o modelo
    print("Não foi possível carregar o modelo. Treinando um novo.")
    autoencoder.trainOnPath('/Users/gustavosore/Documents/Projects/SML/Tests/SoreMachineLearningTests/Data/mammals')
    autoencoder.save_model('./data/autoEncoder.keras')
    print("Modelo treinado e salvo.")

# Gerar 10 imagens usando matrizes aleatórias
encoded_shape = (128,)
for i in range(10):
    random_matrix = tf.random.normal([1, encoded_shape[0]])  # Gerar matriz aleatória
    generated_image = autoencoder.generateImageFromMatrix(random_matrix)  # Gerar imagem usando o decoder
    generated_image = tf.squeeze(generated_image, axis=0)  # Remover a dimensão do batch
    generated_image = (generated_image * 255).numpy().astype('uint8')  # Desnormalizar a imagem
    filename = f'generated_image_{i+1}.png'
    autoencoder.saveGeneratedImage(generated_image, filename)