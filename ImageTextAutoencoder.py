import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

class ImageTextAutoencoder:
    def __init__(self, image_shape=(200, 200, 3), text_embedding_dim=256, latent_dim=128, max_vocab_size=10000, max_sequence_length=100):
        self.image_shape = image_shape
        self.text_embedding_dim = text_embedding_dim
        self.latent_dim = latent_dim
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.model = None
        self.tokenizer = None

    def build_model(self):
        # Encoder de Imagem
        image_input = Input(shape=self.image_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        latent_image = layers.Dense(self.latent_dim, activation='relu')(x)

        # Entrada de Texto
        text_input = Input(shape=(self.max_sequence_length,))
        x = layers.Embedding(self.max_vocab_size, self.text_embedding_dim, input_length=self.max_sequence_length)(text_input)
        x = layers.Flatten()(x)
        latent_text = layers.Dense(self.latent_dim, activation='relu')(x)

        # Combinação das Representações Latentes (Imagem + Texto)
        combined = layers.Add()([latent_image, latent_text])  # Você também pode experimentar layers.Concatenate() ou Dot()

        # Decoder para Gerar a Imagem
        x = layers.Dense(25 * 25 * 128, activation='relu')(combined)  # Supondo que a entrada original seja 200x200
        x = layers.Reshape((25, 25, 128))(x)
        x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        output_image = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)  # Saída 200x200x3

        # Definindo o Modelo
        self.model = models.Model(inputs=[image_input, text_input], outputs=output_image)

        # Compilando o Modelo
        self.model.compile(optimizer='adam', loss='mse')  # Usando mean squared error para a perda da imagem

        # Resumo do Modelo
        self.model.summary()

    def load_texts_and_create_embeddings(self, text_file_path):
        # Carregar textos do arquivo e gerar embeddings
        with open(text_file_path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        texts = [text.strip() for text in texts]  # Remover espaços em branco

        # Criar o tokenizer e ajustar ao vocabulário dos textos
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)

        # Converter os textos em sequências de inteiros
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Padronizar as sequências para que todas tenham o mesmo comprimento
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')

        return np.array(padded_sequences)

    def train_on_dir(self, image_dir, text_embeddings, batch_size=32, epochs=10):
        if self.model is None:
            print("O modelo não foi construído. Chame 'build_model()' primeiro.")
            return

        # Configurando o gerador de dados de imagem
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            image_dir,
            target_size=(200, 200),
            batch_size=batch_size,
            class_mode=None,  # Não usamos classes diretamente, o treinamento será supervisionado por texto
            subset='training',
            shuffle=False
        )

        validation_generator = datagen.flow_from_directory(
            image_dir,
            target_size=(200, 200),
            batch_size=batch_size,
            class_mode=None,  # Não usamos classes diretamente, o treinamento será supervisionado por texto
            subset='validation',
            shuffle=False
        )

        # Assumimos que text_embeddings está alinhado com a ordem de train_generator e validation_generator

        # Ajuste de dados para o treinamento
        self.model.fit(
            [train_generator, text_embeddings], 
            train_generator, 
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([validation_generator, text_embeddings], validation_generator)
        )

    def predict(self, image_input, text_input):
        if self.model is None:
            print("O modelo não foi construído. Chame 'build_model()' primeiro.")
            return None

        return self.model.predict([image_input, text_input])
    
autoencoder = ImageTextAutoencoder(max_vocab_size=5000, max_sequence_length=10)
autoencoder.build_model()
autoencoder.train_on_dir('/Users/gustavosore/Documents/Projects/SML/Tests/SoreMachineLearningTests/Data/mammals', batch_size=32, epochs=10)
