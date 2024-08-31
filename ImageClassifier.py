import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class ImageClassifier:
    def __init__(self):
        self.model = None

    def create_network(self, kernels_layer_1=2, kernels_layer_2=2, kernels_layer_3=2, kernels_layer_4=2, dense_units=100, output_units=45):
        self.model = models.Sequential()

        # Primeira camada de Convolução e MaxPooling
        self.model.add(layers.Conv2D(kernels_layer_1, (3, 3), activation='relu', padding='valid', input_shape=(200, 200, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Segunda camada de Convolução e MaxPooling
        self.model.add(layers.Conv2D(kernels_layer_2, (3, 3), activation='relu', padding='valid'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Terceira camada de Convolução e MaxPooling
        self.model.add(layers.Conv2D(kernels_layer_3, (3, 3), activation='relu', padding='valid'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Quarta camada de Convolução e MaxPooling
        self.model.add(layers.Conv2D(kernels_layer_4, (3, 3), activation='relu', padding='valid'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Flatten para converter o tensor em um vetor para a camada densa
        self.model.add(layers.Flatten())

        # Camada densa (hidden layer)
        self.model.add(layers.Dense(dense_units, activation='relu'))

        # Camada de saída
        self.model.add(layers.Dense(output_units, activation='softmax'))

        # Compilando o modelo
        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

        # Exibindo o resumo do modelo
        self.model.summary()

    def save_network(self, file_path):
        if self.model:
            # Salvando no formato Keras nativo
            self.model.save(file_path)
        else:
            print("Nenhuma rede criada para salvar.")

    def load_network(self, file_path):
        if os.path.exists(file_path):
            self.model = models.load_model(file_path)
        else:
            print(f"O arquivo {file_path} não existe.")

    def train_on_dir(self, data_dir, batch_size=32, epochs=10):
        if not self.model:
            print("Crie a rede antes de treinar.")
            return

        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(200, 200),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse',
            subset='training')

        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(200, 200),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation')

        self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator)

    def test_on_dir(self, data_dir, batch_size=32):
        if not self.model:
            print("Crie ou carregue uma rede antes de testar.")
            return

        datagen = ImageDataGenerator(rescale=1./255)

        test_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(200, 200),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False)  # Não misturar para manter a correspondência das previsões

        # Avaliando o modelo nos dados de teste
        loss, accuracy = self.model.evaluate(test_generator)
        print(f"Acurácia no conjunto de testes: {accuracy*100:.2f}%")

    def predict(self, image):
        if not self.model:
            print("Crie ou carregue uma rede antes de fazer predições.")
            return None

        image = tf.image.resize(image, (200, 200))
        image = tf.image.rgb_to_grayscale(image)
        image = image / 255.0  # Normalizando a imagem
        image = tf.expand_dims(image, axis=0)  # Adicionando a dimensão do batch

        prediction = self.model.predict(image)
        predicted_class = tf.argmax(prediction, axis=1).numpy()

        return predicted_class

nnName = "mammals"

classifier = ImageClassifier()

classifier.create_network(kernels_layer_1=4, kernels_layer_2=16, kernels_layer_3=32, kernels_layer_4=64, dense_units=200, output_units=45)

classifier.train_on_dir('/Users/gustavosore/Documents/Projects/SML/Tests/SoreMachineLearningTests/Data/mammals', batch_size=32, epochs=20)

classifier.save_network(f'./{nnName}.keras')

classifier.test_on_dir('/Users/gustavosore/Documents/Projects/SML/Tests/SoreMachineLearningTests/Data/mammals')