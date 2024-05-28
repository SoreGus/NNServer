import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pydub import AudioSegment

class AudioClassifier:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.optimizer = None

    def init_nn(self):
        model_file = f"{self.name}.h5"
        if os.path.exists(model_file):
            self.model = models.load_model(model_file)
            self.optimizer = tf.keras.optimizers.Adam()
            self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = self.create_nn()
        return self.model

    def create_nn(self):
        self.optimizer = tf.keras.optimizers.Adam()
        model = models.Sequential([
            layers.InputLayer(input_shape=(None, 13)),
            layers.LSTM(64),
            layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def save_nn(self):
        if not self.model:
            return "Rede neural não foi inicializada. Por favor, chame o método 'initNN' primeiro."
        self.model.save(f"data/{self.name}.h5")
        return f"Rede neural {self.name} salva com sucesso!"

    def train_nn(self, audio_files, labels):
        mfccs = [self.extract_features(audio) for audio in audio_files]
        mfccs_padded = self.pad_sequences(mfccs)
        labels = np.array(labels)
        self.model.fit(mfccs_padded, labels, epochs=10, batch_size=32)

    def load_audio(self, file_path):
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        return samples, audio.frame_rate

    def extract_features(self, audio_file, n_mfcc=13):
        audio, sr = self.load_audio(audio_file)
        mfccs = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
        return mfccs.T

    def pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = np.zeros((len(sequences), max_len, 13))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq), :] = seq
        return padded_sequences

    def classify(self, audio_file):
        # Pré-processamento do áudio
        features = self.extract_features(audio_file)
        features = np.expand_dims(features, axis=0)  # Ajustar a forma para a previsão

        # Obter as probabilidades das classes
        probabilities = self.model.predict(features)
        class_index = np.argmax(probabilities)
        confidence = probabilities[0][class_index]

        return class_index, confidence
