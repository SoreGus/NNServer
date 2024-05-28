import os
import json
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
        self.history_file = f"data/history.json"
        self.class_0_count = 0
        self.class_1_count = 0
        self.history = self.load_history()

    def init_nn(self):
        model_file = f"data/{self.name}.h5"
        if os.path.exists(model_file):
            self.model = models.load_model(model_file)
            self.optimizer = tf.keras.optimizers.Adam()
            self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.load_history_for_model()
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
        self.save_history()
        return f"Rede neural {self.name} salva com sucesso!"

    def train_nn(self, audio_files, labels):
        mfccs = [self.extract_features(audio) for audio in audio_files]
        mfccs_padded = self.pad_sequences(mfccs)
        labels = np.array(labels)
        self.model.fit(mfccs_padded, labels, epochs=10, batch_size=32)
        self.update_history(labels)

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

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def load_history_for_model(self):
        if self.name in self.history:
            self.class_0_count = self.history[self.name].get('class_0_count', 0)
            self.class_1_count = self.history[self.name].get('class_1_count', 0)
        else:
            self.class_0_count = 0
            self.class_1_count = 0

    def update_history(self, labels):
        self.class_0_count += np.sum(labels == 0)
        self.class_1_count += np.sum(labels == 1)
        self.history[self.name] = {
            'class_0_count': self.class_0_count,
            'class_1_count': self.class_1_count
        }

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
