import os
import json
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class AudioClassifier:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.optimizer = None
        self.history = {"class_0_count": 0, "class_1_count": 0}
        self.history_file = f"data/{self.name}_history.json"

    def init_nn(self):
        model_file = f"data/{self.name}.h5"
        if os.path.exists(model_file):
            self.model = models.load_model(model_file)
            self.optimizer = tf.keras.optimizers.Adam()
            self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.load_history()
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

    def extract_features(self, audio_file, n_mfcc=13):
        audio, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfccs.T

    def pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = np.zeros((len(sequences), max_len, 13))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq), :] = seq
        return padded_sequences

    def classify(self, audio_file):
        features = self.extract_features(audio_file)
        features = np.expand_dims(features, axis=0)
        probabilities = self.model.predict(features)
        class_index = np.argmax(probabilities)
        confidence = probabilities[0][class_index]
        return class_index, confidence

    def update_history(self, labels):
        self.history["class_0_count"] += int(np.sum(labels == 0))
        self.history["class_1_count"] += int(np.sum(labels == 1))

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
