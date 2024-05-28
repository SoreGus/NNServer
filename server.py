from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)  # Habilitar CORS

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

audio_classifier = None

@app.route('/initNN', methods=['POST'])
def init_nn():
    global audio_classifier
    data = request.get_json()
    if 'name' not in data:
        return "Nome da rede neural não fornecido.", 400
    name = data['name']
    audio_classifier = AudioClassifier(name)
    audio_classifier.init_nn()
    return "Rede neural inicializada com sucesso!"

@app.route('/saveNN', methods=['GET'])
def save_nn():
    if not audio_classifier:
        return "Nenhuma rede neural inicializada. Por favor, chame o método 'initNN' primeiro."
    return audio_classifier.save_nn()

@app.route('/trainNN', methods=['POST'])
def train_nn():
    if not audio_classifier:
        return "Nenhuma rede neural inicializada. Por favor, chame o método 'initNN' primeiro."
    audio_files = request.files.getlist('audio_files')
    labels = request.form.getlist('labels')
    if len(audio_files) != len(labels):
        return "Número de arquivos de áudio e rótulos não correspondem.", 400
    labels = [int(label) for label in labels]
    
    audio_file_paths = []
    for audio_file in audio_files:
        file_path = os.path.join('temp', audio_file.filename)
        audio_file.save(file_path)
        audio_file_paths.append(file_path)

    audio_classifier.train_nn(audio_file_paths, labels)
    
    for file_path in audio_file_paths:
        os.remove(file_path)
        
    return "Treinamento concluído com sucesso!"

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_file_path = f'temp/{audio_file.filename}'
    audio_file.save(audio_file_path)

    try:
        class_index, confidence = audio_classifier.classify(audio_file_path)
        return jsonify({'class': int(class_index), 'confidence': float(confidence)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=8080, debug=True)  # Altere o host para 0.0.0.0 para permitir conexões externas
