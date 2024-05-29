from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from neural_network import AudioClassifier

app = Flask(__name__)
CORS(app)  # Habilitar CORS

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
        os.remove(audio_file_path)
        return jsonify({'class': int(class_index), 'confidence': float(confidence)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('data'):
        os.makedirs('data')
    app.run(host='0.0.0.0', port=8080, debug=True)
