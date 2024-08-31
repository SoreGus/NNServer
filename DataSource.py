import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataSource:
    def __init__(self, target_size=(200, 200)):  # Ajustado para 200x200 pixels
        self.target_size = target_size

    def inputImagesFromPath(self, directory_path, color_mode='rgb'):
        images = []
        labels = []
        class_names = []
        class_index = {}

        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, file_name)

                    # Extrai o nome da classe a partir do caminho relativo
                    class_name = os.path.basename(os.path.dirname(file_path))
                    
                    if class_name not in class_index:
                        class_index[class_name] = len(class_index)
                        class_names.append(class_name)

                    label = class_index[class_name]

                    # Carrega a imagem, redimensiona para 200x200 e converte em array
                    image = load_img(file_path, target_size=self.target_size, color_mode=color_mode)
                    image_array = img_to_array(image)

                    # Normaliza a imagem (opcional)
                    image_array = image_array / 255.0

                    # Adiciona a imagem e o rótulo à lista
                    images.append(image_array)
                    labels.append(label)

        # Converte listas para arrays do numpy
        images = np.array(images)
        labels = np.array(labels)

        return images, labels, class_names

# Exemplo de uso:
# directory_path = 'caminho/para/seu/diretorio/de/imagens'

# data_source = DataSource()  # Agora ajusta para 200x200 pixels
# images, labels, class_names = data_source.inputImagesFromPath(directory_path)

# print("Número de imagens carregadas:", len(images))
# print("Classes encontradas:", class_names)