import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DATA_PATH = 'data/raw'
IMG_HEIGHT = 50
IMG_WIDTH = 200
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def load_data(path=RAW_DATA_PATH):
    X = []  # Liste pour stocker les images
    y = []  # Liste pour stocker les labels
    
    for file_name in tqdm(os.listdir(path)):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            img_path = os.path.join(path, file_name)
            # Extraire le label du nom de fichier (en supposant que le format est 'label.png')
            label = os.path.splitext(file_name)[0]
            
            # Lecture en niveaux de gris
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Vérification de la validité de l'image
        
                # Redimensionnement à la taille définie
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        

            # Normalisation des pixels entre 0 et 1
            img_norm = img_resized.astype(np.float32) / 255.0
            
            # Vérifiez que tous les caractères du label sont dans CHAR_LIST
            valid_label = True
            for char in label:
                if char not in CHAR_LIST:
                    print(f"Warning: Caractère '{char}' non trouvé dans CHAR_LIST pour l'image {img_path}")
                    valid_label = False
                    break
            
            if valid_label:
                X.append(img_norm)  # Ajoute l'image transformée
                y.append(label)     # Ajoute le label extrait du nom de fichier
    
    X = np.array(X)
    X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)  # Reshape pour les réseaux CNN
    
    return X, y

def encode_labels(labels, max_length=5):
    encoded_labels = []  # Liste pour stocker les encodages
    
    for label in labels:
        # Initialiser un tableau de zéros
        label_array = np.zeros((max_length, len(CHAR_LIST)))
        
        # One-hot encoding pour chaque caractère du label
        for i, char in enumerate(label):
            if i >= max_length:  # Si le label est plus long que max_length, tronquer
                break
                
            try:
                idx = CHAR_LIST.index(char)
                label_array[i, idx] = 1  # Correction: i au lieu de 1
            except ValueError:
                print(f"Warning: Caractère '{char}' non trouvé dans CHAR_LIST")
                # On peut utiliser un caractère par défaut ou simplement continuer
                continue
                
        encoded_labels.append(label_array)
    
    return np.array(encoded_labels)
        
def prepare_dataset(test_size=0.2):
    X, y = load_data()
    
    if len(X) == 0:
        raise ValueError("Aucune donnée n'a été chargée. Vérifiez votre dossier de données.")
    
    y_encoded = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    print(f"Dataset prepared: {len(X_train)} train samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    prepare_dataset()