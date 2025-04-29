from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import uuid
import base64

# Importation des constantes depuis votre projet
from src.data_preparation import CHAR_LIST, IMG_HEIGHT, IMG_WIDTH

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/captcha_model.h5'

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle
try:
    model = load_model(MODEL_PATH)
    print(f"Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

def preprocess_image(image_path):
    """Prétraite une image pour la prédiction"""
    # Chargement de l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger l'image {image_path}.")
    
    # Redimensionnement
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalisation
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Ajout des dimensions pour le batch et le canal
    img_input = img_norm.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    return img_input

def decode_predictions(predictions, char_list=CHAR_LIST):
    """Convertit les prédictions en texte"""
    captcha_text = ""
    for pos_preds in predictions:
        char_idx = np.argmax(pos_preds[0])
        captcha_text += char_list[char_idx]
    
    return captcha_text

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_captcha():
    """API pour résoudre un CAPTCHA"""
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    
    # Vérifier si le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Vérifier si le modèle est chargé
    if model is None:
        return jsonify({'error': 'Le modèle n\'est pas disponible. Veuillez vérifier la console du serveur.'}), 500
    
    try:
        # Sauvegarder l'image temporairement
        filename = str(uuid.uuid4()) + '.png'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Prétraiter l'image
        img_input = preprocess_image(filepath)
        
        # Faire la prédiction
        predictions = model.predict(img_input)
        
        # Décoder la prédiction
        captcha_text = decode_predictions(predictions)
        
        # Lire l'image pour l'afficher dans la réponse
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Nettoyage
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'captcha_text': captcha_text,
            'image': img_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)