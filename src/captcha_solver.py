import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

from data_preparation import CHAR_LIST, IMG_HEIGHT, IMG_WIDTH

def load_trained_model(model_path="models/captcha_model.h5"):
    """Charge le modèle entraîné"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas.")
    
    model = load_model(model_path)
    return model

def preprocess_image(image_path):
    """Prétraite une image pour la prédiction"""
    # Vérification de l'existence du fichier
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas.")
    
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

def solve_captcha(image_path, model_path="models/captcha_model.h5"):
    """Résout un CAPTCHA à partir d'une image"""
    # Chargement du modèle
    model = load_trained_model(model_path)
    
    # Prétraitement de l'image
    img_input = preprocess_image(image_path)
    
    # Prédiction
    predictions = model.predict(img_input)
    
    # Décodage de la prédiction
    captcha_text = decode_predictions(predictions)
    
    return captcha_text

def main():
    """Fonction principale"""
    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Résolution automatique de CAPTCHA")
    parser.add_argument("image_path", help="Chemin vers l'image CAPTCHA à résoudre")
    parser.add_argument("--model", default="models/captcha_model.h5", help="Chemin vers le modèle entraîné")
    args = parser.parse_args()
    
    try:
        # Résolution du CAPTCHA
        captcha_text = solve_captcha(args.image_path, args.model)
        print(f"CAPTCHA résolu: {captcha_text}")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()