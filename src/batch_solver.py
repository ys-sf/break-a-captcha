import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import time
import pandas as pd
from tqdm import tqdm

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

def process_batch(image_folder, output_file="batch_results.csv", model_path="models/captcha_model.h5", limit=None):
    """
    Traite un lot d'images CAPTCHA
    
    Args:
        image_folder: Dossier contenant les images CAPTCHA
        output_file: Fichier CSV pour enregistrer les résultats
        model_path: Chemin vers le modèle entraîné
        limit: Nombre maximum d'images à traiter (None = toutes)
    """
    # Vérification de l'existence du dossier
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Le dossier {image_folder} n'existe pas.")
    
    # Chargement du modèle
    print("Chargement du modèle...")
    model = load_trained_model(model_path)
    
    # Récupération des fichiers d'images
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if limit:
        image_files = image_files[:limit]
    
    total_images = len(image_files)
    print(f"Traitement de {total_images} images...")
    
    # Préparation des résultats
    results = []
    start_time = time.time()
    
    # Traitement des images avec une barre de progression
    for image_file in tqdm(image_files, desc="Résolution de CAPTCHAs"):
        image_path = os.path.join(image_folder, image_file)
        file_name = os.path.basename(image_path)
        expected = os.path.splitext(file_name)[0]  # Utilise le nom du fichier comme vérité terrain
        
        try:
            # Prétraitement de l'image
            img_input = preprocess_image(image_path)
            
            # Prédiction
            predictions = model.predict(img_input, verbose=0)  # Désactive l'affichage verbose
            
            # Décodage de la prédiction
            predicted = decode_predictions(predictions)
            
            # Vérification si la prédiction correspond au nom du fichier (vérité terrain)
            is_correct = predicted == expected
            
            # Enregistrement du résultat
            results.append({
                'image': file_name,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct
            })
            
        except Exception as e:
            # En cas d'erreur, on l'enregistre
            results.append({
                'image': file_name,
                'expected': expected,
                'predicted': "ERREUR",
                'correct': False,
                'error': str(e)
            })
    
    # Calcul du temps total
    total_time = time.time() - start_time
    avg_time = total_time / total_images if total_images > 0 else 0
    
    # Création du DataFrame et sauvegarde en CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Calcul des statistiques
    correct_count = df['correct'].sum()
    accuracy = correct_count / total_images if total_images > 0 else 0
    
    # Affichage des résultats
    print("\nRésultats de l'analyse par lot:")
    print(f"Total d'images traitées: {total_images}")
    print(f"CAPTCHAs correctement résolus: {correct_count} ({accuracy:.2%})")
    print(f"Temps total: {total_time:.2f} secondes")
    print(f"Temps moyen par image: {avg_time*1000:.2f} ms")
    print(f"Résultats sauvegardés dans: {output_file}")
    
    return df

def main():
    """Fonction principale"""
    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Résolution automatique par lot de CAPTCHAs")
    parser.add_argument("image_folder", help="Dossier contenant les images CAPTCHA à résoudre")
    parser.add_argument("--output", default="batch_results.csv", help="Fichier CSV pour enregistrer les résultats")
    parser.add_argument("--model", default="models/captcha_model.h5", help="Chemin vers le modèle entraîné")
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximum d'images à traiter")
    args = parser.parse_args()
    
    try:
        # Traitement du lot d'images
        process_batch(args.image_folder, args.output, args.model, args.limit)
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()