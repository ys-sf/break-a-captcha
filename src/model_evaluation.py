import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import cv2

from data_preparation import prepare_dataset, CHAR_LIST, IMG_HEIGHT, IMG_WIDTH

def load_trained_model(model_path="models/captcha_model.h5"):
    """Charge le modèle entraîné"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas. Entraînez d'abord le modèle.")
    
    model = load_model(model_path)
    return model

def decode_predictions(predictions, char_list=CHAR_LIST):
    """
    Convertit les prédictions de probabilités en texte
    
    Args:
        predictions: liste de tableaux de prédictions, un par position
        char_list: liste des caractères possibles
    
    Returns:
        Liste de textes prédits
    """
    # Nombre d'échantillons
    batch_size = predictions[0].shape[0]
    
    # Liste des résultats décodés
    decoded_predictions = []
    
    for i in range(batch_size):
        # Extraire les caractères de chaque position pour cet échantillon
        captcha_text = ""
        for pos_preds in predictions:
            char_idx = np.argmax(pos_preds[i])
            captcha_text += char_list[char_idx]
        decoded_predictions.append(captcha_text)
    
    return decoded_predictions

def decode_labels(y, char_list=CHAR_LIST):
    """Convertit les labels one-hot en texte"""
    # Nombre d'échantillons
    batch_size = y[0].shape[0] if isinstance(y, list) else y.shape[0]
    
    # Préparation des données selon le format
    if isinstance(y, list):
        # Si y est déjà une liste de tableaux par position
        y_pos_arrays = y
    else:
        # Si y est un tableau 3D (nb_samples, max_length, num_classes)
        max_length = y.shape[1]
        y_pos_arrays = [y[:, i, :] for i in range(max_length)]
    
    # Décodage
    decoded_labels = []
    for i in range(batch_size):
        captcha_text = ""
        for pos_array in y_pos_arrays:
            char_idx = np.argmax(pos_array[i])
            captcha_text += char_list[char_idx]
        decoded_labels.append(captcha_text)
    
    return decoded_labels

def calculate_accuracy(y_true, y_pred):
    """
    Calcule l'accuracy globale et par position
    
    Args:
        y_true: liste des textes réels
        y_pred: liste des textes prédits
    
    Returns:
        Dict avec les métriques d'accuracy
    """
    total = len(y_true)
    correct_captchas = 0
    correct_chars = [0] * len(y_true[0])  # Supposons que tous les captchas ont la même longueur
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_captchas += 1
        
        for i, (t, p) in enumerate(zip(true, pred)):
            if t == p:
                correct_chars[i] += 1
    
    # Calcul des métriques
    captcha_accuracy = correct_captchas / total
    char_accuracies = [correct / total for correct in correct_chars]
    overall_char_accuracy = sum(correct_chars) / (total * len(correct_chars))
    
    return {
        "captcha_accuracy": captcha_accuracy,
        "char_accuracies": char_accuracies,
        "overall_char_accuracy": overall_char_accuracy
    }

def visualize_predictions(X_test, y_true, y_pred, num_samples=5):
    """Visualise quelques prédictions pour comparer avec les valeurs réelles"""
    idxs = np.random.choice(len(y_true), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(idxs):
        plt.subplot(num_samples, 1, i+1)
        
        # Convertir l'image au format d'affichage
        img = X_test[idx].reshape(IMG_HEIGHT, IMG_WIDTH)
        plt.imshow(img, cmap='gray')
        
        # Afficher le titre avec vrai/prédit
        plt.title(f"Vrai: {y_true[idx]} | Prédit: {y_pred[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/prediction_examples.png")
    plt.close()

def evaluate_model():
    """Fonction principale d'évaluation"""
    # Charger les données
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # S'assurer que le dossier results existe
    os.makedirs("results", exist_ok=True)
    
    # Charger le modèle
    model = load_trained_model()
    
    # Préparer les données de test pour l'évaluation
    if isinstance(y_test, list):
        y_test_list = y_test
    else:
        # Si y_test est un tableau 3D (nb_samples, max_length, num_classes)
        max_length = y_test.shape[1] 
        y_test_list = [y_test[:, i, :] for i in range(max_length)]
    
    # Évaluer le modèle
    print("Évaluation du modèle...")
    evaluation = model.evaluate(X_test, y_test_list, verbose=1)
    
    # Afficher les scores
    print("\nScores d'évaluation:")
    for i, metric_name in enumerate(model.metrics_names):
        print(f"{metric_name}: {evaluation[i]:.4f}")
    
    # Faire des prédictions
    print("\nGénération des prédictions...")
    predictions = model.predict(X_test)
    
    # Décoder les prédictions et les vrais labels
    y_pred_text = decode_predictions(predictions)
    y_true_text = decode_labels(y_test)
    
    # Calcul de l'accuracy
    accuracy_metrics = calculate_accuracy(y_true_text, y_pred_text)
    
    # Afficher les résultats
    print("\nRésultats d'accuracy:")
    print(f"Accuracy globale des CAPTCHA: {accuracy_metrics['captcha_accuracy']:.4f}")
    print(f"Accuracy globale par caractère: {accuracy_metrics['overall_char_accuracy']:.4f}")
    print("Accuracy par position:")
    for i, acc in enumerate(accuracy_metrics['char_accuracies']):
        print(f"  Position {i+1}: {acc:.4f}")
    
    # Visualiser quelques prédictions
    print("\nVisualisation des exemples de prédictions...")
    visualize_predictions(X_test, y_true_text, y_pred_text)
    
    # Sauvegarder les résultats
    with open("results/evaluation_summary.txt", "w") as f:
        f.write("ÉVALUATION DU MODÈLE DE RECONNAISSANCE CAPTCHA\n")
        f.write("==============================================\n\n")
        f.write(f"Accuracy globale des CAPTCHA: {accuracy_metrics['captcha_accuracy']:.4f}\n")
        f.write(f"Accuracy globale par caractère: {accuracy_metrics['overall_char_accuracy']:.4f}\n")
        f.write("Accuracy par position:\n")
        for i, acc in enumerate(accuracy_metrics['char_accuracies']):
            f.write(f"  Position {i+1}: {acc:.4f}\n")
    
    print("\nÉvaluation terminée. Résultats sauvegardés dans le dossier 'results/'")
    
if __name__ == "__main__":
    evaluate_model()