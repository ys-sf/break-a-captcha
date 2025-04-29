# Breaking CAPTCHA with Machine Learning

Ce projet a pour objectif de casser des CAPTCHA en utilisant un modèle de deep learning (CNN) multi-sorties.

## Structure du projet

```
captcha_breaker_project/
├── data/
│   ├── raw/          # Captchas bruts
│   ├── processed/    # Captchas nettoyés et segmentés
│   └── test/         # Captchas de test générés
├── models/           # Modèles sauvegardés
│   ├── captcha_model.h5
│   └── captcha_model_final.h5
├── results/          # Résultats d'évaluation
│   ├── evaluation_summary.txt
│   ├── prediction_examples.png
│   └── batch_results.csv
├── src/              # Code source
│   ├── data_preparation.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── captcha_solver.py
│   ├── batch_solver.py
│   └── generate_test_data.py
├── tests/            # Tests unitaires
├── requirements.txt  # Dépendances Python
└── README.md         # Ce fichier
```

##  Installation

Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Préparation des données

1. Télécharger un dataset de CAPTCHA (ex : version 2 sur Kaggle)
2. Placer les images brutes dans `data/raw/`
3. Lancer :
   ```bash
   python src/data_preparation.py
   ```
   pour obtenir les tableaux `X_train`, `X_test`, `y_train`, `y_test`.

## Entraînement du modèle

**Lancer le script d'entraînement :**
```bash
python src/model_training.py
```

Ce script :

- Construit un réseau CNN avec plusieurs couches convolutives
- Utilise une architecture multi-sorties (une par position de caractère)
- Applique des techniques de régularisation (Dropout) pour éviter le surapprentissage
- Implémente des callbacks pour sauvegarder le meilleur modèle et arrêter l'entraînement si nécessaire

### Architecture du modèle 

- **Entrée :** Image en niveaux de gris (50×200 pixels)
- **Extraction de caractéristiques :** 3 blocs de `Conv2D` + `MaxPooling2D`
- **Classification :** Couches denses avec 5 sorties indépendantes
- Chaque sortie correspond à un caractère du CAPTCHA

### Paramètres d'entraînement 

- **Optimizer :** Adam
- **Loss :** Categorical Crossentropy
- **Metric :** Accuracy
- **Epochs :** 20 (avec early stopping)
- **Batch size :** 32

### Résultat 

Le modèle entraîné est sauvegardé dans le dossier `models/` :

- `captcha_model.h5` (meilleur modèle selon la validation)
- `captcha_model_final.h5` (modèle final après entraînement complet)

##  Évaluation du modèle

**Lancer le script d'évaluation :**
```bash
python src/model_evaluation.py
```

Ce script :
- Évalue les performances du modèle sur le jeu de test
- Calcule l'accuracy globale et l'accuracy par position de caractère
- Visualise des exemples de prédictions correctes et incorrectes
- Génère un rapport d'évaluation complet

### Métriques évaluées

- **Accuracy globale des CAPTCHA :** pourcentage de CAPTCHA entièrement corrects (~50%)
- **Accuracy par caractère :** pourcentage de caractères correctement identifiés (~85%)
- **Accuracy par position :** performance pour chaque position (1ère position ~97%, positions du milieu ~78-80%)

Les résultats complets sont sauvegardés dans le dossier `results/` :
- `evaluation_summary.txt` : rapport détaillé des performances
- `prediction_examples.png` : visualisation d'exemples de prédictions

## 🔮 Résolution de CAPTCHA

### 1. Résolution individuelle

**Résoudre un CAPTCHA individuel :**
```bash
python src/captcha_solver.py path/to/captcha_image.png
```

Ce script :
- Charge le modèle entraîné
- Prétraite l'image fournie
- Prédit le texte du CAPTCHA
- Affiche le résultat

### 2. Traitement par lots

**Résoudre un lot de CAPTCHAs :**
```bash
python src/batch_solver.py data/test --output results/batch_results.csv
```

Ce script :
- Traite toutes les images dans un dossier spécifié
- Compare les prédictions avec les noms de fichiers (vérité terrain)
- Calcule le taux de réussite et les performances
- Génère un rapport CSV détaillé

Options disponibles :
- `--output` : chemin du fichier de résultats (CSV)
- `--model` : modèle à utiliser
- `--limit` : nombre maximum d'images à traiter

## 🔧 Génération de données de test

Pour générer un jeu de données de test synthétique :

```bash
python src/generate_test_data.py --count 200 --noise 0.4 --distortion 0.5
```

Paramètres configurables :
- `--count` : nombre d'images à générer
- `--length` : longueur du texte CAPTCHA
- `--noise` : niveau de bruit (0.0 à 1.0)
- `--distortion` : niveau de distorsion (0.0 à 1.0)
- `--width`, `--height` : dimensions des images

Les CAPTCHAs générés sont sauvegardés dans `data/test/` avec leur texte comme nom de fichier.

## 📈 Résultats et performances

Avec le modèle actuel :
- **Accuracy globale :** ~50% des CAPTCHAs sont entièrement résolus
- **Accuracy par caractère :** ~85% des caractères sont correctement identifiés
- **Performance par position :** meilleure pour le premier caractère (97%), plus faible au milieu (78-80%)
- **Temps de traitement :** ~100ms par CAPTCHA (sur CPU)

# Interface Web pour CAPTCHA Solver
J'ai créé une interface web complète qui permet d'utiliser facilement votre modèle de reconnaissance de CAPTCHA. Voici ce que j'ai développé:

1. Backend avec Flask (app.py)
Ce fichier Python:

Configure un serveur web avec Flask
Charge votre modèle de reconnaissance de CAPTCHA
Crée une API pour traiter les images téléchargées
Prétraite les images et utilise le modèle pour prédire le texte
Renvoie les résultats au frontend

2. Interface utilisateur HTML (templates/index.html)
Cette page web:

Permet de télécharger une image CAPTCHA
Affiche les résultats de la prédiction
Présente les statistiques de performance du modèle
Offre une interface responsive et moderne avec Bootstrap

3. Styles CSS (static/css/style.css)
Les styles:

Rendent l'interface visuelle et attrayante
Ajoutent des animations et des transitions
Garantissent une bonne lisibilité sur différents appareils

4. Logique JavaScript (static/js/script.js)
Ce script:

Gère l'envoi du formulaire de téléchargement
Communique avec l'API Flask
Affiche les résultats et les messages d'erreur
Ajoute des interactions utilisateur (animations, reset)

Comment utiliser cette interface:

Installez les dépendances:
bashpip install flask opencv-python tensorflow numpy

Créez la structure de dossiers selon le README fourni:
```
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── uploads/
└── models/
    └── captcha_model.h5
```

Copiez vos fichiers dans ces dossiers:

Votre modèle entraîné dans models/
Les fichiers que je viens de créer dans leurs emplacements respectifs


Lancez l'application:
bashpython app.py

Accédez à l'interface dans votre navigateur:
http://127.0.0.1:5000/


Cette interface web offre plusieurs avantages:

Elle est facile à utiliser, même pour des personnes non techniques
Elle présente visuellement les résultats et les statistiques
Elle peut être déployée sur un serveur pour une utilisation à distance
Elle peut servir de base pour une application plus complète

Vous pouvez également étendre cette interface pour:

Traiter plusieurs images simultanément
Sauvegarder un historique des prédictions
Ajouter des options pour télécharger les résultats
Intégrer des mécanismes de feedback pour améliorer le modèle
