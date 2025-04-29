import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Reshape, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

from data_preparation import prepare_dataset, CHAR_LIST, IMG_HEIGHT, IMG_WIDTH

# Constantes
NUM_CLASSES = len(CHAR_LIST)
MAX_LENGTH = 5  # Longueur maximale du CAPTCHA

def build_model():
    # Input layer
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input')
    
    # Couches convolutives
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Flatten
    x = Flatten()(x)
    
    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layers - un pour chaque position du CAPTCHA
    outputs = []
    for i in range(MAX_LENGTH):
        name = f'char_{i}'
        outputs.append(Dense(NUM_CLASSES, activation='softmax', name=name)(x))
    
    # Création du modèle
    model = Model(inputs=input_img, outputs=outputs)
    
    # Compilation avec métriques pour chaque sortie
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'] * MAX_LENGTH  # Une métrique 'accuracy' pour chaque sortie
    )
    
    return model

def train_model(epochs=20, batch_size=32):
    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # Construction du modèle
    model = build_model()
    model.summary()
    
    # Reorganiser y_train et y_test pour les adapter au modèle multi-sorties
    # Si y_train est de forme (nb_samples, MAX_LENGTH, NUM_CLASSES)
    # Nous avons besoin de le convertir en liste de MAX_LENGTH éléments
    # Chaque élément étant un array de forme (nb_samples, NUM_CLASSES)
    
    y_train_list = [y_train[:, i, :] for i in range(MAX_LENGTH)]
    y_test_list = [y_test[:, i, :] for i in range(MAX_LENGTH)]
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/captcha_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrainement
    history = model.fit(
        X_train, y_train_list,
        validation_data=(X_test, y_test_list),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

if __name__ == "__main__":
    # S'assurer que le dossier models existe
    os.makedirs('models', exist_ok=True)
    
    # Entrainement du modèle
    model, history = train_model()
    
    # Sauvegarde du modèle final
    model.save('models/captcha_model_final.h5')
    
    print("Entraînement terminé et modèle sauvegardé !")