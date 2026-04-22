import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  #VGG16 model for transfer learning
from tensorflow.keras.models import load_model, Model      #To design and deploy deep learning models
from tensorflow.keras.utils import plot_model                           #To plot the model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.decomposition import PCA


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images.
    Input : images (list) : liste des images (couleur ou niveaux de gris)
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for i in range(len(images)):
        resized_img = cv2.resize(images[i], (128, 128))
        
        if len(resized_img.shape) > 2:
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = resized_img
            
        hist = cv2.calcHist(
            [gray_img.astype(np.float32)],
            [0],
            None,
            [16],
            [0, 256]  
        )
        
        hist = hist.flatten() / hist.sum() if hist.sum() > 0 else hist.flatten()
        descriptors.append(hist)
    return descriptors


def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images.
    Input : images (list) : liste des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    for img in images:
        img = cv2.resize(img, (128,128))
        fd = hog(
            img,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            visualize = False,
            channel_axis=-1
        )
        descriptors.append(fd)
            
    return descriptors



def compute_vgg16_features(images, layer='fc2'):
    """
    Applique le modèle VGG16 pré-entraîné à de nouvelles images.
    
    Args:
        images (list/numpy.ndarray): Images à traiter
        base_model_vgg16: Modèle VGG16 pré-entraîné
        layer (str): Nom de la couche à extraire
        
    Returns:
        numpy.ndarray: Descripteurs VGG16 pour les nouvelles images
    """ 
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
    
    # Initialiser la liste des descripteurs
    descriptors = []
    
    # Définir une taille de lot
    batch_size = 32
    
    # Traiter les images par lots
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        processed_images = []
        
        # Prétraiter chaque image
        for img in batch:
            # Redimensionner l'image
            img = cv2.resize(img, (224, 224))
            
            # Gérer les images en niveaux de gris
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
                
            processed_images.append(img)
        
        # Convertir le batch en array
        batch_array = np.array(processed_images)
        
        # Prétraiter les images pour VGG16
        batch_array = preprocess_input(batch_array)
        
        # Extraire les features
        batch_features = model.predict(batch_array, verbose=0)
        
        # Ajouter les features à la liste
        descriptors.extend(batch_features)
    
    # Retourner les descripteurs
    return np.array(descriptors)

def compute_resnet50_features(images):
    """
    Extrait les caractéristiques ResNet50 des images et prépare le modèle PCA.
    
    Args:
        images (list/numpy.ndarray): Images à traiter
        
    Returns:
        tuple: (features, pca_model, base_model)
            - features: Caractéristiques extraites après PCA
            - pca_model: Modèle PCA entraîné
            - base_model: Modèle ResNet50 utilisé
    """
    # Initialiser le modèle ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Initialiser la liste des caractéristiques
    features = []
    
    # Définir la taille de lot
    batch_size = 32
    
    # Traiter les images par lots
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        processed_images = []
        
        # Prétraiter chaque image
        for img in batch:
            # Redimensionner l'image
            img = cv2.resize(img, (224, 224))
            
            # Gérer les images en niveaux de gris
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
                
            processed_images.append(img)
        
        # Convertir le batch en array
        batch_array = np.array(processed_images)
        
        # Prétraiter les images pour ResNet50
        batch_array = preprocess_input(batch_array)
        
        # Extraire les features
        batch_features = base_model.predict(batch_array, verbose=0)
        
        # Ajouter les features à la liste
        features.extend(batch_features)
    
    # Convertir en array
    features = np.array(features)
    
    # Appliquer PCA pour réduire la dimensionnalité
    pca = PCA(n_components=100)
    reduced_features = pca.fit_transform(features)
    
    print(f"Variance expliquée par PCA (100 composantes): {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return reduced_features


def compute_ensemble_features(images, hist_features, hog_features, vgg_features, resnet_features):
    """
    Combine plusieurs types de descripteurs pour obtenir des caractéristiques optimales.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Normaliser chaque ensemble de caractéristiques
    scalers = {
        'hist': StandardScaler(),
        'hog': StandardScaler(),
        'vgg': StandardScaler(),
        'resnet': StandardScaler()
    }
    
    normalized_features = {
        'hist': scalers['hist'].fit_transform(hist_features),
        'hog': scalers['hog'].fit_transform(hog_features),
        'vgg': scalers['vgg'].fit_transform(vgg_features),
        'resnet': scalers['resnet'].fit_transform(resnet_features)
    }
    
    # Poids optimisés (à ajuster selon les performances individuelles)
    weights = {
        'hist': 0.2,     # Les histogrammes sont importants pour la texture et couleur
        'hog': 0.1,      # HOG pour les formes et contours
        'vgg': 0.3,      # VGG16 capture bien les formes et textures
        'resnet': 0.4    # ResNet est généralement le plus performant pour les objets
    }
    
    # Concaténer avec poids
    features_concat = np.hstack([
        normalized_features['hist'] * weights['hist'],
        normalized_features['hog'] * weights['hog'],
        normalized_features['vgg'] * weights['vgg'],
        normalized_features['resnet'] * weights['resnet']
    ])
    
    # Réduction finale pour éliminer la redondance
    pca_final = PCA(n_components=150)
    ensemble_features = pca_final.fit_transform(features_concat)
    
    print(f"Ensemble final - Variance expliquée: {sum(pca_final.explained_variance_ratio_)*100:.2f}%")
    
    return ensemble_features

def compute_hu_moments(images):
    """Calcule les moments de Hu pour capturer la forme de l'image."""
    descriptors = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        descriptors.append(hu_moments)
    return descriptors
    
