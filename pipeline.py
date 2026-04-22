import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING


def load_images_from_folder2(folder_path):
    """
    Charge récursivement toutes les images depuis un dossier et ses sous-dossiers.
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les images
        
    Returns:
        tuple: (images, file_paths)
            - images: Liste des images chargées
            - file_paths: Chemins correspondants des fichiers
    """
    images = []
    file_paths = []
    
    # Récupérer tous les sous-dossiers
    subdirs = [d[0] for d in os.walk(folder_path)]
    
    print(f"Chargement des images depuis {len(subdirs)} sous-dossiers...")
    
    # Parcourir chaque sous-dossier
    for subdir in subdirs:
        # Chercher tous les fichiers image dans ce sous-dossier
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for pattern in image_patterns:
            image_files = glob.glob(os.path.join(subdir, pattern))
            
            for image_file in image_files:
                try:
                    # Charger l'image en couleur
                    img = cv2.imread(image_file)
                    if img is not None:
                        # Convertir BGR à RGB (cv2 charge en BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        file_paths.append(image_file)
                except Exception as e:
                    print(f"Erreur lors du chargement de {image_file}: {e}")
    
    print(f"Total de {len(images)} images chargées.")
    return images, file_paths

def load_images_from_folder(folder_path, max_images=300):
    """
    Charge récursivement un nombre limité d'images depuis un dossier et ses sous-dossiers.
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les images.
        max_images (int): Nombre maximal d'images à charger (par défaut : 300).
        
    Returns:
        tuple: (images, file_paths)
            - images: Liste des images chargées.
            - file_paths: Chemins correspondants des fichiers.
    """
    images = []
    file_paths = []
    
    # Récupérer tous les sous-dossiers
    subdirs = [d[0] for d in os.walk(folder_path)]
    
    print(f"Chargement des images depuis {len(subdirs)} sous-dossiers...")
    
    # Parcourir chaque sous-dossier
    for subdir in subdirs:
        # Chercher tous les fichiers image dans ce sous-dossier
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for pattern in image_patterns:
            image_files = glob.glob(os.path.join(subdir, pattern))
            
            for image_file in image_files:
                if len(images) >= max_images:
                    print(f"Limite de {max_images} images atteinte. Arrêt du chargement.")
                    return images, file_paths  # Retourner les images déjà chargées
                
                try:
                    # Charger l'image en couleur
                    img = cv2.imread(image_file)
                    if img is not None:
                        # Convertir BGR à RGB (cv2 charge en BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        file_paths.append(image_file)
                except Exception as e:
                    print(f"Erreur lors du chargement de {image_file}: {e}")
    
    print(f"Nombre total d'images chargées : {len(images)}")
    return images, file_paths

def pipeline():

    print("\n\n ##### Chargement des images ######")

    # Charger les images d'entraînement (pour construire le modèle)
    images_path = "sujet_tp/src/input/val"
    
    # Charger toutes les images depuis tous les sous-dossiers
    train_images, train_paths = load_images_from_folder2(images_path)

    print("\n\n ##### Extraction de Features ######")
    
    # Features HOG
    print("- Calcul features HOG sur images d'entraînement...")
    descriptors_hog = compute_hog_descriptors(train_images)
    print(f"  Forme des descripteurs HOG: {np.array(descriptors_hog).shape}")
    
    # Features Histogramme
    print("- Calcul features Histogram sur images d'entraînement...")
    descriptors_hist = compute_gray_histograms(train_images)
    print(f"  Forme des descripteurs Histogram: {np.array(descriptors_hist).shape}")

    # Features VGG16
    print("- Calcul features VGG16 sur images d'entraînement...")
    features_vgg16 = compute_vgg16_features(train_images)
    print(f"  Forme des descripteurs VGG16: {features_vgg16.shape}")

    # Features ResNet50 + PCA
    print("- Calcul features ResNet50+PCA sur images d'entraînement...")
    features_resnet = compute_resnet50_features(train_images)
    print(f"  Forme des descripteurs ResNet50+PCA: {features_resnet.shape}")

    # Features d'ensemble
    print("- Calcul features Ensemble sur images d'entraînement...")
    features_ensemble = compute_ensemble_features(train_images, descriptors_hist, descriptors_hog, features_vgg16, features_resnet)
    print(f"  Forme des descripteurs Ensemble: {features_ensemble.shape}")

    # Features Moment Hu
    print("- Calcul features Moment Hu sur images d'entrainement...")
    features_momenthu = compute_hu_moments(train_images)
    print(f"  Forme des descripteurs Hu Moment: {np.array(features_momenthu).shape}")

    
    print("\n\n ##### Clustering ######")
    number_cluster = 20

    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(np.array(descriptors_hist))
    descriptors_hog_norm = scaler.fit_transform(np.array(descriptors_hog))
    features_vgg16_norm = scaler.fit_transform(features_vgg16)
    features_resnet_norm = scaler.fit_transform(features_resnet)
    features_ensemble_norm = scaler.fit_transform(features_ensemble)
    features_momenthu_norm = scaler.fit_transform(np.array(features_momenthu))
    
    # Initialisation des modèles (clustering.py)
    kmeans_hog = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_hist = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_vgg16 = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_resnet = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_ensemble = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_momenthu = KMeans(n_clusters=number_cluster, random_state=42)
    
    # Entraînement du modèle Kmeans (clustering.py)
    print("- Clustering avec HOG...")
    kmeans_hog.fit(np.array(descriptors_hog))
    
    print("- Clustering avec Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))
    
    print("- Clustering avec VGG16...")
    kmeans_vgg16.fit(features_vgg16)
    
    print("- Clustering avec ResNet50+PCA...")
    kmeans_resnet.fit(features_resnet)

    print("- Clustering avec Ensemble de features...")
    kmeans_ensemble.fit(features_ensemble)

    print("- Clustering avec le Moment Hu...")
    kmeans_momenthu.fit(np.array(features_momenthu))

    # Entraînement du modèle Spectral Clustering (spectral_clustering dans clustering.py)
    print("- Spectral Clustering avec HOG...")
    labels_hog_spectral = spectral_clustering(np.array(descriptors_hog), n_clusters=number_cluster)
    
    print("- Spectral Clustering avec Histogram...")
    labels_hist_spectral = spectral_clustering(np.array(descriptors_hist), n_clusters=number_cluster)
    
    print("- Spectral Clustering avec VGG16...")
    labels_vgg16_spectral = spectral_clustering(features_vgg16, n_clusters=number_cluster)
    
    print("- Spectral Clustering avec ResNet50+PCA...")
    labels_resnet_spectral = spectral_clustering(features_resnet, n_clusters=number_cluster)
    
    print("- Spectral Clustering avec Ensemble de features...")
    labels_ensemble_spectral = spectral_clustering(features_ensemble, n_clusters=number_cluster)
    
    print("- Spectral Clustering avec le Moment Hu...")
    labels_momenthu_spectral = spectral_clustering(np.array(features_momenthu), n_clusters=number_cluster)

    
    # Calcul des métriques pour l'évaluation non supervisée
    print("\n\n ##### Évaluation ######")
    print("Évaluation sur les images d'entraînement:")
    metric_hist = show_metric(labels_pred=kmeans_hist.labels_, descriptors=np.array(descriptors_hist), 
                                    bool_show=True, name_descriptor="HISTOGRAM", name_model="kmeans", bool_return=True)
    
    metric_hog = show_metric(labels_pred=kmeans_hog.labels_, descriptors=np.array(descriptors_hog), 
                                  bool_show=True, name_descriptor="HOG", name_model="kmeans", bool_return=True)
    
    metric_vgg16 = show_metric(labels_pred=kmeans_vgg16.labels_, descriptors=features_vgg16, 
                                     bool_show=True, name_descriptor="VGG16", name_model="kmeans", bool_return=True)
    
    metric_resnet = show_metric(labels_pred=kmeans_resnet.labels_, descriptors=features_resnet, 
                                      bool_show=True, name_descriptor="RESNET50+PCA", name_model="kmeans", bool_return=True)

    metric_ensemble = show_metric(labels_pred=kmeans_ensemble.labels_, descriptors=features_ensemble, 
                                      bool_show=True, name_descriptor="ENSEMBLE", name_model="kmeans", bool_return=True)

    metric_momenthu = show_metric(labels_pred=kmeans_momenthu.labels_, descriptors=np.array(features_momenthu), 
                                      bool_show=True, name_descriptor="MOMENTHU", name_model="kmeans", bool_return=True)

    #Calcul des métriques pour Spectral Clustering
    print("\nÉvaluation des modèles Spectral Clustering:")
    metric_hist_spectral = show_metric(labels_pred=labels_hist_spectral, descriptors=np.array(descriptors_hist), 
                                        bool_show=True, name_descriptor="HISTOGRAM", name_model="spectral", bool_return=True)
    
    metric_hog_spectral = show_metric(labels_pred=labels_hog_spectral, descriptors=np.array(descriptors_hog), 
                                       bool_show=True, name_descriptor="HOG", name_model="spectral", bool_return=True)
    
    metric_vgg16_spectral = show_metric(labels_pred=labels_vgg16_spectral, descriptors=features_vgg16, 
                                         bool_show=True, name_descriptor="VGG16", name_model="spectral", bool_return=True)
    
    metric_resnet_spectral = show_metric(labels_pred=labels_resnet_spectral, descriptors=features_resnet, 
                                          bool_show=True, name_descriptor="RESNET50+PCA", name_model="spectral", bool_return=True)
    
    metric_ensemble_spectral = show_metric(labels_pred=labels_ensemble_spectral, descriptors=features_ensemble, 
                                           bool_show=True, name_descriptor="ENSEMBLE", name_model="spectral", bool_return=True)
    
    metric_momenthu_spectral = show_metric(labels_pred=labels_momenthu_spectral, descriptors=np.array(features_momenthu), 
                                           bool_show=True, name_descriptor="MOMENTHU", name_model="spectral", bool_return=True)
    
    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist, metric_hog, metric_vgg16, metric_resnet, metric_ensemble, metric_momenthu,
             metric_hist_spectral, metric_hog_spectral, metric_vgg16_spectral, metric_resnet_spectral, 
             metric_ensemble_spectral, metric_momenthu_spectral]

    df_metric = pd.DataFrame(list_dict)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_vgg16 = conversion_3d(features_vgg16_norm)
    x_3d_resnet = conversion_3d(features_resnet_norm)
    x_3d_ensemble = conversion_3d(features_ensemble_norm)
    x_3d_momenthu = conversion_3d(features_momenthu_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, kmeans_hist.labels_, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, kmeans_hog.labels_, kmeans_hog.labels_)
    df_vgg16 = create_df_to_export(x_3d_vgg16, kmeans_vgg16.labels_, kmeans_vgg16.labels_)
    df_resnet = create_df_to_export(x_3d_resnet, kmeans_resnet.labels_, kmeans_resnet.labels_)
    df_ensemble = create_df_to_export(x_3d_ensemble, kmeans_ensemble.labels_, kmeans_ensemble.labels_)
    df_momenthu = create_df_to_export(x_3d_momenthu, kmeans_momenthu.labels_, kmeans_momenthu.labels_)

    # Création des DataFrames pour Spectral Clustering uniquement (labels Spectral Clustering)
    df_hist_spectral = create_df_to_export(x_3d_hist, labels_hist_spectral, labels_hist_spectral)
    df_hog_spectral = create_df_to_export(x_3d_hog, labels_hog_spectral, labels_hog_spectral)
    df_vgg16_spectral = create_df_to_export(x_3d_vgg16, labels_vgg16_spectral, labels_vgg16_spectral)
    df_resnet_spectral = create_df_to_export(x_3d_resnet, labels_resnet_spectral, labels_resnet_spectral)
    df_ensemble_spectral = create_df_to_export(x_3d_ensemble, labels_ensemble_spectral, labels_ensemble_spectral)
    df_momenthu_spectral = create_df_to_export(x_3d_momenthu, labels_momenthu_spectral, labels_momenthu_spectral)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # Sauvegarde des données pour le dashboard
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_vgg16.to_excel(PATH_OUTPUT+"/save_clustering_vgg16_kmeans.xlsx")
    df_resnet.to_excel(PATH_OUTPUT+"/save_clustering_resnet_kmeans.xlsx")
    df_ensemble.to_excel(PATH_OUTPUT+"/save_clustering_ensemble_kmeans.xlsx")
    df_momenthu.to_excel(PATH_OUTPUT+"/save_clustering_momenthu_kmeans.xlsx")

    # Sauvegarde des données pour Spectral Clustering (nouveaux fichiers)
    df_hist_spectral.to_excel(PATH_OUTPUT+"/save_clustering_hist_spectral.xlsx")
    df_hog_spectral.to_excel(PATH_OUTPUT+"/save_clustering_hog_spectral.xlsx")
    df_vgg16_spectral.to_excel(PATH_OUTPUT+"/save_clustering_vgg16_spectral.xlsx")
    df_resnet_spectral.to_excel(PATH_OUTPUT+"/save_clustering_resnet_spectral.xlsx")
    df_ensemble_spectral.to_excel(PATH_OUTPUT+"/save_clustering_ensemble_spectral.xlsx")
    df_momenthu_spectral.to_excel(PATH_OUTPUT+"/save_clustering_momenthu_spectral.xlsx")

     # Sauvegarde des métriques
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")

    print("Calcul du silouhette score pour différents nombres de clusters : ")
    # Kmeans pour différent nombre de Clusters
    cluster_range = [5, 10, 15, 20, 25]
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
    
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


    


if __name__ == "__main__":
    pipeline()