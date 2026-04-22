# Projet de Clustering d’Images (Fruits)  
**ET4 Informatique — Polytech Paris-Saclay**

## Description

Ce projet a pour objectif de comparer différentes méthodes d’extraction de caractéristiques (**feature engineering**) et de **clustering non supervisé** appliquées à des images de fruits.

L’idée principale est d’évaluer comment le choix des descripteurs influence la qualité du regroupement des images.

---

## Méthodologie

Le projet repose sur deux étapes principales :

### 1. Extraction de caractéristiques (features)

Plusieurs approches ont été testées :

- **Histogrammes de couleurs** : capturent la distribution des couleurs dans l’image  
- **HOG (Histogram of Oriented Gradients)** : capture les contours et textures  
- **Moments de Hu** : décrivent la forme des objets (invariants aux transformations)  
- **VGG16 pré-entraîné** : extraction de features profondes  
- **ResNet50 pré-entraîné** : représentation plus robuste grâce aux connexions résiduelles  

---

### 2. Clustering

Deux méthodes de clustering non supervisé ont été utilisées :

- **K-Means** : partitionnement basé sur la distance  
- **Spectral Clustering** : basé sur la structure du graphe des données  

---

### 3. Visualisation des résultats

Un dashboard interactif a été développé avec **Streamlit** permettant de :

- comparer les résultats selon la méthode de feature utilisée  
- comparer les performances des algorithmes de clustering  
- visualiser les groupes d’images générés  

---

## 4. Lancement 

### step 1 : téléchargement des données et installation des packages
    - a. installer les requierements : "pip install -r requierements.txt"
### step 2 : configuration du chemin vers les donnés
    - dans le dossier src/constant.py, modifier la variable "PATH_DATA" par le chemin vers le dossier contenant les données à clusteriser.

### step 3 :  run de la pipeline clustering
    - aller dans le dossier src
    - exécutez la commande : "python pipeline.py"
    
### step 4 : lancement du dashboard
    - aller dans le dossier src 
    - exécutez la commande : "streamlit run dashboard_clustering.py"
