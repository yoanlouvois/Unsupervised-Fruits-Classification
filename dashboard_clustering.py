import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import KMeans
from constant import PATH_OUTPUT
from pipeline import *


# -------------------------
# 1. Chargement optimisé des données
# -------------------------
@st.cache_data
def load_data(file_path):
    """
    Chargement des fichiers Excel avec mise en cache pour accélérer les rechargements.
    """
    return pd.read_excel(file_path)

@st.cache_data(show_spinner=False)
def load_images_lazy(images_path):
    """
    Charge toutes les images depuis un dossier donné, mais diffère l'exécution.

    Args:
        images_path: Chemin racine contenant les images.

    Returns:
        train_images: Liste des images chargées sous forme de tableau Numpy.
        train_paths: Liste des chemins absolus de ces images.
    """
    return load_images_from_folder2(images_path)

# Fichiers pour Kmeans
df_hist_kmeans = load_data(PATH_OUTPUT + "/save_clustering_hist_kmeans.xlsx")
df_hog_kmeans = load_data(PATH_OUTPUT + "/save_clustering_hog_kmeans.xlsx")
df_vgg16_kmeans = load_data(PATH_OUTPUT + "/save_clustering_vgg16_kmeans.xlsx")
df_resnet_kmeans = load_data(PATH_OUTPUT + "/save_clustering_resnet_kmeans.xlsx")
df_ensemble_kmeans = load_data(PATH_OUTPUT + "/save_clustering_ensemble_kmeans.xlsx")
df_momenthu_kmeans = load_data(PATH_OUTPUT + "/save_clustering_momenthu_kmeans.xlsx")

# Fichiers pour Spectral Clustering 
df_hist_spectral = load_data(PATH_OUTPUT + "/save_clustering_hist_spectral.xlsx")
df_hog_spectral = load_data(PATH_OUTPUT + "/save_clustering_hog_spectral.xlsx")
df_vgg16_spectral = load_data(PATH_OUTPUT + "/save_clustering_vgg16_spectral.xlsx")
df_resnet_spectral = load_data(PATH_OUTPUT + "/save_clustering_resnet_spectral.xlsx")
df_ensemble_spectral = load_data(PATH_OUTPUT + "/save_clustering_ensemble_spectral.xlsx")
df_momenthu_spectral = load_data(PATH_OUTPUT + "/save_clustering_momenthu_spectral.xlsx")

# Fichier pour les métriques
df_metric = load_data(PATH_OUTPUT + "/save_metric.xlsx")

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns='Unnamed: 0', inplace=True)

# -------------------------
# 2. Pré-traitement des données (dim reduction, clustering, etc.)
# -------------------------
@st.cache_data
def preprocess_cluster_data(df, n_components=3):
    """
    Si nécessaire, utilise TSNE pour réduire les dimensions des données en 3D.
    """
    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        tsne = TSNE(n_components=n_components, random_state=42)
        tsne_result = tsne.fit_transform(df.iloc[:, :-1])  # Suppose qu'on réduit toutes les colonnes sauf 'cluster'
        df['x'], df['y'], df['z'] = tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2]
    return df

def display_images_by_cluster_streamlit(
    df_clustered, cluster_label, image_paths, images_array, images_per_row=5
):
    """
    Affiche les images correspondant au cluster sélectionné dans un layout de grille.

    Args:
        df_clustered : DataFrame contenant les données des clusters.
        cluster_label : Label du cluster sélectionné.
        image_paths : Liste des chemins des images.
        images_array : Liste (ou NumPy array) des images chargées.
        images_per_row : Nombre d'images affichées par ligne dans le layout (par défaut: 5).
    """
    import math

    # Filtrer les images appartenant au cluster sélectionné
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_label]

    # Récupérer les index des images appartenant au cluster
    cluster_indices = cluster_data.index.tolist()

    if len(cluster_indices) == 0:
        st.warning("Aucune image trouvée pour ce cluster.")
        return

    # Calculer le nombre total d'images et le nombre de lignes nécessaires
    num_images = len(cluster_indices)
    num_rows = math.ceil(num_images / images_per_row)

    st.write(f"Total d'images pour le cluster {cluster_label} : {num_images}")

    # Affichage des images dans une grille
    for row_idx in range(num_rows):
        cols = st.columns(images_per_row)
        for col_idx, image_idx in enumerate(range(row_idx * images_per_row, (row_idx + 1) * images_per_row)):
            if image_idx < num_images:
                # Récupérer l'image et son chemin
                img_array = images_array[cluster_indices[image_idx]]
                img_path = image_paths[cluster_indices[image_idx]]

                # Afficher l'image et son nom
                with cols[col_idx]:
                    st.image(img_array, use_column_width=True, caption=f"{img_path}")

# -------------------------
# 3. Génération de graphiques Plotly
# -------------------------
def generate_3d_scatter(df, x_col, y_col, z_col, cluster_col, selected_cluster=None):
    """
    Génère un scatter 3D Plotly avec mise en évidence d'un cluster si nécessaire.
    """
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=cluster_col,
        title="Visualisation 3D du Clustering",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    if selected_cluster is not None:
        filtered_data = df[df[cluster_col] == selected_cluster]
        fig.add_scatter3d(
            x=filtered_data['x'],
            y=filtered_data['y'],
            z=filtered_data['z'],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f'Cluster {selected_cluster}'
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        )
    )

    return fig

# -------------------------
# 4. Interface Utilisateur (Streamlit)
# -------------------------

# Création de deux onglets pour l'analyse
tab1, tab2 = st.tabs(["Analyse par modèle/descripteur", "Analyse globale"])

# Onglet 1 : Analyse par modèle/descripteur
with tab1:
    st.write('## Résultat de Clustering des images')
    st.sidebar.write("#### Veuillez sélectionner les paramètres et le cluster à analyser")

    # --- Sélection des descripteurs et modèles ---
    descriptor = st.sidebar.selectbox(
        'Sélectionner un descripteur',
        ["HISTOGRAM", "HOG", "VGG16", "RESNET", "ENSEMBLE", "MOMENT HU"]
    )
    model = st.sidebar.selectbox('Sélectionner un modèle', ["KMEANS", "SPECTRAL_CLUSTERING"])

    # --- Sélection du DataFrame approprié ---
    if descriptor == "HISTOGRAM" and model == "KMEANS":
        df = df_hist_kmeans
    elif descriptor == "HOG" and model == "KMEANS":
        df = df_hog_kmeans
    elif descriptor == "VGG16" and model == "KMEANS":
        df = df_vgg16_kmeans
    elif descriptor == "RESNET" and model == "KMEANS":
        df = df_resnet_kmeans
    elif descriptor == "ENSEMBLE" and model == "KMEANS":
        df = df_ensemble_kmeans
    elif descriptor == "MOMENT HU" and model == "KMEANS":
        df = df_momenthu_kmeans
        
    # Cas pour les fichiers Spectral Clustering
    elif descriptor == "HISTOGRAM" and model == "SPECTRAL_CLUSTERING":
        df = df_hist_spectral
    elif descriptor == "HOG" and model == "SPECTRAL_CLUSTERING":
        df = df_hog_spectral
    elif descriptor == "VGG16" and model == "SPECTRAL_CLUSTERING":
        df = df_vgg16_spectral
    elif descriptor == "RESNET" and model == "SPECTRAL_CLUSTERING":
        df = df_resnet_spectral
    elif descriptor == "ENSEMBLE" and model == "SPECTRAL_CLUSTERING":
        df = df_ensemble_spectral
    elif descriptor == "MOMENT HU" and model == "SPECTRAL_CLUSTERING":
        df = df_momenthu_spectral
    else:
        st.warning(f"Modèle {model} pour le descripteur {descriptor} n'est pas disponible.")
        df = pd.DataFrame()

    # --- Vérification des données ---
    if not df.empty:
        # Pré-traitement des données pour la visualisation et clustering
        df_processed = preprocess_cluster_data(df)

        # --- Sélection d’un Cluster ---
        cluster_options = df_processed['cluster'].unique()
        selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', cluster_options)

        # --- Affichage des graphiques ---
        st.write(f"### Analyse pour le Descripteur : {descriptor}")
        st.write(f"#### Visualisation 3D du clustering avec descripteur {descriptor}")

        # Génération du graphique 3D
        with st.spinner("Chargement de la visualisation 3D..."):
            fig = generate_3d_scatter(
                df_processed,
                x_col='x',
                y_col='y',
                z_col='z',
                cluster_col='cluster',
                selected_cluster=selected_cluster
            )
            st.plotly_chart(fig)

        # --- Option pour afficher les images ---
        st.write(f"### Visualisation des images pour le cluster : {selected_cluster}")
        images_path = "sujet_tp/src/input/val"

        # Bouton pour afficher les images du cluster
        if st.button(f"Afficher les images du Cluster {selected_cluster}"):
            with st.spinner("Chargement des images..."):
                # Charger les images via la fonction `load_images_lazy`
                train_images, train_paths = load_images_lazy(images_path)

                # Afficher les images pour le cluster sélectionné
                display_images_by_cluster_streamlit(
                    df_clustered=df_processed,
                    cluster_label=selected_cluster,
                    image_paths=train_paths,
                    images_array=train_images,
                    images_per_row=5
                )
    else:
        # Si le DataFrame est vide
        st.error("Aucune donnée disponible pour ce modèle et descripteur.")

# Onglet 2 : Analyse globale
with tab2:
    st.write("## Analyse Globale des métriques du clustering")

    # Visualisation des métriques si elles existent
    if not df_metric.empty:
        st.write("### Aperçu des métriques globales")
        st.dataframe(df_metric)

        # Sélection de colonnes numériques pour les métriques
        numerical_cols = df_metric.select_dtypes(include=[np.number]).columns.tolist()

        if numerical_cols:
            # L'utilisateur sélectionne la métrique à visualiser
            metric_col = st.selectbox("Sélectionner une métrique", numerical_cols)

            # Génération du graphique avec Plotly
            fig = px.bar(
                df_metric,
                x="descriptor",                  # Nom d'un descripteur
                y=metric_col,                    # Colonne numérique sélectionnée
                title=f"Évaluation des métriques : {metric_col}",
                color="name_model",              # Remplace 'model' par 'name_model'
                barmode="group"
            )

            # Affichage dans Streamlit
            st.plotly_chart(fig)
    else:
        st.error("Aucune métrique disponible pour l'analyse.")