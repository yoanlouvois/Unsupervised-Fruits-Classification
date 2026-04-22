from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Initialise un objet KMeans.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - max_iter (int): Le nombre maximum d'itérations pour l'algorithme (par défaut 300).
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        """
        Initialise les centres de clusters avec n_clusters points choisis aléatoirement à partir des données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        hasard = np.random.RandomState(self.random_state)
        maxi = X.shape[0]

        random_indices = hasard.choice(maxi, self.n_clusters, replace=False)

        self.cluster_centers_ = X[random_indices].copy()

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        puis retourne l'indice du cluster le plus proche pour chaque point.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster le plus proche pour chaque point.
        """
        indices = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            distances = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                distances[i] = np.sum((X[j] - self.cluster_centers_[i]) ** 2)
            
            indices[j] = np.argmin(distances)
        return indices

    def fit(self, X):
        """
        Exécute l'algorithme K-means sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        self.initialize_centers(X)
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        
        for iteration in range(self.max_iter):
            new_labels = self.nearest_cluster(X)
            
            if np.array_equal(new_labels, self.labels_):
                break
            self.labels_ = new_labels
            
            for i in range(self.n_clusters):
                if np.sum(self.labels_ == i) > 0:
                    self.cluster_centers_[i] = np.mean(X[self.labels_ == i], axis=0)
        return self

    def predict(self, X):
        """
        Prédit l'appartenance aux clusters pour les données X en utilisant les centres de clusters appris pendant l'entraînement.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster prédit pour chaque point.
        """
        return self.nearest_cluster(X)


def spectral_clustering(descriptors, n_clusters=20):
    """
    Applique le Spectral Clustering sur des données de descripteurs.

    Args:
        descriptors: Matrice des descripteurs (features).
        n_clusters: Nombre de clusters à générer.

    Returns:
        labels: Les labels des clusters pour chaque échantillon.
    """
    # Initialisation du modèle Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    # Ajustement du clustering et récupération des labels
    labels = spectral.fit_predict(descriptors)
    return labels

    

def show_metric(labels_pred, descriptors, bool_return=False, name_descriptor="", name_model="kmeans", bool_show=True):
    """
    Calcule et affiche les métriques clés pour évaluer un clustering.
    
    Args:
        labels_pred: Étiquettes des clusters
        descriptors: Descripteurs utilisés pour le clustering
        bool_return: Retourner les métriques (True/False)
        name_descriptor: Nom du descripteur utilisé
        name_model: Nom du modèle de clustering
        bool_show: Afficher les métriques (True/False)
    
    Returns:
        Un dictionnaire des métriques si bool_return=True
    """
    results = {}
    
    # Conversion des labels en entiers si nécessaire
    labels = np.round(labels_pred).astype(int) if np.asarray(labels_pred).dtype.kind == 'f' else labels_pred
    
    # Calcul des métriques principales avec gestion des exceptions
    try:
        results["silhouette"] = metrics.silhouette_score(descriptors, labels)
        results["davies_bouldin"] = metrics.davies_bouldin_score(descriptors, labels)
        results["calinski_harabasz"] = metrics.calinski_harabasz_score(descriptors, labels)
    except Exception as e:
        if bool_show:
            print(f"Erreur dans le calcul des métriques: {e}")
        results["silhouette"] = results["davies_bouldin"] = results["calinski_harabasz"] = float('nan')
    
    # Statistiques de base sur les clusters
    cluster_counts = np.bincount(labels)
    min_cluster_size = cluster_counts.min() if len(cluster_counts) > 0 else 0
    max_cluster_size = cluster_counts.max() if len(cluster_counts) > 0 else 0
    
    # Ajout du rapport Min/Max (gestion des divisions par zéro)
    min_max_ratio = min_cluster_size / max_cluster_size if max_cluster_size > 0 else float('nan')
    
    results.update({
        "num_clusters": len(cluster_counts),
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "min_max_ratio": min_max_ratio,  # Ajout du rapport Min/Max
        "descriptor": name_descriptor,
        "name_model": name_model
    })
    
    # Affichage si demandé
    if bool_show:
        print(f"\n--- Métriques pour {name_descriptor} ({name_model}) ---")
        print(f"Silhouette: {results['silhouette']:.4f} | Davies-Bouldin: {results['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz: {results['calinski_harabasz']:.2f}")
        print(f"Clusters: {results['num_clusters']} | Taille min/max: {results['min_cluster_size']}/{results['max_cluster_size']}")
        print(f"Rapport Min/Max: {results['min_max_ratio']:.4f}")
    
    return results if bool_return else None
