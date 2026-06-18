# ml_engine/ml_core.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.datasets import make_classification
from xgboost import XGBRegressor, XGBClassifier
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# ==================== UNSUPERVISED LEARNING IMPORTS ====================
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from kneed import KneeLocator
# =====================================================================

class MLEngine:
    def __init__(self):
        self.df = None
        self.target_col = None
        self.focus_col = None  # Column user selects for unsupervised analysis
        self.problem_type = None
        self.models_results = {}
        self.best_model = None
        self.best_score = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.X_test = None
        self.y_test = None
        self.best_model_obj = None
        self.feature_names = []
        self.unsupervised_recommendations = {}
        self.optimal_k = None
        self.has_outliers = False
        self.outlier_count = 0

    def load_data(self, filepath):
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            self.df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            self.df = pd.read_json(filepath)
        return self.df

    def detect_problem_type(self, target_column=None):
        """
        Auto-detect if dataset is supervised or unsupervised.
        If target_column provided -> supervised (classification/regression)
        If target_column is None -> unsupervised
        """
        if target_column is None or target_column == '':
            self.problem_type = 'unsupervised'
            self.target_col = None
            return 'unsupervised'

        self.target_col = target_column
        if pd.api.types.is_numeric_dtype(self.df[target_column]):
            unique_count = self.df[target_column].nunique()
            self.problem_type = 'regression' if unique_count > 20 else 'classification'
        else:
            self.problem_type = 'classification'
        return self.problem_type

    def get_numeric_columns(self):
        """Get list of numeric columns for user to select from (unsupervised mode)"""
        return list(self.df.select_dtypes(include=[np.number]).columns)

    def analyze_dataset_for_unsupervised(self, focus_column=None):
        """
        Analyze dataset characteristics and auto-determine:
        - Optimal k (clusters) via Elbow + Silhouette
        - Outlier presence and count
        - Best algorithm recommendations

        Parameters:
        -----------
        focus_column : str or None
            Column user wants to focus on. If None, uses all numeric columns.

        Returns:
        --------
        dict : Analysis results with recommendations
        """
        self.focus_col = focus_column

        # Prepare data
        if focus_column and focus_column in self.df.columns:
            X = self.df[[focus_column]].copy()
        else:
            X = self.df.select_dtypes(include=[np.number]).copy()

        if X.empty:
            raise ValueError("No numeric columns found for unsupervised analysis.")

        X_clean = X.dropna()
        X_scaled = self.scaler.fit_transform(X_clean)

        n_samples = len(X_scaled)
        n_features = X_scaled.shape[1]

        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'focus_column': focus_column,
            'dataset_characteristics': {}
        }

        # ========== AUTO-DETERMINE OPTIMAL K ==========
        max_k = min(10, n_samples - 1)
        if max_k < 2:
            analysis['optimal_k'] = 2
            analysis['k_method'] = 'default (too few samples)'
        else:
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_k + 1)

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                inertias.append(km.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, labels))

            # Elbow method
            try:
                kn = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
                elbow_k = kn.elbow if kn.elbow else 3
            except:
                elbow_k = 3

            # Silhouette method
            best_sil_k = list(k_range)[np.argmax(silhouette_scores)]

            # Combine: prefer silhouette, fallback to elbow
            self.optimal_k = best_sil_k if silhouette_scores[np.argmax(silhouette_scores)] > 0.3 else elbow_k

            analysis['optimal_k'] = self.optimal_k
            analysis['elbow_k'] = elbow_k
            analysis['silhouette_k'] = best_sil_k
            analysis['k_method'] = 'silhouette' if self.optimal_k == best_sil_k else 'elbow'
            analysis['silhouette_scores'] = {k: round(v, 4) for k, v in zip(k_range, silhouette_scores)}
            analysis['inertias'] = {k: round(v, 4) for k, v in zip(k_range, inertias)}

        # ========== OUTLIER DETECTION ==========
        try:
            iso_forest = IsolationForest(contamination='auto', random_state=42)
            outlier_labels = iso_forest.fit_predict(X_scaled)
            self.outlier_count = list(outlier_labels).count(-1)
            self.has_outliers = self.outlier_count > 0
            outlier_ratio = self.outlier_count / n_samples

            analysis['outlier_analysis'] = {
                'has_outliers': self.has_outliers,
                'outlier_count': self.outlier_count,
                'outlier_ratio': round(outlier_ratio, 4),
                'contamination': round(outlier_ratio, 4) if outlier_ratio > 0 else 0.1
            }
        except Exception as e:
            analysis['outlier_analysis'] = {'error': str(e)}

        # ========== DATASET CHARACTERISTICS ==========
        analysis['dataset_characteristics'] = {
            'size_category': 'small' if n_samples < 500 else 'medium' if n_samples < 5000 else 'large',
            'dimensionality': 'low' if n_features <= 5 else 'medium' if n_features <= 20 else 'high',
            'sparsity': round((self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])), 4),
            'has_outliers': self.has_outliers
        }

        # ========== ALGORITHM RECOMMENDATIONS ==========
        recommendations = []

        # K-Means (always recommend as baseline)
        kmeans_score = 'highly_recommended'
        if n_samples < 100:
            kmeans_score = 'recommended'
        recommendations.append({
            'algorithm': 'K-Means',
            'recommendation': kmeans_score,
            'reason': f"Fast and scalable. Optimal k={self.optimal_k} detected.",
            'suitable_for': ['general clustering', 'large datasets', 'spherical clusters'],
            'optimal_k': self.optimal_k
        })

        # Hierarchical Clustering
        if n_samples < 2000:
            hierarchical_score = 'highly_recommended'
        elif n_samples < 10000:
            hierarchical_score = 'recommended'
        else:
            hierarchical_score = 'not_recommended'

        recommendations.append({
            'algorithm': 'Hierarchical',
            'recommendation': hierarchical_score,
            'reason': 'Good for small-medium datasets. Shows cluster hierarchy via dendrogram.',
            'suitable_for': ['small datasets', 'hierarchical relationships', 'interpretable clusters'],
            'optimal_k': self.optimal_k
        })

        # DBSCAN
        if self.has_outliers and n_samples > 200:
            dbscan_score = 'highly_recommended'
        elif n_samples > 500:
            dbscan_score = 'recommended'
        else:
            dbscan_score = 'optional'

        recommendations.append({
            'algorithm': 'DBSCAN',
            'recommendation': dbscan_score,
            'reason': f"Great for finding arbitrary cluster shapes and noise points. {'Outliers detected!' if self.has_outliers else 'No significant outliers.'}",
            'suitable_for': ['arbitrary shapes', 'noise detection', 'density-based clusters'],
            'optimal_k': 'auto-detected'
        })

        # PCA
        if n_features > 5:
            pca_score = 'highly_recommended'
        else:
            pca_score = 'optional'

        recommendations.append({
            'algorithm': 'PCA',
            'recommendation': pca_score,
            'reason': f"Dimensionality reduction. {n_features} features detected — can reduce while preserving variance.",
            'suitable_for': ['high dimensional data', 'feature reduction', 'visualization'],
            'optimal_k': None
        })

        # t-SNE
        if n_samples < 5000:
            tsne_score = 'recommended'
        else:
            tsne_score = 'not_recommended'

        recommendations.append({
            'algorithm': 't-SNE',
            'recommendation': tsne_score,
            'reason': 'Excellent for 2D/3D visualization of high-dimensional clusters.',
            'suitable_for': ['visualization', 'non-linear structures', 'exploratory analysis'],
            'optimal_k': None
        })

        # Isolation Forest (Anomaly Detection)
        if self.has_outliers:
            iso_score = 'highly_recommended'
        else:
            iso_score = 'optional'

        recommendations.append({
            'algorithm': 'Isolation Forest',
            'recommendation': iso_score,
            'reason': f"Anomaly detection. {self.outlier_count} outliers detected ({round(self.outlier_count/n_samples*100, 2)}% of data).",
            'suitable_for': ['anomaly detection', 'fraud detection', 'noise filtering'],
            'optimal_k': None
        })

        # Local Outlier Factor
        if n_samples < 5000 and self.has_outliers:
            lof_score = 'recommended'
        else:
            lof_score = 'optional'

        recommendations.append({
            'algorithm': 'Local Outlier Factor',
            'recommendation': lof_score,
            'reason': 'Density-based anomaly detection. Good for local outliers.',
            'suitable_for': ['local outliers', 'density-based anomalies', 'small datasets'],
            'optimal_k': None
        })

        analysis['recommendations'] = recommendations
        self.unsupervised_recommendations = recommendations

        return analysis

    def train_unsupervised(self, algorithm_name, focus_column=None, n_clusters=None, random_state=42):
        """
        Train a specific unsupervised algorithm selected by the user.

        Parameters:
        -----------
        algorithm_name : str
            One of: 'K-Means', 'DBSCAN', 'Hierarchical', 'PCA', 't-SNE', 
                    'Isolation Forest', 'Local Outlier Factor'
        focus_column : str or None
            Column to focus on (from user selection)
        n_clusters : int or None
            Number of clusters. If None, uses auto-detected optimal_k.
        random_state : int
            Random seed
        """
        self.problem_type = 'unsupervised'
        self.focus_col = focus_column

        # Use auto-detected k if not provided
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3

        # Prepare data
        if focus_column and focus_column in self.df.columns:
            X = self.df[[focus_column]].copy()
        else:
            X = self.df.select_dtypes(include=[np.number]).copy()

        X = X.dropna()
        X_scaled = self.scaler.fit_transform(X)

        n_samples = len(X_scaled)
        results = {}

        # ---------- K-MEANS ----------
        if algorithm_name == 'K-Means':
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                results = {
                    'algorithm': 'K-Means',
                    'n_clusters': n_clusters,
                    'silhouette_score': round(silhouette_score(X_scaled, labels), 4),
                    'calinski_harabasz_score': round(calinski_harabasz_score(X_scaled, labels), 4),
                    'davies_bouldin_score': round(davies_bouldin_score(X_scaled, labels), 4),
                    'inertia': round(kmeans.inertia_, 4),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_counts': pd.Series(labels).value_counts().to_dict(),
                    'labels': labels.tolist(),
                    'convergence': kmeans.n_iter_
                }
                self.best_model_obj = kmeans
            except Exception as e:
                results = {'error': str(e)}

        # ---------- DBSCAN ----------
        elif algorithm_name == 'DBSCAN':
            try:
                # Auto-tune eps based on k-distance graph
                from sklearn.neighbors import NearestNeighbors
                neigh = NearestNeighbors(n_neighbors=min(5, n_samples-1))
                neigh.fit(X_scaled)
                distances, _ = neigh.kneighbors(X_scaled)
                distances = np.sort(distances[:, -1])

                # Simple heuristic: eps at the "elbow" of k-distance
                eps = np.percentile(distances, 90) if len(distances) > 0 else 0.5

                dbscan = DBSCAN(eps=eps, min_samples=min(5, n_samples//10 + 1))
                labels = dbscan.fit_predict(X_scaled)
                n_clusters_dbscan = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                results = {
                    'algorithm': 'DBSCAN',
                    'eps': round(eps, 4),
                    'min_samples': dbscan.min_samples,
                    'n_clusters': n_clusters_dbscan,
                    'n_noise_points': n_noise,
                    'noise_ratio': round(n_noise / n_samples, 4) if n_samples > 0 else 0,
                    'cluster_counts': pd.Series(labels).value_counts().to_dict(),
                    'labels': labels.tolist()
                }

                if n_clusters_dbscan > 1:
                    mask = labels != -1
                    if mask.sum() > n_clusters_dbscan:
                        results['silhouette_score'] = round(silhouette_score(X_scaled[mask], labels[mask]), 4)

                self.best_model_obj = dbscan
            except Exception as e:
                results = {'error': str(e)}

        # ---------- HIERARCHICAL ----------
        elif algorithm_name == 'Hierarchical':
            try:
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                labels = agg.fit_predict(X_scaled)

                results = {
                    'algorithm': 'Hierarchical',
                    'n_clusters': n_clusters,
                    'silhouette_score': round(silhouette_score(X_scaled, labels), 4),
                    'calinski_harabasz_score': round(calinski_harabasz_score(X_scaled, labels), 4),
                    'davies_bouldin_score': round(davies_bouldin_score(X_scaled, labels), 4),
                    'linkage': 'ward',
                    'cluster_counts': pd.Series(labels).value_counts().to_dict(),
                    'labels': labels.tolist()
                }
                self.best_model_obj = agg
            except Exception as e:
                results = {'error': str(e)}

        # ---------- PCA ----------
        elif algorithm_name == 'PCA':
            try:
                n_components = min(n_clusters, X_scaled.shape[1]) if n_clusters else min(2, X_scaled.shape[1])
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)

                explained_variance = pca.explained_variance_ratio_.tolist()
                cumulative_variance = np.cumsum(explained_variance).tolist()

                results = {
                    'algorithm': 'PCA',
                    'n_components': pca.n_components_,
                    'explained_variance_ratio': [round(v, 4) for v in explained_variance],
                    'cumulative_variance': [round(v, 4) for v in cumulative_variance],
                    'total_variance_explained': round(sum(explained_variance), 4),
                    'components': pca.components_.tolist(),
                    'transformed_shape': list(X_pca.shape)
                }
                self.best_model_obj = pca
            except Exception as e:
                results = {'error': str(e)}

        # ---------- t-SNE ----------
        elif algorithm_name == 't-SNE':
            try:
                perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
                tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X_scaled)

                results = {
                    'algorithm': 't-SNE',
                    'n_components': 2,
                    'perplexity': perplexity,
                    'kl_divergence': round(tsne.kl_divergence_, 4) if hasattr(tsne, 'kl_divergence_') else None,
                    'n_iter': tsne.n_iter_ if hasattr(tsne, 'n_iter_') else None,
                    'transformed_shape': list(X_tsne.shape)
                }
                self.best_model_obj = tsne
            except Exception as e:
                results = {'error': str(e)}

        # ---------- ISOLATION FOREST ----------
        elif algorithm_name == 'Isolation Forest':
            try:
                contamination = min(0.5, max(0.01, self.outlier_count / n_samples)) if self.outlier_count > 0 else 0.1
                iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
                labels = iso_forest.fit_predict(X_scaled)
                n_outliers = list(labels).count(-1)
                n_inliers = list(labels).count(1)

                # Anomaly scores
                scores = iso_forest.decision_function(X_scaled)

                results = {
                    'algorithm': 'Isolation Forest',
                    'contamination': contamination,
                    'n_outliers': n_outliers,
                    'n_inliers': n_inliers,
                    'outlier_ratio': round(n_outliers / n_samples, 4),
                    'mean_anomaly_score': round(float(np.mean(scores)), 4),
                    'labels': labels.tolist(),
                    'anomaly_scores': scores.tolist()
                }
                self.best_model_obj = iso_forest
            except Exception as e:
                results = {'error': str(e)}

        # ---------- LOCAL OUTLIER FACTOR ----------
        elif algorithm_name == 'Local Outlier Factor':
            try:
                contamination = min(0.5, max(0.01, self.outlier_count / n_samples)) if self.outlier_count > 0 else 0.1
                lof = LocalOutlierFactor(n_neighbors=min(20, n_samples-1), contamination=contamination)
                labels = lof.fit_predict(X_scaled)
                n_outliers = list(labels).count(-1)
                n_inliers = list(labels).count(1)

                # Negative outlier factor scores
                scores = lof.negative_outlier_factor_

                results = {
                    'algorithm': 'Local Outlier Factor',
                    'n_neighbors': lof.n_neighbors,
                    'contamination': contamination,
                    'n_outliers': n_outliers,
                    'n_inliers': n_inliers,
                    'outlier_ratio': round(n_outliers / n_samples, 4),
                    'mean_lof_score': round(float(np.mean(scores)), 4),
                    'labels': labels.tolist(),
                    'lof_scores': scores.tolist()
                }
                self.best_model_obj = lof
            except Exception as e:
                results = {'error': str(e)}

        else:
            results = {'error': f'Unknown algorithm: {algorithm_name}'}

        self.best_model = algorithm_name
        self.models_results = {algorithm_name: results}

        # Set best_score based on algorithm type
        if 'silhouette_score' in results:
            self.best_score = results['silhouette_score']
        elif 'n_outliers' in results:
            self.best_score = results.get('outlier_ratio', 0)
        elif 'total_variance_explained' in results:
            self.best_score = results['total_variance_explained']
        else:
            self.best_score = None

        return results

    def create_unsupervised_visualizations(self, algorithm_name=None, random_state=42):
        """
        Create visualizations for the selected unsupervised algorithm.
        """
        if algorithm_name is None:
            algorithm_name = self.best_model

        visualizations = {}

        # Prepare data
        if self.focus_col and self.focus_col in self.df.columns:
            X = self.df[[self.focus_col]].copy()
        else:
            X = self.df.select_dtypes(include=[np.number]).copy()

        X = X.dropna()
        X_scaled = self.scaler.fit_transform(X)
        n_samples = len(X_scaled)

        # ========== K-MEANS VISUALIZATIONS ==========
        if algorithm_name == 'K-Means':
            # 1. Elbow Plot
            try:
                inertias = []
                sil_scores = []
                k_range = range(2, min(11, n_samples))
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X_scaled, km.labels_))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Elbow
                ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
                if self.optimal_k:
                    ax1.axvline(x=self.optimal_k, color='red', linestyle='--', label=f'Optimal k={self.optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.set_title('Elbow Method')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Silhouette
                ax2.plot(list(k_range), sil_scores, 'go-', linewidth=2, markersize=8)
                if self.optimal_k:
                    ax2.axvline(x=self.optimal_k, color='red', linestyle='--', label=f'Optimal k={self.optimal_k}')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Analysis')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['elbow_silhouette'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['elbow_silhouette'] = None

            # 2. Cluster Scatter (PCA 2D)
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                kmeans = KMeans(n_clusters=self.optimal_k or 3, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                                  alpha=0.7, edgecolors='k', s=50)
                plt.colorbar(scatter, label='Cluster')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title(f'K-Means Clusters (k={self.optimal_k}) in PCA Space', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['cluster_scatter'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['cluster_scatter'] = None

            # 3. Cluster Distribution Bar Chart
            try:
                labels = kmeans.labels_ if 'kmeans' in locals() else KMeans(
                    n_clusters=self.optimal_k or 3, random_state=random_state, n_init=10
                ).fit_predict(X_scaled)
                counts = pd.Series(labels).value_counts().sort_index()

                plt.figure(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(counts)))
                plt.bar([f'Cluster {c}' for c in counts.index], counts.values, color=colors)
                plt.xlabel('Cluster')
                plt.ylabel('Number of Points')
                plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['cluster_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['cluster_distribution'] = None

        # ========== DBSCAN VISUALIZATIONS ==========
        elif algorithm_name == 'DBSCAN':
            # 1. DBSCAN Cluster Scatter
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                from sklearn.neighbors import NearestNeighbors
                neigh = NearestNeighbors(n_neighbors=min(5, n_samples-1))
                neigh.fit(X_scaled)
                distances, _ = neigh.kneighbors(X_scaled)
                distances = np.sort(distances[:, -1])
                eps = np.percentile(distances, 90) if len(distances) > 0 else 0.5

                dbscan = DBSCAN(eps=eps, min_samples=min(5, n_samples//10 + 1))
                labels = dbscan.fit_predict(X_scaled)

                plt.figure(figsize=(10, 8))
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                for label, color in zip(unique_labels, colors):
                    if label == -1:
                        color = 'red'
                        label_name = 'Noise'
                    else:
                        label_name = f'Cluster {label}'
                    mask = labels == label
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=label_name, 
                               alpha=0.7, edgecolors='k', s=50)

                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title('DBSCAN Clusters (Red = Noise Points)', fontsize=14, fontweight='bold')
                plt.legend()
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['dbscan_clusters'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['dbscan_clusters'] = None

            # 2. k-Distance Graph (for eps tuning)
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(distances, 'b-', linewidth=1)
                plt.axhline(y=eps, color='red', linestyle='--', label=f'Selected eps={eps:.4f}')
                plt.xlabel('Points (sorted by distance)')
                plt.ylabel('k-Distance')
                plt.title('k-Distance Graph for DBSCAN eps Selection', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['k_distance'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['k_distance'] = None

        # ========== HIERARCHICAL VISUALIZATIONS ==========
        elif algorithm_name == 'Hierarchical':
            # 1. Dendrogram
            try:
                if n_samples > 100:
                    idx = np.random.choice(n_samples, 100, replace=False)
                    X_sample = X_scaled[idx]
                else:
                    X_sample = X_scaled

                linked = linkage(X_sample, method='ward')

                plt.figure(figsize=(14, 6))
                dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
                plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
                plt.xlabel('Sample Index')
                plt.ylabel('Distance')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['dendrogram'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['dendrogram'] = None

            # 2. Cluster Scatter
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                agg = AgglomerativeClustering(n_clusters=self.optimal_k or 3)
                labels = agg.fit_predict(X_scaled)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', 
                                  alpha=0.7, edgecolors='k', s=50)
                plt.colorbar(scatter, label='Cluster')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title(f'Hierarchical Clusters (k={self.optimal_k}) in PCA Space', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['hierarchical_scatter'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['hierarchical_scatter'] = None

        # ========== PCA VISUALIZATIONS ==========
        elif algorithm_name == 'PCA':
            # 1. Explained Variance Plot
            try:
                pca_full = PCA()
                pca_full.fit(X_scaled)
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)

                plt.figure(figsize=(10, 6))
                plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                       pca_full.explained_variance_ratio_, alpha=0.7, label='Individual', color='steelblue')
                plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', label='Cumulative', linewidth=2)
                plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
                plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Threshold')
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance Ratio')
                plt.title('PCA Explained Variance', fontsize=14, fontweight='bold')
                plt.legend()
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['pca_variance'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['pca_variance'] = None

            # 2. PCA 2D Projection
            try:
                pca_2d = PCA(n_components=2)
                X_pca = pca_2d.fit_transform(X_scaled)

                plt.figure(figsize=(10, 8))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k', s=50, c='steelblue')
                plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
                plt.title('PCA 2D Projection', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['pca_2d'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['pca_2d'] = None

            # 3. Feature Loadings (Biplot)
            try:
                if len(self.feature_names) >= 2:
                    pca_2d = PCA(n_components=2)
                    pca_2d.fit(X_scaled)

                    plt.figure(figsize=(10, 8))
                    scale = 1.5
                    for i, feature in enumerate(self.feature_names[:X_scaled.shape[1]]):
                        plt.arrow(0, 0, pca_2d.components_[0, i] * scale, pca_2d.components_[1, i] * scale,
                                 head_width=0.05, head_length=0.05, fc='red', ec='red')
                        plt.text(pca_2d.components_[0, i] * scale * 1.1, 
                                pca_2d.components_[1, i] * scale * 1.1, feature, 
                                color='red', ha='center', va='center')

                    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
                    plt.title('PCA Feature Loadings (Biplot)', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    buf.seek(0)
                    visualizations['pca_biplot'] = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
            except:
                visualizations['pca_biplot'] = None

        # ========== t-SNE VISUALIZATIONS ==========
        elif algorithm_name == 't-SNE':
            try:
                perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
                tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X_scaled)

                # Also run K-Means to color points
                kmeans = KMeans(n_clusters=self.optimal_k or 3, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', 
                                  alpha=0.7, edgecolors='k', s=50)
                plt.colorbar(scatter, label='Cluster')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title(f't-SNE Visualization (perplexity={perplexity})', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['tsne_scatter'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['tsne_scatter'] = None

        # ========== ANOMALY DETECTION VISUALIZATIONS ==========
        elif algorithm_name in ['Isolation Forest', 'Local Outlier Factor']:
            # 1. Anomaly Scatter Plot
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                if algorithm_name == 'Isolation Forest':
                    model = IsolationForest(contamination=0.1, random_state=random_state)
                    labels = model.fit_predict(X_scaled)
                    scores = model.decision_function(X_scaled)
                    score_label = 'Anomaly Score'
                else:
                    model = LocalOutlierFactor(n_neighbors=min(20, n_samples-1), contamination=0.1)
                    labels = model.fit_predict(X_scaled)
                    scores = model.negative_outlier_factor_
                    score_label = 'Negative Outlier Factor'

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

                # Scatter with outlier colors
                colors = ['red' if l == -1 else 'steelblue' for l in labels]
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, edgecolors='k', s=50)
                ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                ax1.set_title(f'{algorithm_name}\nRed = Outliers, Blue = Inliers', fontsize=12, fontweight='bold')

                # Score distribution histogram
                ax2.hist(scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                ax2.axvline(x=np.percentile(scores, 10), color='red', linestyle='--', label='Outlier Threshold')
                ax2.set_xlabel(score_label)
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'{score_label} Distribution', fontsize=12, fontweight='bold')
                ax2.legend()

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['anomaly_analysis'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['anomaly_analysis'] = None

        # ========== COMMON VISUALIZATION: Correlation Heatmap ==========
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['correlation_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        except:
            visualizations['correlation_heatmap'] = None

        return visualizations

    # =====================================================================

    def clean_data(self):
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)

        for col in df.select_dtypes(include=['object']).columns:
            if col != self.target_col:
                df[col] = self.le.fit_transform(df[col].astype(str))

        self.feature_names = [c for c in df.columns if c != self.target_col]
        self.df = df
        return df

    def get_eda_summary(self):
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_cols': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object']).columns),
            'missing': self.df.isnull().sum().sum(),
            'basic_stats': self.df.describe().to_dict()
        }

    def train_models(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        if self.problem_type == 'classification':
            y = self.le.fit_transform(y.astype(str))
            class_names = list(self.le.classes_)
        else:
            class_names = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.X_test = X_test
        self.y_test = y_test

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=10),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
                'SVR': SVR(),
                'SVM Regressor': SVR(kernel='rbf'),
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost Regressor'] = XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
        else:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'SVM': SVC(kernel='rbf', random_state=random_state, probability=True),
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost Classifier'] = XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0, use_label_encoder=False, eval_metric='logloss')

        results = {}

        for name, model in models.items():
            try:
                if 'SVR' in name or 'SVM' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                if self.problem_type == 'regression':
                    train_score = model.score(X_train, y_train)
                    test_score = r2_score(y_test, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)

                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = None

                    results[name] = {
                        'train_r2': round(train_score, 4),
                        'test_r2': round(test_score, 4),
                        'test_rmse': round(test_rmse, 4),
                        'test_mae': round(test_mae, 4),
                        'feature_importance': importance
                    }
                else:
                    train_score = model.score(X_train, y_train)
                    test_score = accuracy_score(y_test, y_pred)

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred, output_dict=True)

                    # ROC curve data (for binary classification)
                    roc_data = None
                    if hasattr(model, 'predict_proba') and len(class_names) == 2:
                        try:
                            y_proba = model.predict_proba(X_test_scaled if 'SVM' in name else X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc(fpr, tpr), 4)}
                        except:
                            pass

                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = None

                    results[name] = {
                        'train_accuracy': round(train_score, 4),
                        'test_accuracy': round(test_score, 4),
                        'confusion_matrix': cm.tolist(),
                        'classification_report': cr,
                        'roc_data': roc_data,
                        'feature_importance': importance,
                        'class_names': class_names
                    }

            except Exception as e:
                results[name] = {'error': str(e)}

        self.models_results = results

        # Find best model
        if self.problem_type == 'regression':
            best_name = max(results, key=lambda x: results[x].get('test_r2', 0))
            self.best_score = results[best_name]['test_r2']
        else:
            best_name = max(results, key=lambda x: results[x].get('test_accuracy', 0))
            self.best_score = results[best_name]['test_accuracy']

        self.best_model = best_name
        self.best_model_obj = models[best_name]

        return results

    def create_visualizations(self):
        """Create visualizations for the best model"""
        visualizations = {}

        # Route to unsupervised visualizations
        if self.problem_type == 'unsupervised':
            return self.create_unsupervised_visualizations()

        # 1. Confusion Matrix Heatmap
        if self.problem_type == 'classification' and self.best_model_obj is not None:
            try:
                if 'SVM' in self.best_model or 'SVR' in self.best_model:
                    y_pred = self.best_model_obj.predict(self.scaler.transform(self.X_test))
                else:
                    y_pred = self.best_model_obj.predict(self.X_test)

                cm = confusion_matrix(self.y_test, y_pred)

                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=self.le.classes_,
                          yticklabels=self.le.classes_)
                plt.title(f'Confusion Matrix - {self.best_model}', fontsize=14, fontweight='bold')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except Exception as e:
                visualizations['confusion_matrix'] = None

        # 2. Feature Importance
        if self.best_model_obj is not None and hasattr(self.best_model_obj, 'feature_importances_'):
            try:
                importance = self.best_model_obj.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(15)

                plt.figure(figsize=(10, 8))
                plt.barh(feature_imp['feature'], feature_imp['importance'], color='steelblue')
                plt.xlabel('Importance')
                plt.title(f'Feature Importance - {self.best_model}', fontsize=14, fontweight='bold')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['feature_importance'] = None

        # 3. Classification Report Bar Chart
        if self.problem_type == 'classification':
            best_result = self.models_results.get(self.best_model, {})
            if 'classification_report' in best_result:
                try:
                    cr = best_result['classification_report']
                    metrics_df = pd.DataFrame({
                        'Precision': [cr.get(c, {}).get('precision', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']],
                        'Recall': [cr.get(c, {}).get('recall', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']],
                        'F1-Score': [cr.get(c, {}).get('f1-score', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
                    }, index=[c for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']])

                    if not metrics_df.empty:
                        plt.figure(figsize=(10, 6))
                        metrics_df.plot(kind='bar', colormap='viridis')
                        plt.title('Classification Report', fontsize=14, fontweight='bold')
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        plt.close()
                        buf.seek(0)
                        visualizations['classification_report'] = base64.b64encode(buf.read()).decode('utf-8')
                        buf.close()
                except:
                    visualizations['classification_report'] = None

        # 4. Model Comparison Bar Chart
        try:
            if self.problem_type == 'regression':
                model_scores = {k: v.get('test_r2', 0) for k, v in self.models_results.items() if 'test_r2' in v}
            else:
                model_scores = {k: v.get('test_accuracy', 0) for k, v in self.models_results.items() if 'test_accuracy' in v}

            if model_scores:
                plt.figure(figsize=(12, 6))
                colors = ['#00d4aa' if k == self.best_model else '#667eea' for k in model_scores.keys()]
                plt.bar(model_scores.keys(), model_scores.values(), color=colors)
                plt.title('Model Comparison', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['model_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        except:
            visualizations['model_comparison'] = None

        # 5. Correlation Heatmap
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['correlation_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        except:
            visualizations['correlation_heatmap'] = None

        return visualizations

    def get_results(self):
        return {
            'problem_type': self.problem_type,
            'target_column': self.target_col,
            'focus_column': self.focus_col,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'all_results': self.models_results
        }
    
    def auto_detect_dataset_type(self):

        if self.df is None:
            raise ValueError("Dataset not loaded")


        df = self.df

        print("CHECKING DATASET")


        # remove ID columns
        columns = [
            col for col in df.columns
            if "id" not in col.lower()
        ]


        print("USABLE COLUMNS:", columns)


        # If dataset has no obvious label column
        # treat it as unsupervised

        for col in columns:

            unique = df[col].nunique()

            print(
                col,
                unique,
                df[col].dtype
            )


            # only consider a column target if:
            # - numeric
            # - very few unique values
            # - not text

            if (
                df[col].dtype != "object"
                and unique <= 10
            ):

                return {
                    "type": "supervised",
                    "targets": [col]
                }


        return {
            "type": "unsupervised",
            "targets": []
        }