from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Custom transformer for text preprocessing and clustering
class Text_to_ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column, n_clusters= 2, n_components= 2):
        self.text_column = text_column
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.best_kmeans = None
        self.best_n_components = None
        self.input_features = None

    def fit(self, X, y=None):
        self.input_features = list(X.columns)
        text_data = X[self.text_column]
        
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english') #, max_df=0, min_df=1
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # Create a pipeline for PCA and K-means
        pca_kmeans_pipeline = Pipeline([
            ('pca', TruncatedSVD()),
            ('kmeans', KMeans())
        ])
        
        # Create a parameter grid for both PCA and K-means
        param_grid = {
            'pca__n_components': [self.n_components],
            'pca__random_state': [101],
            'kmeans__n_clusters': [self.n_clusters],
            'kmeans__init': ['k-means++', 'random'],
            'kmeans__random_state': [101]
        }
        
        # Use GridSearchCV to find the best combination of n_components and n_clusters
        grid_search = GridSearchCV(pca_kmeans_pipeline, param_grid, cv=5)
        grid_search.fit(tfidf_matrix)
        
        # Store the best K-means model and n_components
        self.best_kmeans = grid_search.best_estimator_.named_steps['kmeans']
        self.best_n_components = grid_search.best_estimator_.named_steps['pca'].n_components
        
        return self

    def transform(self, X):
        text_data = X[self.text_column]
        
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # Apply PCA with the best number of components
        pca = TruncatedSVD(n_components=self.best_n_components)
        reduced_data = pca.fit_transform(tfidf_matrix)
        
        # Get cluster labels from the best K-means model
        cluster_labels = self.best_kmeans.predict(reduced_data)
        
        # Drop the original text column from the DataFrame
        X_transformed = X.drop(columns=[self.text_column])
        
        # Add the cluster labels to the DataFrame
        X_transformed[self.text_column + '_cluster'] = cluster_labels
        
        self.output_features = X_transformed.columns.tolist()
        
        return X_transformed
    
    def get_feature_names_out(self, input_features = None):
        return np.array(self.output_features)
