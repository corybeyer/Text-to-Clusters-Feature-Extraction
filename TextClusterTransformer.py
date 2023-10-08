from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# Custom transformer for text preprocessing and clustering
class TextClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column):
        # Initialize the text column and best K-means model
        self.text_column = text_column
        self.best_kmeans = None

    def fit(self, X, y=None, return_best_model=False):
        # Extract the text data from the given column
        text_data = X[self.text_column]
        
        # Step 1: Perform TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # Step 2: Perform PCA to reduce dimensionality
        pca = TruncatedSVD(n_components=3)
        reduced_data = pca.fit_transform(tfidf_matrix)
        
        # Step 3: Perform K-means clustering using GridSearchCV to find the best model
        param_grid = {'n_clusters': [2, 3, 4, 5]}
        kmeans = KMeans()
        grid_search = GridSearchCV(kmeans, param_grid, cv=5)
        grid_search.fit(reduced_data)
        
        # Store the best K-means model
        self.best_kmeans = grid_search.best_estimator_
        
        # Optionally return the best model
        if return_best_model:
            return self, self.best_kmeans
        return self

    def transform(self, X, return_best_model=False):
        # Extract the text data from the given column
        text_data = X[self.text_column]
        
        # Step 1: Perform TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # Step 2: Perform PCA to reduce dimensionality
        pca = TruncatedSVD(n_components=3)
        reduced_data = pca.fit_transform(tfidf_matrix)
        
        # Step 3: Use the best K-means model to get cluster labels
        cluster_labels = self.best_kmeans.predict(reduced_data)
        
        # Step 4: Add cluster labels to the original data
        X_out = X.copy()
        X_out = X_out.drop(self.text_column, axis = 1)
        X_out[self.text_column + '_cluster'] = cluster_labels
        
        # Optionally return the best model
        if return_best_model:
            return X_out, self.best_kmeans
        return X_out
