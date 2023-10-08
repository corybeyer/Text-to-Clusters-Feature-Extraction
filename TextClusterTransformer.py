class TextClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column):
        self.text_column = text_column
        self.best_kmeans = None  # Initialize best_kmeans to None

    def fit(self, X, y=None, return_best_model=False):
        text_data = X[self.text_column]
        
        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # PCA
        pca = TruncatedSVD(n_components=3)
        reduced_data = pca.fit_transform(tfidf_matrix)
        
        # K-means with GridSearchCV
        param_grid = {'n_clusters': [2, 3, 4, 5]}
        kmeans = KMeans()
        grid_search = GridSearchCV(kmeans, param_grid, cv=5)
        grid_search.fit(reduced_data)
        
        self.best_kmeans = grid_search.best_estimator_
        
        if return_best_model:
            return self, self.best_kmeans
        return self

    def transform(self, X, return_best_model=False):
        text_data = X[self.text_column]
        
        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        
        # PCA
        pca = TruncatedSVD(n_components=3)
        reduced_data = pca.fit_transform(tfidf_matrix)
        
        # Get cluster labels from the best model
        cluster_labels = self.best_kmeans.predict(reduced_data)
        
        # Add cluster labels to original data
        X_out = X.copy()
        X_out[self.text_column + '_cluster'] = cluster_labels
        
        if return_best_model:
            return X_out, self.best_kmeans
        return X_out
