# Text-to-Clusters-Feature-Extraction
The TextClusterTransformer class is a custom transformer for text preprocessing and clustering. It inherits from scikit-learn's BaseEstimator and TransformerMixin classes. The transformer performs the following steps:
TextClusterTransformer Class
Overview
The TextClusterTransformer class is a custom transformer for text preprocessing and clustering. It inherits from scikit-learn's BaseEstimator and TransformerMixin classes. The transformer performs the following steps:

TF-IDF Vectorization: Converts text data into a TF-IDF matrix.
Dimensionality Reduction: Applies Principal Component Analysis (PCA) to reduce the dimensionality of the TF-IDF matrix.
K-means Clustering: Uses K-means clustering to cluster the reduced-dimensionality data.
Grid Search: Finds the best K-means model using GridSearchCV.
Class Attributes
text_column: The name of the DataFrame column containing the text data to be processed.
best_kmeans: Stores the best K-means model found by GridSearchCV.
Methods
__init__(self, text_column)
Parameters
text_column: str
The name of the DataFrame column containing the text data to be processed.
Returns
None

fit(self, X, y=None, return_best_model=False)
Fits the transformer to the data.

Parameters
X: pandas DataFrame
The input data.
y: Ignored
Not used, present for API consistency by convention.
return_best_model: bool, default=False
If True, returns the best K-means model along with self.
Returns
self: The fitted transformer.
self.best_kmeans: The best K-means model (only if return_best_model=True).
transform(self, X, return_best_model=False)
Transforms the data.

Parameters
X: pandas DataFrame
The input data.
return_best_model: bool, default=False
If True, returns the best K-means model along with the transformed data.
Returns
X_out: pandas DataFrame
The transformed data with an additional column containing the cluster labels.
self.best_kmeans: The best K-means model (only if return_best_model=True).
