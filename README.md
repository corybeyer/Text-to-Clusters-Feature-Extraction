TextClusterTransformer: A Custom Scikit-Learn Transformer for Text Clustering
Overview
TextClusterTransformer is a custom Scikit-Learn transformer designed for text preprocessing and clustering. It performs the following operations:

TF-IDF Vectorization: Converts text data into a numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).
Principal Component Analysis (PCA): Reduces the dimensionality of the TF-IDF matrix.
K-means Clustering: Applies K-means clustering to the reduced data to generate cluster labels.
GridSearchCV: Finds the best K-means model based on a predefined set of hyperparameters.
Installation
This is a custom transformer and doesn't require separate installation. You can include it directly in your Python code.

Usage
Here's how to use TextClusterTransformer:

Initialization
import TextClusterTransformer  # Replace 'your_module' with the actual module name
# Initialize the transformer
text_cluster_transformer = TextClusterTransformer(text_column='your_text_column')



