# TextClusterTransformer: A Custom Scikit-Learn Transformer for Text Clustering

## Overview
TextClusterTransformer is a custom Scikit-Learn transformer designed for text preprocessing and clustering. It performs the following operations:

TF-IDF Vectorization: Converts text data into a numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).
Principal Component Analysis (PCA): Reduces the dimensionality of the TF-IDF matrix.
K-means Clustering: Applies K-means clustering to the reduced data to generate cluster labels.
GridSearchCV: Finds the best K-means model based on a predefined set of hyperparameters.

## Installation
This is a custom transformer and doesn't require separate installation. You can include it directly in your Python code.

## Usage
Here's how to use TextClusterTransformer:

### Initialization
To initialize the transformer, import TextClusterTransformer from your module and then create an instance by specifying the text column you want to process.

### Fitting the Model
To fit the transformer to your data, call the fit method on your TextClusterTransformer instance and pass in your training data.

### Transforming the Data
To transform the data, call the transform method on your TextClusterTransformer instance and pass in the data you want to transform.

### Fitting and Transforming
You can also fit and transform in one step by calling the fit_transform method on your TextClusterTransformer instance and passing in your training data.

## Methods
__init__(self, text_column)
Initializes the transformer. Takes the name of the text column to be processed.

fit(self, X, y=None, return_best_model=False)
Fits the transformer to the data. Optionally returns the best K-means model. Parameters include X for the input data, y which is not used but included for compatibility, and return_best_model to optionally return the best K-means model.

transform(self, X, return_best_model=False)
Transforms the data. Optionally returns the best K-means model. Parameters include X for the input data and return_best_model to optionally return the best K-means model.

## Attributes
best_kmeans: Stores the best K-means model found during fitting.

## Example
To use the TextClusterTransformer, initialize it by specifying the text column, fit it to your training data, and then transform your data.

Contributing
This is a custom transformer, and contributions are welcome. Feel free to fork, modify, and submit pull requests.


