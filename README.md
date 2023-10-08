# TextClusterTransformer: A Custom Scikit-Learn Transformer for Text Clustering

## Overview
TextClusterTransformer is a custom Scikit-Learn transformer designed for text preprocessing and clustering. It performs the following operations:

TF-IDF Vectorization: Converts text data into a numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).
Principal Component Analysis (PCA): Reduces the dimensionality of the TF-IDF matrix.
K-means Clustering: Applies K-means clustering to the reduced data to generate cluster labels.
GridSearchCV: Finds the best K-means model based on a predefined set of hyperparameters.

## Installation
This is a custom transformer and doesn't require separate installation. You can include it directly in your Python code.

text_cluster_transformer = TextClusterTransformer(text_column='your_text_column')

## Usage
Here's how to use TextClusterTransformer:

### Initialization
To initialize the transformer, import TextClusterTransformer from your module and then create an instance by specifying the text column you want to process.

### Fitting the Model
To fit the transformer to your data, call the fit method on your TextClusterTransformer instance and pass in your training data.

text_cluster_transformer.fit(X_train)

### Transforming the Data
To transform the data, call the transform method on your TextClusterTransformer instance and pass in the data you want to transform.

X_transformed = text_cluster_transformer.transform(X_train)

### Fitting and Transforming
You can also fit and transform in one step by calling the fit_transform method on your TextClusterTransformer instance and passing in your training data.

X_transformed = text_cluster_transformer.fit_transform(X_train)

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

### Using in a Pipeline
To use TextClusterTransformer in a pipeline, you can add it as one of the steps in your pipeline definition. This allows you to chain multiple preprocessing steps and even end with an estimator.

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('text_cluster', TextClusterTransformer(text_column='your_text_column')),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

### Accessing the Best Model
After fitting your pipeline or TextClusterTransformer, you can access the best model by navigating to the named_steps attribute of your pipeline and then accessing the best_kmeans attribute.

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('text_cluster', TextClusterTransformer(text_column='your_text_column')),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

### Using with GridSearchCV
You can also use TextClusterTransformer as a step in a pipeline that you pass to GridSearchCV. This allows you to perform hyperparameter tuning not just for your estimator but also for the text clustering.

from sklearn.model_selection import GridSearchCV

param_grid = {
    'text_cluster__n_clusters': [2, 3, 4],
    'classifier__n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

## Contributing
This is a custom transformer, and contributions are welcome. Feel free to fork, modify, and submit pull requests.


