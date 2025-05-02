Machine Learning Course GUI
Overview
This application provides a comprehensive graphical user interface for exploring machine learning algorithms, dimensionality reduction techniques, and data visualization methods. It is designed for educational purposes in robotics and mechatronics applications, focusing on sensor data compression, feature extraction, and techniques to prevent overfitting.
Features
Data Management

Load built-in datasets (Iris, Breast Cancer, Digits, Boston Housing, MNIST)
Import custom datasets from CSV files
Apply various preprocessing techniques:

Data scaling (Standard, Min-Max, Robust)
Missing value handling (Mean/Median imputation, Interpolation, Forward/Backward Fill)
Custom train-test-validation splitting



Classical Machine Learning

Regression Algorithms:

Linear Regression
Logistic Regression
SGD Regression


Classification Algorithms:

Naive Bayes
Support Vector Machines
Decision Trees
Random Forests
K-Nearest Neighbors
SGD Classifier



Dimensionality Reduction

Linear Methods:

Principal Component Analysis (PCA) with explained variance visualization
Linear Discriminant Analysis (LDA) with class separation metrics


Clustering:

K-Means with elbow method for optimal k selection
Silhouette score calculation for clustering quality evaluation


Non-linear Methods:

t-SNE with interactive 2D/3D projections and perplexity tuning
UMAP (Uniform Manifold Approximation and Projection) as a faster t-SNE alternative


Mathematical Foundations:

Eigenvector computation for covariance matrices
Data projection visualizations



Deep Learning

Multi-Layer Perceptron architecture design
CNN/RNN architecture templates
Customizable training parameters

Validation Techniques

K-fold cross-validation with multiple metrics (Accuracy, MSE, RMSE, R²)
Custom dataset splitting options (e.g., 70-15-15)
Performance metrics visualization

Visualization

Interactive 2D/3D plots with Matplotlib
Enhanced visualization with Plotly for better data exploration
Cluster quality assessment
Training history visualization for neural networks

Requirements

Python 3.6+
Required packages:

PyQt6 (GUI framework)
NumPy (numerical operations)
Pandas (data manipulation)
Matplotlib (visualization)
scikit-learn (machine learning algorithms)
TensorFlow (deep learning)
Plotly (enhanced visualization, optional)
UMAP-learn (dimensionality reduction, optional)



Installation

Ensure Python 3.6 or higher is installed
Install required packages:

bashpip install pyqt6 numpy pandas matplotlib scikit-learn tensorflow
pip install plotly umap-learn
Usage

Run the application:

bashpython nesly_29102023.py

Load a dataset using the dropdown menu or import your own CSV file
Select preprocessing options (scaling, missing value handling)
Navigate between tabs to access different ML algorithms and visualization techniques
Train models and visualize results

Important Notes

When using the Boston Housing dataset, the application will fetch it directly from its original source, as it has been removed from newer versions of scikit-learn.
The visualization tab provides interactive features when using Plotly. A web browser window will open for full interactivity.
For large datasets, some dimensionality reduction methods (particularly t-SNE) may take considerable time to compute.
When running Silhouette Score analysis, do not use other visualization functions until you're done examining the results.

Troubleshooting

If visualization components don't render properly, try resizing the window.
If encountering "NaN" errors, try different missing value handling methods.
For memory errors on large datasets, try reducing the sample size or using UMAP instead of t-SNE.

Acknowledgments
This application was developed as an educational tool for machine learning courses, particularly focused on applications in robotics and mechatronics.
