# 🧠 MKT3434 Machine Learning GUI

**MKT3434 Course of Dept. Mechatronics Eng. at YTU instructed by Ertugrul Bayraktar**

---

## 🚀 Overview

This repository provides an advanced GUI framework for exploring and implementing various machine learning algorithms. Built with PyQt6, the GUI offers an intuitive interface for data loading, preprocessing, model training, and visualization. The application supports both classical machine learning techniques and deep learning approaches.

---

## ✨ Key Features

### 📊 Data Management
- **Multiple Dataset Options**: Load built-in datasets (Iris, Breast Cancer, Digits, etc.) or custom CSV files
- **Missing Value Handling**: Comprehensive solutions for handling missing data:
  - Mean Imputation
  - Median Imputation
  - Interpolation
  - Forward Fill
  - Backward Fill
- **Data Scaling**: Various scaling methods including Standard, Min-Max, and Robust scaling
- **Missing Data Visualization**: Interactive visualization of missing data patterns

### 🧮 Classical Machine Learning Algorithms
- **Regression**:
  - Linear Regression
  - SGD Regression with configurable loss functions
- **Classification**:
  - Logistic Regression
  - Support Vector Machines with enhanced kernel options
  - Decision Trees
  - Random Forests
  - K-Nearest Neighbors
  - Naive Bayes with customizable priors
  - SGD Classifier with configurable loss functions

### 🔄 Advanced Model Training
- **Loss Function Selection**:
  - Regression: MSE, MAE, Huber Loss
  - Classification: Cross-Entropy, Hinge Loss
- **Hyperparameter Tuning**: Configure model-specific parameters through the GUI
- **Performance Metrics**: Automatic calculation and display of relevant metrics

### 🧠 Deep Learning
- **Neural Network Design**: Interactive layer-by-layer network construction
- **Layer Types**: Support for Dense, Conv2D, MaxPooling2D, Flatten, and Dropout layers
- **Training Configuration**: Customizable batch size, epochs, and learning rate

### 📈 Visualization
- **Model Performance**: Visualize model predictions and performance metrics
- **Training History**: Plot accuracy and loss curves for neural networks
- **Data Exploration**: Visualize dataset characteristics and patterns

---

## 🏁 Getting Started

### ⚙️ Prerequisites:

Ensure you have the following installed:
- Python 3.8+

### 📦 Required dependencies:

```bash
pip install numpy pandas matplotlib PyQt6 scikit-learn tensorflow scipy
```

### 🚀 Running the Application:

```bash
python mali.py
```

---

## 💡 Usage Guide

1. **Load Data**: Select a dataset from the dropdown or load a custom CSV file
2. **Preprocess Data**: Apply scaling and handle missing values if needed
3. **Select Algorithm**: Choose a machine learning algorithm from the tabs
4. **Configure Parameters**: Set appropriate parameters for the selected algorithm
5. **Train Model**: Click the "Train" button for the selected algorithm
6. **Analyze Results**: View performance metrics and visualizations

---

## 🔧 Recent Enhancements

- **Enhanced SVM Implementation**: Added support for different kernels and hyperparameters
- **Advanced Naive Bayes**: Implemented customizable priors and detailed model explanations
- **Missing Value Handling**: Added comprehensive tools for detecting and handling missing data
- **Loss Function Selection**: Implemented configurable loss functions for applicable algorithms

---

## 📚 Educational Purpose

This GUI is designed as an educational tool for exploring machine learning concepts. It allows users to:
- Understand the impact of different preprocessing techniques
- Compare performance across various algorithms
- Visualize how different parameters affect model behavior
- Gain practical experience with machine learning workflows

---

## 🤝 Contributing

Contributions to improve the functionality, usability, or documentation of this application are welcome!

---

## 📜 License

This project is part of the MKT3434 course at YTU and is intended for educational purposes.
