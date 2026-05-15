# Credit Card Fraud Detection Using ANN

A deep learning project that uses an Artificial Neural Network (ANN) to detect fraudulent credit card transactions from the Kaggle Credit Card Fraud Detection dataset.

##  Dataset

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains credit card transactions made by European cardholders in September 2013. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions (highly imbalanced dataset - 0.172% fraud cases).

### Features:
- **V1-V28:** Principal components obtained with PCA (anonymized features)
- **Time:** Seconds elapsed between each transaction and the first transaction
- **Amount:** Transaction amount
- **Class:** Target variable (1 = fraud, 0 = normal)

##  Project Overview

This project implements a 3-layer Artificial Neural Network to classify credit card transactions as fraudulent or legitimate. The model addresses the class imbalance problem through undersampling and achieves robust fraud detection performance.

##  Model Architecture

```
Input Layer:  30 features (V1-V28, Time, Amount)
Hidden Layer 1: 32 neurons, ReLU activation
Hidden Layer 2: 16 neurons, ReLU activation
Output Layer:   1 neuron, Sigmoid activation
```

**Total Parameters:** The model uses a simple yet effective architecture optimized for binary classification.

##  Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - Data preprocessing and evaluation metrics
- **Matplotlib** - Data visualization


```

## Workflow

### 1. **Data Loading & Exploration**
- Load the dataset using Pandas
- Analyze class distribution (Normal vs Fraud)
- Visualize class imbalance

### 2. **Data Preprocessing**
- **Feature Scaling:** Standardize `Time` and `Amount` features (V1-V28 are already PCA-transformed)
- **Class Balancing:** Use undersampling technique
  - All 492 fraud cases included
  - 2,460 normal cases sampled (5:1 ratio)
  - Total balanced dataset: 2,952 samples

### 3. **Data Splitting**
- **Training Set:** 80% of balanced data
- **Test Set:** 20% of balanced data
- Validation split: 20% of training data used during training

### 4. **Model Training**
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 20
- **Batch Size:** 32

### 5. **Model Evaluation**
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Training/Validation Loss and Accuracy curves

##  Results

The model is evaluated using:
- **Precision:** How many predicted frauds are actually frauds
- **Recall:** How many actual frauds are detected
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives

Training and validation curves are plotted to monitor:
- Loss convergence
- Accuracy improvement
- Potential overfitting

##  Repository Structure

```
Credit-Card-Fraud-Detection/
│
├── fraud_detection_v2.ipynb          # Main Jupyter notebook
├── fraud_detection_presentation.pptx # Project presentation
├── README.md                          # Project documentation
└── creditcard.csv                     # Dataset (not included - download separately)
```

## Key Concepts

### Class Imbalance Handling
The original dataset is highly imbalanced (99.83% normal, 0.17% fraud). This project uses **undersampling** to create a balanced training set with a 5:1 ratio of normal to fraud cases.

### Why ANN?
Artificial Neural Networks are effective for:
- Learning complex, non-linear patterns
- Automatic feature extraction
- Binary classification tasks
- Handling high-dimensional data

##  Potential Improvements

- Experiment with different balancing techniques (SMOTE, oversampling)
- Try different architectures (more layers, different neuron counts)
- Add dropout layers to prevent overfitting
- Implement class weights instead of undersampling
- Cross-validation for more robust evaluation
- Hyperparameter tuning (learning rate, batch size, epochs)
- Compare with other algorithms (Random Forest, XGBoost, SVM)

#
