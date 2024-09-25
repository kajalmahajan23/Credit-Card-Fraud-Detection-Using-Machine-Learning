# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project aims to detect fraudulent transactions in credit card datasets using various machine learning algorithms. Fraud detection is a significant challenge in the financial sector, and this project employs several classification techniques to identify potentially fraudulent activities.

### Objectives
- To preprocess and analyze the credit card transaction dataset.
- To implement multiple machine learning algorithms for fraud detection.
- To evaluate and compare the performance of each model.
- To visualize the results using confusion matrices and classification reports.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Implementation](#model-implementation)
- [Metrics](#metrics)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used
This project utilizes the following technologies and libraries:
- **Python**: Programming language used for implementation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For creating static visualizations.
- **Seaborn**: For enhancing Matplotlib visualizations.
- **Scikit-learn**: For implementing machine learning algorithms and metrics.
- **Imbalanced-learn**: For oversampling techniques (SMOTE) to address class imbalance.
- **XGBoost**: For gradient boosting algorithm.
- **Power BI**: For advanced data visualization and reporting.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud) from Kaggle.

### Dataset Features
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: 28 anonymized features resulting from a PCA transformation. These features represent the transaction details.
- **Amount**: Transaction amount.
- **Class**: Target variable where `1` indicates a fraudulent transaction and `0` indicates a legitimate transaction.

## Installation
To run this project, you need to have Python installed on your machine. You can create a virtual environment and install the required libraries using pip:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Usage
To run the model, execute the Python script. The code performs the following steps:
1. Load the dataset.
2. Clean the data by removing missing values.
3. Split the data into training and testing sets.
4. Scale the features.
5. Handle imbalanced classes using SMOTE (Synthetic Minority Over-sampling Technique).
6. Train and evaluate the models.
7. Save confusion matrices and classification reports for each model.

## Model Implementation
This project implements the following machine learning models:

1. **Logistic Regression**
   - A statistical method for predicting binary classes.
   - Optimizes log-likelihood using maximum likelihood estimation.

2. **K-Nearest Neighbors (KNN)**
   - A non-parametric method used for classification based on closest training examples in the feature space.
   - Simple and effective for smaller datasets.

3. **XGBoost**
   - An implementation of gradient boosted decision trees designed for speed and performance.
   - Regularization to prevent overfitting, making it suitable for complex datasets.

## Metrics
The performance of each model is evaluated using:
- **Confusion Matrix**: A table used to describe the performance of a classification model.
- **Classification Report**: A report that includes precision, recall, f1-score, and support for each class.

## Results
The project evaluates the performance of each model using metrics such as:
- **Precision**: The accuracy of the positive predictions.
- **Recall**: The ability of the classifier to find all positive instances.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

The results are saved as CSV files named after the respective models:
- `Logistic_Regression_confusion_matrix.csv`
- `Logistic_Regression_classification_report.csv`
- `KNN_confusion_matrix.csv`
- `KNN_classification_report.csv`
- `XGBoost_confusion_matrix.csv`
- `XGBoost_classification_report.csv`

## Visualization
Confusion matrices for each model are visualized using Seaborn heatmaps to provide a clearer understanding of the model's performance. These visualizations are displayed as plots during execution.

## License
This project is licensed under the MIT License.

Created with ❤️ by Kajal Mahajan