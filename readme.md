# Breast Cancer Classification Project

## Problem Statement
One of the most prevalent cancers in the world is breast cancer. For patients to survive, an early and precise diagnosis is essential.

This project's goal is to develop and evaluate several machine learning classification models in order to determine whether a tumor is:

1. Cancerous or malignant
2. Benign (not cancerous)

To identify the top-performing model, we apply six distinct classification models and assess them using a variety of performance metrics.

## Dataset
Dataset: UCI Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository

Total Instances: 569

Total Features: 30 numeric features

Target Variable: Binary classification

0 → Malignant

1 → Benign

Feature Types:

All features are numeric and derived from digitized images of fine needle aspirate (FNA) of breast masses.

Examples:

Mean Radius

Mean Texture

Mean Perimeter

Mean Area

Mean Smoothness

etc.

The dataset contains no missing values and is moderately balanced.

## Models Implemented

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

## Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- MCC Score

## Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861   | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286   | 0.8174 |
| kNN                      | 0.9561   | 0.9681 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| Naive Bayes              | 0.9386   | 0.9868 | 0.9452    | 0.9583 | 0.9517   | 0.8676 |
| Random Forest (Ensemble) | 0.9561   | 0.9924 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| XGBoost (Ensemble)       | 0.9474   | 0.9904 | 0.9459    | 0.9722 | 0.9589   | 0.8864 |

## Observation Table

| ML Model Name            | Observation about Model Performance                                                                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved the highest overall performance with 98.25% accuracy and highest MCC (0.9623). Indicates the dataset is nearly linearly separable. Excellent balance between precision and recall. |
| Decision Tree            | Lowest performing model among all. Slight overfitting observed. Lower MCC indicates weaker generalization compared to ensemble methods.                                                     |
| kNN                      | Strong performance after feature scaling. High recall (97.22%) makes it effective in detecting benign cases. Sensitive to feature scaling.                                                  |
| Naive Bayes              | Very high AUC (0.9868) but slightly lower accuracy. Independence assumption may limit performance slightly. Computationally efficient.                                                      |
| Random Forest (Ensemble) | Very strong and stable model. High AUC (0.9924) and strong recall. Reduces overfitting compared to Decision Tree. Robust performance.                                                       |
| XGBoost (Ensemble)       | Excellent AUC (0.9904) and strong recall. Slightly lower accuracy than Logistic Regression but captures complex feature interactions effectively.                                           |

## Insights

Logistic Regression performed best overall, suggesting that the dataset is highly structured and nearly linearly separable.

Ensemble models (Random Forest and XGBoost) provided robust and stable performance.

Decision Tree showed signs of overfitting.

All models achieved AUC > 0.91, indicating strong classification capability.

High MCC values confirm strong correlation between predicted and actual classes.

## How to Run

pip install -r requirements.txt
streamlit run app.py
python app.py