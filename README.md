# Telco Customer Churn Prediction using Ensemble Learning

## Project Overview

This project is a Machine Learning application for predicting whether a telecom customer is likely to churn.

The project focuses on Ensemble Learning and compares different Machine Learning approaches, including Logistic Regression, Support Vector Machine (SVM), Random Forest, Gradient Boosting, and Stacking.

The final trained model is integrated into a Flask backend and an interactive web frontend where users can enter customer information and receive a churn prediction with its probability.

---

## Project Objectives

The main objectives of this project are:

- Predict customer churn using Machine Learning.
- Preprocess numerical and categorical customer data.
- Compare different Machine Learning models.
- Apply Ensemble Learning techniques.
- Evaluate model performance using multiple metrics.
- Build a complete Machine Learning application.
- Connect the trained model to a Flask backend.
- Provide an interactive frontend for making predictions on new customers.

---

# Machine Learning Models

Several models were trained and evaluated during the project.

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

It predicts the probability of a customer belonging to the churn or non-churn class.

The model was configured with balanced class weights to help handle the class distribution.

---

## 2. Support Vector Machine (SVM)

A Support Vector Classifier (SVC) was used with an RBF kernel.

SVM attempts to find a decision boundary that separates the different classes.

The model was configured with balanced class weights and probability estimation to support probability-based predictions.

---

## 3. Random Forest

Random Forest was used as the main Bagging-based ensemble model.

It combines multiple decision trees trained using different bootstrap samples of the training data.

The predictions of the individual trees are combined to produce the final prediction.

The Random Forest model used:

- 100 trees
- Balanced class weights
- Random state = 42

---

# Ensemble Learning Techniques

## Bagging

Bagging stands for Bootstrap Aggregating.

The idea is to train multiple models using different bootstrap samples of the training data and then combine their predictions.

In this project, Random Forest represents the Bagging approach.

```text
Training Data
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
Tree Tree Tree
 ↓    ↓    ↓
 └────┼────┘
      ↓
 Combined Prediction
```

## Boosting

Gradient Boosting was also used in the project.

Instead of building independent models, Boosting builds models sequentially.

Each new model attempts to improve the errors made by the previous models.

The Gradient Boosting model used:

- 1800 estimators
- Learning rate = 0.05
- Maximum tree depth = 2
- Random state = 42

## Stacking Ensemble

Stacking was used to combine predictions from different Machine Learning models.

The Stacking ensemble contains three base learners:

- Logistic Regression
- SVM
- Random Forest

Their prediction probabilities are passed to a final Logistic Regression model called the meta-learner.
```text
                Input Features
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
 Logistic Regression  SVM    Random Forest
        ↓             ↓             ↓
        └─────────────┼─────────────┘
                      ↓
              Prediction Probabilities
                      ↓
             Logistic Regression
                Meta-Learner
                      ↓
                Final Prediction
```
The Stacking model uses 5-fold cross-validation and predict_proba as the stacking method.

---

## Data Preprocessing

The dataset contains both numerical and categorical features.

The preprocessing pipeline includes:

- Handling numerical features
- Scaling numerical features
- Encoding categorical features
- Preparing the data for Machine Learning models

The same preprocessing steps are applied when making predictions on new customers.

---

## Model Evaluation

The models were evaluated and compared using classification performance metrics.

The project includes evaluation using metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

The model comparison helps determine which approach performs better on the validation/test data.

---

## Web Application

The Machine Learning model was integrated into a web application.

The frontend allows the user to enter customer information through an interactive form.

The application then sends the information to the Flask backend.

The backend:

- Receives the customer data.
- Prepares the input data.
- Applies the saved preprocessing steps.
- Loads the trained Machine Learning model.
- Generates the prediction.
- Calculates the prediction probability.
- Returns the result to the frontend.

---

## Technologies Used
### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Ensemble Learning
### Backend
- Flask
- Python
- SQLite
- Pickle
### Frontend
- HTML
- CSS
- JavaScript
### Development Environment
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## Author

**Abdallah Mohamed Sayed**

GitHub:
https://github.com/Abdallah443

LinkedIn:
https://www.linkedin.com/in/abdallah-mohamed-367973281/
