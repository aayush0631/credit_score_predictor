# 💳 Credit Risk Prediction System

An end-to-end Machine Learning project that predicts whether a loan applicant is a **good or bad credit risk** using financial and demographic data. The project includes **data analysis, visualization, model training, and a Streamlit web app for real-time predictions**.

---

## 🚀 Live Demo (Optional)
If deployed:
https://your-streamlit-link-here

---

## 📊 Problem Statement

Banks and financial institutions need to evaluate whether a customer is likely to repay a loan. This project builds a predictive system to classify applicants as:

- ✅ Good Credit Risk
- ❌ Bad Credit Risk

---

## 📁 Dataset

- German Credit Dataset
- Features include:
  - Age
  - Sex
  - Job
  - Housing type
  - Saving & checking accounts
  - Credit amount
  - Loan duration
  - Purpose

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Distribution of numerical features
- Boxplots for outlier detection
- Count plots for categorical variables
- Correlation heatmap
- Relationship between age, credit amount, and risk factors

---

## ⚙️ Data Preprocessing

- Handled missing values by replacing with `"unknown"`
- Converted categorical variables using **One-Hot Encoding**
- Encoded target variable:
  - `good → 1`
  - `bad → 0`
- Split dataset into training and testing sets (80/20)

---

## 🤖 Machine Learning Models Used

The following models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier (Best Performing Model)

---

## 🏆 Best Model

✔ **XGBoost Classifier**  
✔ Highest F1-score and accuracy  
✔ Handles imbalance and non-linear relationships well  

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🖥️ Web App (Streamlit)

A user-friendly interface built using Streamlit allows users to:

- Input customer details
- Get real-time prediction
- View credit risk instantly

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 📦 Installation

```bash
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model
pip install -r requirements.txt
