import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

import joblib

# ---------------------------
# SETTINGS
# ---------------------------
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv('data/archive/german_credit_data.csv')

# Basic inspection
print(df['Risk'].value_counts())
print(df.head())
print(df.info())
print(df.describe(include='all').T)
print(df.isna().sum())
print(df.duplicated().sum())

# ---------------------------
# DATA CLEANING
# ---------------------------

# Drop useless column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Handle missing values (IMPORTANT: keep data)
df['saving accounts'] = df['saving accounts'].fillna('unknown')
df['checking account'] = df['checking account'].fillna('unknown')

# ---------------------------
# EDA (VISUALS)
# ---------------------------

# Numerical distribution
df[['age','credit amount','duration']].hist(bins=7, edgecolor='black')
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Boxplots (outliers)
plt.figure(figsize=(10,5))
for i, col in enumerate(['age','credit amount','duration']):
    plt.subplot(1,3,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Categorical distributions
categorical_cols = ['sex','job','housing','saving accounts','checking account','purpose']
plt.figure(figsize=(12,10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation
sns.heatmap(df[['age','job','credit amount','duration']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter
sns.scatterplot(data=df, x='age', y='credit amount', hue='sex')
plt.title('Age vs Credit Amount')
plt.show()

# Violin plot
sns.violinplot(
    data=df,
    x='saving accounts',
    y='credit amount',
    hue='saving accounts',
    palette='Pastel1',
    legend=False
)
plt.title('Saving Accounts vs Credit Amount')
plt.show()

# ---------------------------
# FEATURE SELECTION
# ---------------------------
features = ['age','sex','job','housing','saving accounts','checking account','credit amount','duration']
target = 'risk'

df_model = df[features + [target]].copy()

# ---------------------------
# TARGET ENCODING
# ---------------------------
# Convert labels to numeric (ML requires numbers)
df_model[target] = df_model[target].map({'good': 1, 'bad': 0})

# ---------------------------
# FEATURE ENCODING
# ---------------------------
# One-hot encoding avoids false relationships between categories
df_model = pd.get_dummies(df_model, drop_first=True)

# ---------------------------
# SPLIT DATA
# ---------------------------
X = df_model.drop(target, axis=1)
y = df_model[target]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# TRAINING FUNCTION
# ---------------------------
def train_model(name, model, param_grid):
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # better for imbalance
        n_jobs=-1
    )

    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(x_test)
    f1 = f1_score(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))

    return best_model, f1, y_pred

# ---------------------------
# MODELS
# ---------------------------

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000)
lr_param = {'C': [0.1, 1, 10]}
best_lr, f1_lr, pred_lr = train_model("Logistic Regression", lr, lr_param)

# Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_param = {'max_depth': [3,5,10,None]}
best_dt, f1_dt, pred_dt = train_model("Decision Tree", dt, dt_param)

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_param = {'n_estimators':[200], 'max_depth':[5,10,None]}
best_rf, f1_rf, pred_rf = train_model("Random Forest", rf, rf_param)

# Extra Trees
et = ExtraTreesClassifier(class_weight='balanced', random_state=42)
et_param = {'n_estimators':[200], 'max_depth':[5,10,None]}
best_et, f1_et, pred_et = train_model("Extra Trees", et, et_param)

# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)
xgb_param = {
    'n_estimators':[200],
    'max_depth':[3,5],
    'learning_rate':[0.05,0.1]
}
best_xgb, f1_xgb, pred_xgb = train_model("XGBoost", xgb, xgb_param)

# ---------------------------
# FINAL COMPARISON
# ---------------------------
print("\n===== FINAL COMPARISON =====")
print(f"LR F1: {f1_lr:.3f}")
print(f"DT F1: {f1_dt:.3f}")
print(f"RF F1: {f1_rf:.3f}")
print(f"ET F1: {f1_et:.3f}")
print(f"XGB F1: {f1_xgb:.3f}")

# Select best model
best_model = best_xgb

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(y_test, pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
importances = best_xgb.feature_importances_
feat_names = X.columns

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(8,5))
feat_imp.head(10).plot(kind='bar')
plt.title('Top Feature Importance')
plt.show()

# ---------------------------
# SAVE MODEL
# ---------------------------
joblib.dump(best_model, 'credit_model.pkl')
print("Model saved successfully!")
#%%
joblib.dump(X.columns, 'model_columns.pkl')