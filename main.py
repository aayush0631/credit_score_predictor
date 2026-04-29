import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.core.groupby import categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

df = pd.read_csv('data/archive/german_credit_data.csv')
print(df['Risk'].value_counts())
print(df.head())
print(df.info())
print(df.describe(include='all').T)
print(df['Job'].unique())
print(df.isna().sum())
print(df.duplicated().sum())
df=df.dropna().reset_index(drop=True)
print(df.info())
print(df.columns)
df.drop(columns=['Unnamed: 0'], inplace=True)
df.columns = df.columns.str.strip().str.lower()
df[['age','credit amount','duration']].hist(bins=7,edgecolor='black')
plt.suptitle('distribution of numerical features', fontsize=20)
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(10,5))
for i,col in enumerate(['age','credit amount','duration']):
    plt.subplot(1,3,i+1)
    sns.boxplot(y=df[col],color='skyblue')
    plt.title(col)

plt.tight_layout()
plt.show()
# %%
categorical_cols = ['sex','job','housing','saving accounts','checking account','purpose']
plt.figure(figsize=(10,10))

for i, col in enumerate(categorical_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'distribution of {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
corr=df[['age','job','credit amount','duration']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

#%%

df.groupby('job')['credit amount'].mean()
#%%
df.groupby('sex')['credit amount'].mean()
# we fund male asking higher credit amount

#%%
pd.pivot_table(df,values='credit amount',index='housing',columns='purpose')