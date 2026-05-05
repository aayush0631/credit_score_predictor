import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.core.groupby import categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
import joblib

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

categorical_cols = ['sex','job','housing','saving accounts','checking account','purpose']
plt.figure(figsize=(10,10))

for i, col in enumerate(categorical_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'distribution of {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


corr=df[['age','job','credit amount','duration']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()



df.groupby('job')['credit amount'].mean()

df.groupby('sex')['credit amount'].mean()
# we fund male asking higher credit amount


pd.pivot_table(df,values='credit amount',index='housing',columns='purpose')


sns.scatterplot(data=df,x='age',y='credit amount',hue='sex',alpha=0.7,palette='Set1')
plt.title('credit amount vs age colored by sex and sized by duration')
plt.show()


sns.violinplot(
    data=df,
    x='saving accounts',
    y='credit amount',
    hue='saving accounts',
    palette='Pastel1',
    legend=False
)
plt.title('credit amount vs saving accounts')
plt.show()

df['risk'].value_counts(normalize=True)*100
features=['age','sex','job','housing','saving accounts','checking account','credit amount','duration']
target='risk'
df_model=df[features+[target]].copy()
df.head()

cat_cols = df_model.select_dtypes(include=['object', 'string']).columns.drop('risk')
le_dict={}
for col in cat_cols:
    le=LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col]=le
    joblib.dump(le,f'{col}_encoder.pkl')

le_target=LabelEncoder()

target
df_model[target]=le_target.fit_transform(df_model[target])
df_model[target].value_counts(normalize=True)*100
#%%
joblib.dump(le_target,'target_encoder.pkl')

x=df_model.drop(target,axis=1)
y=df_model[target]
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

def train_model(model,param_grid,x_train,y_train,x_test,y_test):
    grid = GridSearchCV(model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)
    grid.fit(x_train,y_train)
    best_model=grid.best_estimator_
    y_pred=best_model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    return best_model,acc,grid.best_params_

dt=DecisionTreeClassifier(random_state=1,class_weight='balanced')
dt_param_grid={
    'max_depth':[3,5,7,10,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
}

best_dt,acc_dt,params_dt=train_model(dt,dt_param_grid,x_train,y_train,x_test,y_test)
print('decision tree accuracy',acc_dt)
print('best parameters',params_dt)

rf=RandomForestClassifier(random_state=1,n_jobs=-1,class_weight='balanced')
rf_param_grid={
    'max_depth':[5,7,10,None],
    'n_estimators':[100,200],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
}

best_rf,acc_rf,param_rf=train_model(rf,rf_param_grid,x_train,y_train,x_test,y_test)
print('random forest accuracy',acc_rf)
print('best parameters',param_rf)