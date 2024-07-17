#importing libraries

import sklearn
from feature_engine.outliers import ArbitraryOutlierCapper
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pickle import load
from fuzzywuzzy import process
import streamlit as st
import os


#importing the dataset
data = pd.read_csv('modified_insurance.csv')
data

data.info()

data.describe()

data.isnull().sum()


features = ['sex', 'smoker', 'region']
plt.subplots(figsize = (20, 10))
for i, col in enumerate(features):
  plt.subplot(1, 3, i+1)

  x= data[col].value_counts()
  plt.pie(x.values, labels = x.index, autopct = '%1.1f%%')
  plt.title(f'{col}')
plt.show()

features = ['age', 'bmi']

plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
	plt.subplot(1, 2, i + 1)
	sns.scatterplot(data=data, x=col,
				y='expenses',
				hue='smoker')
plt.show()


data.drop_duplicates(inplace=True)
sns.boxplot(data['age'])

sns.boxplot(data['bmi'])

Q1=data['bmi'].quantile(0.25)
Q2=data['bmi'].quantile(0.5)
Q3=data['bmi'].quantile(0.75)
iqr=Q3-Q1
lowlim=Q1-1.5*iqr
upplim=Q3+1.5*iqr
print(lowlim)
print(upplim)


arb=ArbitraryOutlierCapper(min_capping_dict={'bmi':13.6749},max_capping_dict={'bmi':47.315})
data[['bmi']]=arb.fit_transform(data[['bmi']])
sns.boxplot(data['bmi'])

#data wrangling
data['bmi'].skew()
data['age'].skew()

data['sex']=data['sex'].map({'male':0,'female':1})
data['smoker']=data['smoker'].map({'yes':1,'no':0})
data['region']=data['region'].map({'riftvalley':0, 'western':1,'central':2,'coast':3, 'north-eastern': 4})

data.corr()

#model Development

X = data.drop('expenses', axis=1)
y = data['expenses']

l1 = []
l2 = []
l3 = []
cvs = 0

for i in range(40, 50):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
  lrmodel = LinearRegression()
  lrmodel.fit(X_train, y_train)
  l1.append(lrmodel.score(X_train, y_train))
  l2.append(lrmodel.score(X_test, y_test))
  cvs = cross_val_score(lrmodel, X, y, cv=5).mean()
  l3.append(cvs)
  df1=pd.DataFrame({'train acc':l1,'test acc':l2,'cvs':l3})
  df1

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
lrmodel = LinearRegression()
lrmodel.fit(xtrain, ytrain)
print(lrmodel.score(xtrain, ytrain))
print(lrmodel.score(xtest, ytest))
print(cross_val_score(lrmodel, X, y, cv=5).mean())


from sklearn.metrics import r2_score
from sklearn.svm import SVR
svrmodel=SVR()
svrmodel.fit(xtrain,ytrain)
ypredtrain1=svrmodel.predict(xtrain)
ypredtest1=svrmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain1))
print(r2_score(ytest,ypredtest1))
print(cross_val_score(svrmodel,X,y,cv=5,).mean())


rfmodel=RandomForestRegressor(random_state=42)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=RandomForestRegressor(random_state=42)
param_grid={'n_estimators':[10,40,50,98,100,120,150]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
rfmodel=RandomForestRegressor(random_state=42,n_estimators=120)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,y,cv=5,).mean())


gbmodel=GradientBoostingRegressor()
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=GradientBoostingRegressor()
param_grid={'n_estimators':[10,15,19,20,21,50],'learning_rate':[0.1,0.19,0.2,0.21,0.8,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
gbmodel=GradientBoostingRegressor(n_estimators=19,learning_rate=0.2)
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,y,cv=5,).mean())


xgmodel=XGBRegressor()
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=XGBRegressor()
param_grid={'n_estimators':[10,15,20,40,50],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
xgmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,y,cv=5,).mean())

feats=pd.DataFrame(data=grid.best_estimator_.feature_importances_,index=X.columns,columns=['Importance'])
feats

important_features=feats[feats['Importance']>0.01]
important_features


data.columns
df1.columns


Xf=data.drop(data[['expenses']],axis=1)
X=data.drop(data[['expenses']],axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xf,y,test_size=0.2,random_state=42)
finalmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
finalmodel.fit(xtrain,ytrain)
ypredtrain4=finalmodel.predict(xtrain)
ypredtest4=finalmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(finalmodel,X,y,cv=5,).mean())

with open('insurancemodelf.pkl', 'rb') as file:
    finalmodel = load(file)

# Function to generate prediction based on user input
def predict_expenses(age, sex, bmi, children, smoker, region):
    # Mapping categorical inputs
    sex = 0 if sex == 'male' else 1
    smoker = 1 if smoker == 'yes' else 0
    region = {'riftvalley': 0, 'western': 1, 'central': 2, 'coast': 3, 'north-eastern': 3}[region]
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Make the prediction
    predicted_expenses = finalmodel.predict(input_data)
    
    return predicted_expenses[0]

# Example usage with new_data
st.title('Insurance Expense Predictor')

age = st.number_input('Enter age:', min_value=0, max_value=100, value=25)
sex = st.selectbox('Select sex:', ('male', 'female'))
bmi = st.number_input('Enter BMI:', min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input('Enter number of children:', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker:', ('yes', 'no'))
region = st.selectbox('Select region:', ('riftvalley', 'western', 'central', 'coast', 'north-eastern'))

if st.button('Predict'):
    predicted_expenses = predict_expenses(age, sex, bmi, children, smoker, region)
    st.write(f"The predicted insurance expenses are: ${predicted_expenses:.2f}")


    data = {
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Children': children,
        'Smoker': smoker,
        'Region': region,
        'Predicted_Expenses': predicted_expenses
    }
    
    # Append to a CSV file or database
    df = pd.DataFrame(data, index=[0])
    df.to_csv('insurance_predictions.csv', mode='a', header=not os.path.exists('insurance_predictions.csv'), index=False)