# -*- coding: utf-8 -*-
"""Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fTrzaJDaoNJ2_d4Ev5BxysoygJwtya7s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pip install seaborn
import seaborn as sns
import os
import math
import scipy.stats
import copy
!pip install scikit-learn


#%%
os.chdir("C:\\Users\\18045\\Documents\\Python\\datamining\\Project") #Sean's Directory
#os.chdir() #Linsie's directory
#os.chdir() #Sreya's directory
RECS_DF = pd.read_csv("recs2020_public_v5.csv")
code_book = pd.read_excel('RECS 2020 Codebook for Public File - v5.xlsx')
'''
#Pulling the columns of interest. 

data_df = RECS_DF.iloc[:,:9]

column_names_list = ['TYPEHUQ', 'INTERNET', 'HHSEX', 'HHAGE', 'EMPLOYHH', 'EDUCATION',
                     'HOUSEHOLDER_RACE', 'COLDMA', 'HOTMA', 'NOACEL','NOHEATEL', 'MONEYPY','PAYHELP']

if all(col_name in RECS_DF.columns for col_name in column_names_list):
    data = RECS_DF[column_names_list]
    
RECs_dfs = pd.concat([data_df, data], axis=1)
'''
#%%

#import data (smaller sized csv of our key attributes only - see github for the code I used)
#make sure to drag the csv file (in our drive) to the colab variable panel.
RECs_dfs = pd.read_csv("RECsample_df.csv")

'''
Initial corr plot
'''
sample_df_corr = RECs_dfs.iloc[:,9:]
sns.heatmap(sample_df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")

"""Based on our corr plot is looks like:
  EMPLOYHH and HHAGE,
  EDUCATION and MONEYPY,
  NOACEL and NOHEATEL,
are *relatively*, positively correlated.

It also looks like:
  EMPLOYHH and MONEYPY,
  EMPLOYHH and EDUCATION,
  MONEYPY and PAYHELP,
are *relativley* negatively correlated.
"""

data_rf = sample_df_corr.drop('UATYP10', axis =1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = data_rf ['PAYHELP']
X = data_rf .drop('PAYHELP', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

rf = RandomForestClassifier(n_estimators=5, random_state=5)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

"""(need to analyze this)

Now we'll assign our levels for each variable.
"""
#%%
#Map TYPEHUC Type factors
mapping = {1: 'Mobile home',
           2: 'Single-family house detached from any other house',
           3: 'Single-family house attached to one or more other houses',
           4: 'Apartment in a building with 2 to 4 units',
           5: 'Apartment in a building with 5 or more units'}

RECs_dfs['TYPEHUQ'] = RECs_dfs['TYPEHUQ'].map(mapping)

#Map INTERNET Type factors
mapping = {0: 'No access to the Internet',
           1: 'Yes, via cell phone/Internet provider',
           2: 'Yes, without cell phone/Internet provider'}

RECs_dfs['INTERNET'] = RECs_dfs['INTERNET'].map(mapping)

#Map HHSEX Type factors
mapping = {1: 'Female',
           2: 'Male'}

RECs_dfs['HHSEX'] = RECs_dfs['HHSEX'].map(mapping)

#Map EMPLOYHH Type factors
mapping = {1: 'Employed full-time',
           2: 'Employed part-time',
           3: 'Retired',
           4: 'Not employed'}

RECs_dfs['EMPLOYHH'] = RECs_dfs['EMPLOYHH'].map(mapping)

#Map EDUCATION Type factors
mapping = {1: 'Less than high school diploma or GED',
           2: 'High school diploma or GED',
           3: 'Some College of Associates degree',
           4: 'Bachelor’s degree',
           5: 'Master’s, Professional, or Doctoral degree'}

RECs_dfs['EDUCATION'] = RECs_dfs['EDUCATION'].map(mapping)

#Map HOUSEHOLDER_RACE Type factors
mapping = {1: 'White Alone',
           2: 'Black or African/American Alone',
           3: 'American Indian or Alaska Native Alone',
           4: 'Asian Alone',
           5: 'Native Hawaiian or Other Pacific Islander Alone',
           6: '2 or More Races Selected'}

RECs_dfs['HOUSEHOLDER_RACE'] = RECs_dfs['HOUSEHOLDER_RACE'].map(mapping)

#Map 'COLDMA', 'HOTMA', 'NOACEL','NOHEATEL', 'PAYHELP' Type factors
mapping = {1: 'Yes',
           0: 'No',
           -2: "NA"}

RECs_dfs['COLDMA'] = RECs_dfs['COLDMA'].map(mapping)
RECs_dfs['HOTMA'] = RECs_dfs['HOTMA'].map(mapping)
RECs_dfs['NOACEL'] = RECs_dfs['NOACEL'].map(mapping)
RECs_dfs['NOHEATEL'] = RECs_dfs['NOHEATEL'].map(mapping)
RECs_dfs['PAYHELP'] = RECs_dfs['PAYHELP'].map(mapping)

#Map MONEYPY Type factors
mapping = {1: 'Less than $5,000',
           2: '$5,000 - $7,499',
           3: '$7,500 - $9,999',
           4: '$10,000 - $12,499',
           5: '$12,500 - $14,999',
           6: '$15,000 - $19,999',
           7: '$20,000 - $24,999',
           8: '$25,000 - $29,999',
           9: '$30,000 - $34,999',
           10: '$35,000 - $39,999',
           11: '$40,000 - $49,999',
           12: '$50,000 - $59,999',
           13: '$60,000 - $74,999',
           14: '$75,000 - $99,999',
           15: '$100,000 - $149,999',
           16: '$150,000 or more'}

RECs_dfs['MONEYPY'] = RECs_dfs['MONEYPY'].map(mapping)
#%%
"""Figures and other Plots."""

#PLOT Income by Education
money_order = ['Less than $5,000', '$5,000 - $7,499', '$7,500 - $9,999', '$10,000 - $12,499', '$12,500 - $14,999',
               '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999', '$30,000 - $34,999', '$35,000 - $39,999',
               '$40,000 - $49,999', '$50,000 - $59,999', '$60,000 - $74,999', '$75,000 - $99,999', '$100,000 - $149,999',
               '$150,000 or more']

edu_order = ['Less than high school diploma or GED',
             'High school diploma or GED',
             'Some College of Associates degree',
             'Bachelor’s degree',
             'Master’s, Professional, or Doctoral degree']

plt.figure(figsize=(12, 6))
sns.countplot(data=RECs_dfs, x ="MONEYPY", hue='EDUCATION',  palette="Set2", order = money_order, hue_order = edu_order)
plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Income Level by Education")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Education", labels= edu_order)
plt.show()
#%%
#EMPLOYHH by MONEYPY

plt.figure(figsize=(12, 6))
sns.countplot(data=RECs_dfs , x="MONEYPY", hue="EMPLOYHH", order=money_order, alpha=0.6)
plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Employment Status by Income")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Employment Status")
plt.show()

RECs_dfs_filtered = RECs_dfs[RECs_dfs['COLDMA'].isin(['Yes', 'No'])]
#%%
#PLOT Income by Assisstance
money_order = ['Less than $5,000', '$5,000 - $7,499', '$7,500 - $9,999', '$10,000 - $12,499', '$12,500 - $14,999',
               '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999', '$30,000 - $34,999', '$35,000 - $39,999',
               '$40,000 - $49,999', '$50,000 - $59,999', '$60,000 - $74,999', '$75,000 - $99,999', '$100,000 - $149,999',
               '$150,000 or more']

assistance_order = ['Yes', 'No', 'NA']

plt.figure(figsize=(12, 6))

#sns.countplot(data=RECs_dfs, x="MONEYPY", hue='COLDMA', order=money_order, hue_order=assistance_order)
#sns.countplot(data=RECs_dfs, x="MONEYPY", hue='HOTMA',  order=money_order, hue_order=assistance_order)
#sns.countplot(data=RECs_dfs, x="MONEYPY", hue='NOACEL',  order=money_order, hue_order=assistance_order)
#sns.countplot(data=RECs_dfs, x="MONEYPY", hue='NOHEATEL',  order=money_order, hue_order=assistance_order)
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='PAYHELP',  order=money_order, hue_order=assistance_order)

plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Income by All Assistance Level")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Assistance Inputation", labels= assistance_order)
plt.show()

money_order = ['Less than $5,000', '$5,000 - $7,499', '$7,500 - $9,999', '$10,000 - $12,499', '$12,500 - $14,999',
               '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999', '$30,000 - $34,999', '$35,000 - $39,999',
               '$40,000 - $49,999', '$50,000 - $59,999', '$60,000 - $74,999', '$75,000 - $99,999', '$100,000 - $149,999',
               '$150,000 or more']

assistance_order = ['Yes']

filtered_df = RECs_dfs[RECs_dfs['COLDMA'] == 'Yes']
plt.figure(figsize=(12, 6))
sns.countplot(data=filtered_df , x="MONEYPY", hue='COLDMA',  order=money_order, hue_order=assistance_order, alpha=0.6)
plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Medical Attention Needed From Cold by Income")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Medical Attention for Cold", labels= assistance_order )
plt.show()

filtered_df_hot = RECs_dfs[RECs_dfs['HOTMA'] == 'Yes']
plt.figure(figsize=(12, 6))
sns.countplot(data=filtered_df_hot , x="MONEYPY", hue='HOTMA',  order=money_order, hue_order=assistance_order, alpha=0.6, palette = 'autumn_r')
plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Medical Attention Needed From Heat by Income")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Medical Attention for Heat", labels= assistance_order )
plt.show()
#%%
# Chi squared test for income braket and hotma and then coldma

from scipy.stats import chi2_contingency

table = pd.crosstab(RECs_dfs['MONEYPY'], RECs_dfs['HOTMA'])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p)

table = pd.crosstab(RECs_dfs['MONEYPY'], RECs_dfs['COLDMA'])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p)

"""It looks like COLDMA is more significant on income than HOTMA. This may be due to the lower costs of taking a cool shower in heat, or the access to a freezer (ice water) in heat, than access to the variety of fuels to heat a home."""

table = pd.crosstab(RECs_dfs['MONEYPY'], RECs_dfs['PAYHELP'])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p)

"""Predictable results where PAYHELP is significant with income levels. Nothing too insightful here."""
#%%
# Logistical Prediction
# test accuracy with prediction score
#!pip install statsmodels

import statsmodels.api as sm
from statsmodels.formula.api import glm

data = RECs_dfs
data = data[data['PAYHELP'] != 'NA']

model = glm(formula='PAYHELP ~  HHAGE + C(MONEYPY) + C(HOTMA) + C(COLDMA) ', data=data, family=sm.families.Binomial())
model_fit = model.fit()
print(model_fit.summary())

from sklearn.metrics import confusion_matrix
confusion_matrix()

RECs_dfs.columns


#%%
'''
Linear Regression of Income level by attributes of education and cold medical assistance.

'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Selecting our columns, and making them dummies
selected_columns = ['MONEYPY', 'HOUSEHOLDER_RACE', 'EDUCATION', 'COLDMA']

df = RECS_DF[selected_columns]

df['MONEYPY'] = pd.to_numeric(df['MONEYPY'], errors='coerce')  

df = pd.get_dummies(df, columns=['HOUSEHOLDER_RACE', 'EDUCATION'], drop_first=True)

#remove NAs
df.dropna(inplace=True)

#Splitting the dataset for Y and Xs, then training and testing
X = df.drop('COLDMA', axis=1).values
y = df['COLDMA'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Results and performance of our data
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

colnames = pd.DataFrame(df.columns)
coefs_df = pd.DataFrame(model.coef_)

coefs_colname = pd.concat([colnames, coefs_df], axis = 1)
coefs_colname.columns = ['Name', 'Coefficient'] 

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

print(f'Here are our parameters:\n {coefs_colname}') 
print(f'Here is our intercept: {model.intercept_}')
'''
Our linear regression X variables explain 1.02% of the Y variable(income) based on the r-squared value.
So overall our regression should not be considered in our analysis.
This is helpful to know since we can explore other models. 
Despite this poor performance, we can gleam that lower education statuses correlate to lower incomes.
'''


#%%
'''
Multilinear Regression

'''

# Selecting our columns and subsetting the data.
selected_columns = [
    'COLDMA',
    'MONEYPY',
    'HHAGE',
    'PAYHELP',
    'NOACEL',
    'NOHEATEL',
    'EDUCATION',
    'HOUSEHOLDER_RACE'
]

selected_data = RECS_DF[selected_columns].copy()  # copying avoid chained assignment

# Converting columns to numeric
selected_data = selected_data.apply(pd.to_numeric, errors='coerce')

selected_data = selected_data.dropna()

print("Shape after dropping missing values:", selected_data.shape)

# Convert 'PAYHELP' column to categorical
selected_data['PAYHELP'] = pd.Categorical(selected_data['PAYHELP'], categories=[-2, 0, 1])

# Convert 'PAYHELP' to dummy variables
selected_data = pd.get_dummies(selected_data, columns=['PAYHELP'], prefix='PAYHELP', drop_first=True)

# Convert 'EDUCATION' to dummy variables
selected_data = pd.get_dummies(selected_data, columns=['EDUCATION'], prefix='EDU', drop_first=True)

# Convert 'HOUSEHOLDER_RACE' to dummy variables
selected_data = pd.get_dummies(selected_data, columns=['HOUSEHOLDER_RACE'], prefix='RACE', drop_first=True)

# Add a constant term for the intercept
selected_data = sm.add_constant(selected_data)
