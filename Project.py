# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:48:21 2023

@author: 18045
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pip install seaborn
import seaborn as sns
import os
import math
import scipy.stats
import copy

os.chdir("C:\\Users\\18045\\Documents\\Python\\datamining\\Project") #Sean's Directory
#os.chdir() #Linsie's directory
#os.chdir() #Sreya's directory
RECS_DF = pd.read_csv("recs2020_public_v5.csv")
code_book = pd.read_excel('RECS 2020 Codebook for Public File - v5.xlsx')

#%%
'''
Pulling the columns of interest. 
'''

data_df = RECS_DF.iloc[:,:9]

column_names_list = ['TYPEHUQ', 'INTERNET', 'HHSEX', 'HHAGE', 'EMPLOYHH', 'EDUCATION',
                     'HOUSEHOLDER_RACE', 'COLDMA', 'HOTMA', 'NOACEL','NOHEATEL', 'MONEYPY','PAYHELP']

if all(col_name in RECS_DF.columns for col_name in column_names_list):
    data = RECS_DF[column_names_list]
    
RECs_dfs = pd.concat([data_df, data], axis=1)

#%% 
'''
Initial corr plot
'''

sample_df_corr = RECs_dfs.iloc[:,9:]
sns.heatmap(sample_df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")

#%%

#RECs_dfs.to_csv('RECsample_df.csv')

#%%

pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = data['PAYHELP']  
X = data.drop('PAYHELP', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=10, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#%%
'''
Applying level info
'''

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
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='EDUCATION', palette="Set2", order = money_order, hue_order = edu_order)
plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Income Level by Education")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Education", labels= edu_order)
plt.show()

#%%
#PLOT Income by Assisstance
money_order = ['Less than $5,000', '$5,000 - $7,499', '$7,500 - $9,999', '$10,000 - $12,499', '$12,500 - $14,999',
               '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999', '$30,000 - $34,999', '$35,000 - $39,999',
               '$40,000 - $49,999', '$50,000 - $59,999', '$60,000 - $74,999', '$75,000 - $99,999', '$100,000 - $149,999',
               '$150,000 or more']

assistance_order = ['Yes', 'No', 'NA']

plt.figure(figsize=(12, 6))

sns.countplot(data=RECs_dfs, x="MONEYPY", hue='COLDMA', palette="Set2", order=money_order, hue_order=assistance_order)
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='HOTMA', palette="Set2", order=money_order, hue_order=assistance_order)
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='NOACEL', palette="Set2", order=money_order, hue_order=assistance_order)
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='NOHEATEL', palette="Set2", order=money_order, hue_order=assistance_order)
sns.countplot(data=RECs_dfs, x="MONEYPY", hue='PAYHELP', palette="Set2", order=money_order, hue_order=assistance_order)

plt.xlabel("Income Level")
plt.ylabel("Count")
plt.title("Income by Assistance Level")
plt.xticks(rotation=30, ha='right')
plt.legend(title="Assistance Inputation", labels= assistance_order)
plt.show()

#%%
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

#%%

table = pd.crosstab(RECs_dfs['MONEYPY'], RECs_dfs['PAYHELP'])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p)

#%%

#!pip install statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm

data = RECs_dfs
data = data[data['PAYHELP'] != 'NA']

model = glm(formula='PAYHELP ~  HHAGE + C(MONEYPY) + C(HOTMA) + C(COLDMA) ', data=data, family=sm.families.Binomial())
model_fit = model.fit()
print(model_fit.summary())