# %%
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
#os.chdir("C:\\Users\\lzou7\\OneDrive\\Desktop\\GWU\\Data Mining\\Final Project") #Linsie's directory
#os.chdir() #Sreya's directory
RECS_DF = pd.read_csv("recs2020_public_v5.csv")
code_book = pd.read_excel('RECS 2020 Codebook for Public File - v5.xlsx')

#Pulling the columns of interest. 

data_df = RECS_DF.iloc[:,:9]

column_names_list = ['TYPEHUQ', 'INTERNET', 'HHSEX', 'HHAGE', 'EMPLOYHH', 'EDUCATION',
                     'HOUSEHOLDER_RACE', 'COLDMA', 'HOTMA', 'NOACEL','NOHEATEL', 'MONEYPY','PAYHELP']

if all(col_name in RECS_DF.columns for col_name in column_names_list):
    data = RECS_DF[column_names_list]
    
RECs_dfs = pd.concat([data_df, data], axis=1)

#%%

#import data (smaller sized csv of our key attributes only - see github for the code I used)
#make sure to drag the csv file (in our drive) to the colab variable panel.
RECs_dfs = pd.read_csv("RECsample_df.csv")

'''
Initial corr plot
'''
sample_df_corr = RECs_dfs.iloc[:,10:].dropna()
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
#%%
data_rf = sample_df_corr  #.drop('UATYP10', axis =1)
data_rf = data_rf[data_rf['PAYHELP'] != -2]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = data_rf ['PAYHELP']
X = data_rf.drop('PAYHELP', axis=1)

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
sns.countplot(data=RECs_dfs, x ="MONEYPY",  palette="Set2", hue='EDUCATION', hue_order = edu_order, order = money_order)
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
from IPython.display import display, Latex

data = RECs_dfs
data = data[data['PAYHELP'] != 'NA']

model = glm(formula='PAYHELP ~  HHAGE + C(MONEYPY) + C(HOTMA) + C(COLDMA) ', data=data, family=sm.families.Binomial())
model_fit = model.fit()
print(model_fit.summary())

latex_output = model_fit.summary2().as_latex()
display(Latex(latex_output))


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

# Convert categorical 
selected_data['PAYHELP'] = pd.Categorical(selected_data['PAYHELP'], categories=[0, 1])

#convert to dummies
selected_data = pd.get_dummies(selected_data, columns=['PAYHELP'], prefix='PAYHELP', drop_first=True)
selected_data = pd.get_dummies(selected_data, columns=['EDUCATION'], prefix='EDU', drop_first=True)

# dependent variable 
y = selected_data['COLDMA']

# independent variables 
X = selected_data.drop('COLDMA', axis=1)

print("Shape before fitting the model:", X.shape)

# Fit the multilinear regression model
model = sm.OLS(y.astype(float), X.astype(float)).fit()

# Print the regression results
print(model.summary())
latex_output = model.summary().as_latex()
display(Latex(latex_output))

'''
Running a multilinear regression the X variables explains 3.6% of the variation in the COLDMA variable.
Compared to the Linear Regression conducted previously this model is better.
However, the r-squared value 3.6% is still very small.
Nonetheless, the multilinear model shows when people seek medical attention when the home is too cold. 
they are also unable to use other temperature systems like AC and heating equipment.
Some people who sought medical attention because the home was too cold also received assistance to pay
energy bills. This finding is concerning as it indicates welfare assistance may not do enough to reduce
mortality risks from experiencing energy poverty.
'''

#%%
'''
Logistic regression - for HOTMA and COLDMA
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Selecting relevant columns and dropping NAs
selected_columns = ['EDUCATION', 'HOUSEHOLDER_RACE', 'NOACEL', 'MONEYPY', 'COLDMA', 'HOTMA']

selected_data = RECS_DF[selected_columns]

selected_data = selected_data.dropna()

# setting (X) and (y)
X = selected_data.drop(['COLDMA', 'HOTMA'], axis=1)  
y_coldma = selected_data['COLDMA']  
y_hotma = selected_data['HOTMA']  

# Splitting the dataset into training and testing 
X_train_coldma, X_test_coldma, y_train_coldma, y_test_coldma = train_test_split(X, y_coldma, test_size=0.2, random_state=42)

logreg_model_coldma = LogisticRegression()

# Fitting the model for training data from COLDMA
logreg_model_coldma.fit(X_train_coldma, y_train_coldma)

# Making test predictions for COLDMA
y_pred_coldma = logreg_model_coldma.predict(X_test_coldma)

# Evaluating the COLDMA model
accuracy_coldma = accuracy_score(y_test_coldma, y_pred_coldma)
conf_matrix_coldma = confusion_matrix(y_test_coldma, y_pred_coldma)
classification_rep_coldma = classification_report(y_test_coldma, y_pred_coldma)

# COLDMA results
print("Results for COLDMA:")
print(f"Accuracy: {accuracy_coldma}")
print("Confusion Matrix:")
print(conf_matrix_coldma)
print("Classification Report:")
print(classification_rep_coldma)

'''
The confusion matrix shows that the model does not detect any true positives. 
Instead our model has high scores in precision and recall for identifying negatives. 
The model is accurate in detecting our 3.6k true negatives, which increases the model acuracy. 
However, our model does not detecting any true positive values, which showcase its failure. 
In fact, our model over-fitted at a 99.4% accuracy so its inflating the false positives higher than actual true positives.
'''

#%%
# Split the dataset into training and testing for HOTMA
X_train_hotma, X_test_hotma, y_train_hotma, y_test_hotma = train_test_split(X, y_hotma, test_size=0.2, random_state=42)

logreg_model_hotma = LogisticRegression()

# Fitting the model on the training data for HOTMA
logreg_model_hotma.fit(X_train_hotma, y_train_hotma)

# Making predictions on the test data for HOTMA
y_pred_hotma = logreg_model_hotma.predict(X_test_hotma)

# Evaluating the model for HOTMA
accuracy_hotma = accuracy_score(y_test_hotma, y_pred_hotma)
conf_matrix_hotma = confusion_matrix(y_test_hotma, y_pred_hotma)
classification_rep_hotma = classification_report(y_test_hotma, y_pred_hotma)

# HOTMA results 
print("\nResults for HOTMA:")
print(f"Accuracy: {accuracy_hotma}")
print("Confusion Matrix:")
print(conf_matrix_hotma)
print("Classification Report:")
print(classification_rep_hotma)

'''
The confusion matrix for 'HOTMA' shows the same outcome as the confusion matrix for 'COLDMA'.
The model does not detect any true positives. 
Instead our model has high scores in precision and recall for identifying negatives. 
It accurately detects our 3.6k true negatives, increasing the model accuracy. 
However, since the model does not detecting any true positive values it fails in this aspect. 
This model for 'HOTMA' like the model for 'COLDMA' is over-fitted and it inflates the false positives 
higher than actual true positives.
'''

#%%
'''
SVM
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Selecting columns
selected_features_svm = [
    'MONEYPY',
    'PAYHELP',
    'EDUCATION',
    'EMPLOYHH',
    'TYPEHUQ',
    'NOACEL',
    'NOHEATEL'
]

# Create a subset of the DataFrame with selected features for SVM
selected_data_svm = RECS_DF[selected_features_svm]

# Convert categorical variables to dummy variables
selected_data_svm = pd.get_dummies(selected_data_svm, columns=['EMPLOYHH', 'TYPEHUQ'])

# Drop rows with missing values
selected_data_svm = selected_data_svm.dropna()
selected_data_svm = selected_data_svm[selected_data_svm['PAYHELP'] != -2]

# set (X_svm) and (y_svm) 
X_svm = selected_data_svm.drop(['PAYHELP'], axis=1)  
y_svm = selected_data_svm['PAYHELP']  

# Split the data into training and testing 
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

# Standardize the features (important for SVM)
scaler_svm = StandardScaler()
X_train_scaled_svm = scaler_svm.fit_transform(X_train_svm)
X_test_scaled_svm = scaler_svm.transform(X_test_svm)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model and predict
svm_model.fit(X_train_scaled_svm, y_train_svm)

y_pred_svm = svm_model.predict(X_test_scaled_svm)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
classification_rep_svm = classification_report(y_test_svm, y_pred_svm)

print('SVM Results:')
print(f'Accuracy: {accuracy_svm}')
print('Classification Report:')
print(classification_rep_svm)

'''
The model performs well for predicting Class 0, with high detection of negatives of PAYHELP status. 
But even with the support vector, it is unable to classify Class 1, 
or true positives, which informs us that there is not enough Class 1 responses in the dataset to infer meaningful predictions.
'''

#%%
'''
KNN-Payhelp
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select relevant columns for KNN
selected_features_knn = [
    'MONEYPY',
    'EDUCATION',
    'EMPLOYHH',
    'TYPEHUQ',
    'NOACEL',
    'NOHEATEL'
]

selected_data_knn = RECS_DF[selected_features_knn + ['PAYHELP']]

# set dummies
selected_data_knn = pd.get_dummies(selected_data_knn, columns=['EMPLOYHH', 'TYPEHUQ'])
selected_data_knn = selected_data_knn[selected_data_knn['PAYHELP'] != -2]
selected_data_knn = selected_data_knn.dropna()

# Setting y and x
X_knn = selected_data_knn.drop(['PAYHELP'], axis=1)  
y_knn = selected_data_knn['PAYHELP'] 

# Split the data 
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

# Standardize the Xs
scaler_knn = StandardScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train_knn)
X_test_scaled_knn = scaler_knn.transform(X_test_knn)

knn_model = KNeighborsClassifier(n_neighbors=5)

# Training
knn_model.fit(X_train_scaled_knn, y_train_knn)

# Test predictions
y_pred_knn = knn_model.predict(X_test_scaled_knn)

# Evaluates
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
classification_rep_knn = classification_report(y_test_knn, y_pred_knn)

# Print the KNN results
print('KNN Results:')
print(f'Accuracy: {accuracy_knn}')
print('Classification Report:')
print(classification_rep_knn)

'''
The KNN Model for the 'PAYHELP' variable has an accuracy of 73%.
In particular the model performs well for Class 0 in all 4 evaluation metrics.
KNN provides higher results than the other models for detecting false positives, but not for false negatives.
It does not perform well in detecting false positives and false negatives.
'''
#%%
'''
KNN-Coldma Hotma
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select relevant columns for KNN
selected_features_knn = [
    'MONEYPY',
    'EDUCATION',
    'EMPLOYHH',
    'TYPEHUQ',
    'NOACEL',
    'NOHEATEL'
]

selected_data_knn = RECS_DF[selected_features_knn + ['COLDMA', 'HOTMA']]

# Convert dummy variables
selected_data_knn = pd.get_dummies(selected_data_knn, columns=['EMPLOYHH', 'TYPEHUQ'])

selected_data_knn = selected_data_knn.dropna()

X_knn = selected_data_knn.drop(['COLDMA', 'HOTMA'], axis=1)  
y_coldma = selected_data_knn['COLDMA']  
y_hotma = selected_data_knn['HOTMA']  

# Split the data 
X_train_knn, X_test_knn, y_train_coldma, y_test_coldma = train_test_split(X_knn, y_coldma, test_size=0.2, random_state=42)
_, _, y_train_hotma, y_test_hotma = train_test_split(X_knn, y_hotma, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler_knn = StandardScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train_knn)
X_test_scaled_knn = scaler_knn.transform(X_test_knn)

knn_model_coldma = KNeighborsClassifier(n_neighbors=5)
knn_model_hotma = KNeighborsClassifier(n_neighbors=5)

# Training
knn_model_coldma.fit(X_train_scaled_knn, y_train_coldma)
knn_model_hotma.fit(X_train_scaled_knn, y_train_hotma)

# test predictions
y_pred_coldma = knn_model_coldma.predict(X_test_scaled_knn)
y_pred_hotma = knn_model_hotma.predict(X_test_scaled_knn)

# Evaluate the KNN models for COLDMA and HOTMA
accuracy_coldma = accuracy_score(y_test_coldma, y_pred_coldma)
classification_rep_coldma = classification_report(y_test_coldma, y_pred_coldma)

accuracy_hotma = accuracy_score(y_test_hotma, y_pred_hotma)
classification_rep_hotma = classification_report(y_test_hotma, y_pred_hotma)

# Print the KNN results for COLDMA
print('KNN Results for COLDMA:')
print(f'Accuracy: {accuracy_coldma}')
print('Classification Report:')
print(classification_rep_coldma)

# Print the KNN results for HOTMA
print('\nKNN Results for HOTMA:')
print(f'Accuracy: {accuracy_hotma}')
print('Classification Report:')
print(classification_rep_hotma)
'''
The KNN models for COLDMA and HOTMA variables both exhibit high accuracy of 99.46%. 
However, the models perform well only for Class 0, with precision and recall of 99% and 100%, respectively. 
For Class 1, the precision, recall, and F1-score are all 0%, indicating the model struggles to predict this class. 
The high accuracy is primarily driven by correct predictions for Class 0, while Class 1 predictions are ineffective.
'''

#%%
'''
Gradient Boosting 
'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

selected_columns = [
    'TYPEHUQ', 'INTERNET', 'HHSEX', 'HHAGE', 'EMPLOYHH', 'EDUCATION', 'HOUSEHOLDER_RACE',
    'MONEYPY', 'PAYHELP', 'COLDMA', 'HOTMA', 'NOACEL', 'NOHEATEL'
]

selected_data = RECS_DF[selected_columns]

# Convert to dummies
selected_data = pd.get_dummies(selected_data, columns=['TYPEHUQ', 'EMPLOYHH', 'EDUCATION', 'HOUSEHOLDER_RACE'], drop_first=True)

# Drop rows with missing values
selected_data = selected_data.dropna()
selected_data = selected_data[selected_data['PAYHELP'] != -2]

# Split data into features (X) and target variables (y)
X = selected_data.drop(['COLDMA', 'HOTMA'], axis=1)
y_cold = selected_data['COLDMA']
y_hot = selected_data['HOTMA']

# Split the data into training and testing sets
X_train, X_test, y_train_cold, y_test_cold, y_train_hot, y_test_hot = train_test_split(
    X, y_cold, y_hot, test_size=0.2, random_state=42
)

# Initialize Gradient Boosting models
cold_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
hot_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the models
cold_model.fit(X_train, y_train_cold)
hot_model.fit(X_train, y_train_hot)

# Make predictions
cold_predictions = cold_model.predict(X_test)
hot_predictions = hot_model.predict(X_test)

# Evaluate model performance
cold_accuracy = accuracy_score(y_test_cold, cold_predictions)
hot_accuracy = accuracy_score(y_test_hot, hot_predictions)

print("Cold Model Accuracy:", cold_accuracy)
print("Hot Model Accuracy:", hot_accuracy)

# Additional evaluation metrics
print("\nCold Model Classification Report:")
print(classification_report(y_test_cold, cold_predictions))

print("\nHot Model Classification Report:")
print(classification_report(y_test_hot, hot_predictions))

'''
Both the Cold Model and Hot Model accurately predict negative outcomes more than positive outcomes.
This pulls on the weighted averages and shows both models are overfitted.
The macro average scores for both model is a better representation of the model values compared to the
weighted average scores.
'''

#%%
'''
Gradient Boosting - ONLY USING : employ,money and type of house

'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

selected_columns = ['EMPLOYHH', 'MONEYPY', 'TYPEHUQ', 'COLDMA', 'HOTMA']

selected_data = RECS_DF[selected_columns]

# Convert to dummies
selected_data = pd.get_dummies(selected_data, columns=['TYPEHUQ'], drop_first=True)

selected_data = selected_data.dropna()

# Split data into X and Y
X = selected_data.drop(['COLDMA', 'HOTMA'], axis=1)
y_cold = selected_data['COLDMA']
y_hot = selected_data['HOTMA']

# Splitting into training and testing
X_train, X_test, y_train_cold, y_test_cold, y_train_hot, y_test_hot = train_test_split(
    X, y_cold, y_hot, test_size=0.2, random_state=42
)

cold_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
hot_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

cold_model.fit(X_train, y_train_cold)
hot_model.fit(X_train, y_train_hot)

cold_predictions = cold_model.predict(X_test)
hot_predictions = hot_model.predict(X_test)

# Evaluate model performance
cold_accuracy = accuracy_score(y_test_cold, cold_predictions)
hot_accuracy = accuracy_score(y_test_hot, hot_predictions)

print("Cold Model Accuracy:", cold_accuracy)
print("Hot Model Accuracy:", hot_accuracy)

# Additional evaluation metrics
print("\nCold Model Classification Report:")
print(classification_report(y_test_cold, cold_predictions))

print("\nHot Model Classification Report:")
print(classification_report(y_test_hot, hot_predictions))

'''
Both the Cold Model and Hot Model accurately predict negative outcomes more accurately than positive outcomes.
This pulls on the weighted averages of the Gradient Boosting classifier and shows both models are overfitted.
The macro average scores for both model is again a better representation of the model values compared to the
weighted average scores.
'''
