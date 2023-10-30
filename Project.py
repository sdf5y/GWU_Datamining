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

#%%
'''
Pulling the columns of interest. 
'''

data_df = RECS_DF.iloc[:,:9]

column_names_list = ['TYPEHUQ', 'INTERNET', 'HHSEX', 'HHAGE', 'EMPLOYHH', 'EDUCATION',
                     'HOUSEHOLDER_RACE', 'COLDMA', 'HOTMA', 'NOACEL','NOHEATEL', 'MONEYPY','PAYHELP']

if all(col_name in RECS_DF.columns for col_name in column_names_list):
    data = RECS_DF[column_names_list]
    
RECs_dfs = pd.concat([data_df, data])

#%% 
'''
Initial corr plot
'''

sample_df_corr = RECs_dfs.iloc[:,9:]
sns.heatmap(sample_df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")

#%%

#RECs_dfs.to_csv('RECsample_df.csv')

