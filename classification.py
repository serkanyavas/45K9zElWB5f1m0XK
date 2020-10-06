#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn.model_selection import GridSearchCV

#Data Load
def import_data():
    df = pd.read_csv('data.csv')
    return df

#Label encoding
def encode(df):
    label_cols = ['y', 'default', 'housing', 'loan']
    for col in label_cols:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.codes
    obj_cols = df.select_dtypes(include=['object'])
    num_cols =df.select_dtypes(exclude=['object'])
    obj = pd.get_dummies(obj_cols, columns=obj_cols.columns)
    df = pd.concat([obj, num_cols], axis=1,sort=True)
    return df

#Outlier Deteciton
def out_det(df): 
    for column in df.columns[:]:
        if df[column].dtypes == 'int64' or df[column].dtypes == 'float64' :
            col=df[column]
            std=col.std()
            avg=col.mean()
            three_sigma_plus = avg + (3 * std)
            three_sigma_minus =  avg - (3 * std)
            outliers = col[((df[column] > three_sigma_plus) | (df[column] < three_sigma_minus))].index
            print(column, outliers)
        else:
            pass
    df.drop(index=outliers, inplace=True)
    return df

#Feature Engineering
def loan_merge(df):
    df = df.assign(debt = df['housing']+df['loan']) 
    drop_list = ['housing','loan']
    df = df.drop(drop_list,axis = 1 )  
    return df

#Data Visualization
def visual(df):
    df.plot(kind="scatter", x="duration", y="campaign",
                  s=df["y"]*30, label="y", figsize=(10,7), colorbar=False,
                  )
    plt.title('Relation between campain, duration and y')
    plt.legend()
    
    count = ["age","job","marital","education","default","contact","month","y"]
    for i in count:
        plt.figure(figsize=(8, 4))
        sns.countplot(i, data=df)
        plt.xticks(rotation=90)
        plt.title('Distribution of Columns')
    plt.show()
#looking for Correlations
def corcheck(df):
    cor=df.corr()
    a = cor["y"].sort_values(ascending=False)
    return print("{} \n\n BONUS 2::::: As we can see most important feature is duration. But campaign num must be low.".format(a))
#Splittin data to Train and Test
def split_dataandmodel(df):
    x = df.drop('y', axis = 1)
    y = df['y'].copy()
    est = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=14, criterion='gini')
    return(x,y,est)
    
#Determine best params for model
def gri_search(est, x, y) :
    param_grid = [
            {'n_estimators': [100,200, 400], 
             'criterion':['gini', 'entropy'],
             'max_depth' : [4,8]},
            ]
    grc = GridSearchCV(estimator=est, param_grid=param_grid, cv= 5)
    grc.fit(x, y)
    return print("{}".format(grc.best_params_))
    
#K-Fold Validation
def kfold(x, y,est):
    cv = RepeatedKFold(n_splits=5,  random_state=14)
    scores = cross_val_score(est, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return print('Accuracy: %'  ,(mean(scores))*100)

def bonusone(df):
    return df.to_csv('imported.csv')  
    
if __name__ == '__main__':
    df = import_data()
    df_1 = encode(df)
    df_2 = out_det(df_1) 
    df_3 = loan_merge(df_2) 
    visual(df)
    corcheck(df_3)
    x,y,est = split_dataandmodel(df_3)  
    ### Worked for once and determined best parametres
    #gri_search(est, x, y)
    kfold(x, y,est)
    bonusone(df_3)