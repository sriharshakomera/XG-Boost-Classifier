# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:50:48 2019

@author: Sriharsha Komera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data set
path='F:\\Krish\\XG Boost\\Xgboost-master\\Churn_Modelling.csv'
dataset=pd.read_csv(path)

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encode the ctegorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#splitting the dataset into train test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#fitting the Xgboost to the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

 #predict the test set
 y_pred=classifier.predict(X_test)
 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
accuracy=accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)
 