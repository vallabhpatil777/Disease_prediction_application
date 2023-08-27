#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.svm import SVR
from matplotlib.colors import ListedColormap

import joblib

test=pd.read_csv(r"C:\Users\MSI\Downloads\archive (7)\Testing.csv") 
train=pd.read_csv(r"C:\Users\MSI\Downloads\archive (7)\Training.csv")

data = pd.concat([train, test])
data.head(5)
data.tail(5)
data.columns
data.shape
data.info()
data.isnull().sum()

#Importing the train_test_split functionality 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data["prognosis"]


le = LabelEncoder()
data["prognosis"] = le.fit_transform(data["prognosis"])
data["prognosis"]

X, y=data.drop("prognosis",axis=1), data[["prognosis"]]

#Spliting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

# Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb=gnb.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
dct =  DecisionTreeClassifier()
dct=dct.fit(X_train,y_train)


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



joblib.dump(gnb,"model.joblib")


joblib.dump(dct,"Decisiontree.joblib")

