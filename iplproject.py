# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 01:40:43 2018
@author: adars
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv('datafile (1).csv')
x=dataset.iloc[:,0:5].values
y=dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label=LabelEncoder()
x[:, 0]=label.fit_transform(x[:, 0])
label1=LabelEncoder()
x[:, 1]=label1.fit_transform(x[:, 1])
ht=OneHotEncoder(categorical_features=[0,1])
x=ht.fit_transform(x).toarray()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
polyreg=PolynomialFeatures(degree=2)
x_poly=polyreg.fit_transform(x_train)
polyreg.fit(x_poly,y_train)
lin1=LinearRegression()
lin1.fit(x_poly,y_train)
x_test=polyreg.fit_transform(x_test)
y_pred=lin1.predict(x_test)
from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test,y_pred)
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(x_train,y_train)
y_pred1=reg.predict(x_test)