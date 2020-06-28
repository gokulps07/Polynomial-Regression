# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:10:45 2020

@author: PSG
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
#Reading data 
data=pd.read_csv('data1.csv')
#Seperation of dependent and independent Variable
X=data.iloc[:,0:1]
y=data.iloc[:,1]
#Creating PolyFeature for the Independent Variable
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(3)
X_new=poly.fit_transform(X)
#poly.fit(X_new,y)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_new,y)
#Plotting of Graph
mpl.scatter(X,y,color='blue')
mpl.title("Polynomial Model (Red-Model Created Blue-Input from dataset)")
mpl.plot(X,reg.predict(poly.fit_transform(X)),color='red')


