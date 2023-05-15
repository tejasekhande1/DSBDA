import numpy as np
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
print(data.head())

print(data.tail())

print("The shape of the data is: ")
print(data.shape)

print(data.isnull().sum())

X = data.iloc[:,0:13]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
print(model.fit(X_train, y_train))

print(model.score(X_test,y_test))
