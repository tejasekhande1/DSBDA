import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets

iris = datasets.load_iris()
print(iris);

df = pd.DataFrame(iris['data'])
print(df.head())

df[4] = iris['target']
print(df.head())

# Adding column names
df.rename(columns = {0:'SepalLengthCm', 1:'SepalWidthCm', 2:'PetalLengthCm', 3:'PetalWidthCm', 4:'Species'}, inplace = True)
print(df.head())

print(df.describe())

print(df.shape)

print(df.mean())

print(df.median())

# Calculated only for categorical data
print(df.Species.mode())

print(df.SepalLengthCm.std())
print(df.SepalWidthCm.std())
print(df.PetalLengthCm.std())
print(df.PetalWidthCm.std())
