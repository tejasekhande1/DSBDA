import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('BostonHousing.csv')

print(df.head())
print(df.tail())
print(df.info())
print(df.isna())

data= df[['rm', 'lstat', 'medv']]

print(data.head())
sns.boxplot(x=df['rm'])
sns.boxplot(x=df['lstat'])
sns.scatterplot(data=df, x="rm", y="medv")
sns.scatterplot(data=df, x="lstat", y="medv")
plt.show()

def Remove_outlier(df, var):
    Q1=df[var].quantile(0.25)
    Q3=df[var].quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df[var]<(Q1-1.5*IQR)) | (df[var]>(Q3+1.5*IQR)))]
    return df_final

sns.heatmap(df.corr(), annot=True)
fig, ax=plt.subplots(figsize=(16,8))
sns.heatmap(df.corr(), annot=True)
plt.show()
data = df[['rm', 'lstat', 'medv']]
print(data.head())
x=df[['rm','lstat']]
y=df['medv']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train=train_test_split(x,y,test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
output = model.predict(x_test)
print(output)

print(y_test)
def LinearRegressionModel(x_train,y_train,x_test,y_test):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x_train,y_train)
    output = model.predict(x_test)

#from sklearn.metrics import mean_absolute_errorprint("MAE: ", mean_absolute_error(y_test, output))
 #   print("MAE: ", mean_absolute_error(y_test, output))
  #  print("Model Score: ", model.score(x_test,y_test))

