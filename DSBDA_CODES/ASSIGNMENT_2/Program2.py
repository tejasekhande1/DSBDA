def DetectOutlier(df,var):
	Q1 = df[var].quantile(0.25)
	Q3 = df[var].quantile(0.75)
	IQR = Q3 - Q1
	high, low = Q3 + 1.5 * IQR, Q1-1.5*IQR
	
print("Highest allowed in variable :", var, high)
print("lowest allowed in variable:", var, low)
	
count= df[(df[var]>high) | (df[var]<low)][var].count()
	
print("Total outliers in:",var,':', count)
	
df=df[((df[var]>=low) & (df[var] <= high))]
	
    print('Outlier removed in',var)
    return df

import pandas as pd
df = pd.read_csv("tecdiv.csv")
print("Student Academic Performance Dataset is successfully loaded :")
    
import seaborn as sns
import matplotlib.pyplot as plt
    
choice = 1
while(choice != 10):
    print('------------------Menu---------------')
    print('1.Display information of dataset')
    print('2.Display Statistical information of Numerical Columns')
    print('3.Find and Fill the missing values')
    print('4.4Detect Outlier')
    print('5.Data transformation: conversion of categorical toquantatitive')
    print('6.Boxplot with 2 variables (gender and raisedhands)')
    print('7.Boxplot with 3 variables (gender, nationality, discussion)')
    print('8.Scatterplot to see relation between (raisedhands,VisitedResources)')
    print('10.Exit')

    choice = int(input('enter your choice:'))
    if choice == 1:
        print("Information of dataset:\n", df.info)
        print('Shape of dataset (row * column):', df.shape)
        print('Columns name:', df.columns)
        print('Total emlements in dataset:', df.size)
        print('Datatype of attributes (columns):', df.dtypes)
        print('First 5 rows:\n', df.head().T)
        print('last 5 rows:\n', df.tail().T)
        print('Any 5 rows:\n', df.sample(5).T)

    if choice == 2:
        print('Statistical information of numericalcolumns:\n',df.describe())

    if choice == 3:
        print('Total number of Null values in dataset:',df.isna().sum())

    if choice == 4:
        numcolumns = ['raisedhands', 'VisITedResources','AnnouncementsView', 'Discussion']
        fig, axes =plt.subplots(2,2)
        fig.suptitle('Before removing outliers')
        sns.boxplot(data=df, x='raisedhands', ax=axes[0,0])
        sns.boxplot(data=df, x='VisITedResources', ax=axes[0, 1])
        sns.boxplot(data=df, x='AnnouncementsView', ax=axes[1,0])
        sns.boxplot(data=df, x='Discussion', ax=axes[1,1])
        fig.tight_layout()
        plt.show()

    if choice == 5:
        df['gender']=df['gender'].astype('category')
        print('Data types of Gender=', df.dtypes['gender'])
        df['gender']=df['gender'].cat.codes
        print('Data types od gender after label encoding=',
        df.dtypes['gender'])
        print('Gender Values:',df['gender'].unique())
        
    if choice == 6:
        sns.boxplot(data=df, x='raisedhands', y='gender', hue='gender')
        plt.title('Boxplot with 2 variables gender and raisedhands')
        plt.show()
        
    if choice == 7:
        sns.boxplot(data=df, x='Discussion', y='NationalITy', hue ='gender')
        plt.title("Boxplot with 3 variables gender, NationalITy and discussion")
        plt.show()

    if choice == 8:
        sns.boxplot(data=df, x='raisedhands', y='VisITedResources')
        plt.title('Scatterplot for raisedhands, VisITedResources')
        plt.show()
        
    if choice == 10:
        break
