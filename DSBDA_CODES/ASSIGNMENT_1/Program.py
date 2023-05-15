import pandas as pd
import numpy as np

df=pd.read_csv('autodata.csv')
print("Student Performance dataset is loaded :")
choice = 1

while(choice != 10):
    print('---------------Menu------------------')
    print('1.Display information of dataset')
    print('2.Display Statistical information of Numerical columns')
    print('3.Find missing values')
    print('4.Change Datatype of columns')
    print('5.Conversion of categorical to quantitative')
    print('6.Normalisation using min-max scaling')
    print('7.Reload the dataset')
    print('10.Exit')
    
    choice=int(input('Enter your choice:'))

    if choice == 1:
        print('Information of dataset:\n',df.info)
        print('Shape of dataset (row * column):',df.shape)
        print('Columns name:',df.columns)
        print('Total emlements in dataset:',df.size)
        print('Datatype of attributes (columns):',df.dtypes)
        print('First 5 rows:\n',df.head().T)
        print('last 5 rows:\n',df.tail().T)
        print('Any 5 rows:\n',df.sample(5).T)

    if choice == 2:
        print('Statistical information of numerical columns:\n',df.describe())

    if choice == 3:
        print('Total number of Null values in dataset:',df.isna().sum())

    if choice == 4:
        df['sl_no']=df['sl_no'].astype('int8')
        print('Check Datatype of sl_no',df.dtypes)
        df['ssc_p']=df['ssc_p'].astype('int8')
        print('Check Datatype of ssc_p',df.dtypes)

    if choice ==5:
        choice1 = 'a'
        while(choice1!='e'):
            print('a.Find and Replace Method ')
            print('b.Label Encoding Using cat.codes')
            print('c.One Hot Encoding')
            print('d.Using Scikit-Learn Library')
            print('e.Go back')
            choice1=input('Enter your choice for conversion of Categorical to Quantitative:')
            df=pd.read_csv('Placement_Data_Full_Class.csv')
            print('Placement dataset is again successfully loaded into dataframe')
            print(df.head().T)

            if choice1 == 'a':
                df['gender']=df['gender'].astype('category')
                print('Data types of Gender=', df.dtypes['gender'])
                df['gender']=df['gender'].cat.codes
                print('Data types of gender after label encoding=',df.dtypes['gender'])
                print('Gender Values :', df['gender'].unique())
            
            if choice1 == 'c':
                df=pd.get_dummies(df,columns=['gender'],prefix='sex')
                print(df.head().T)
            
  #          if choice1 == 'd':
  #              from sklearn.preprocessing import OrdinalEncoderenc = OrdinalEncoder()
   #             df[['gender']]=enc.fit_transform(df[['gender']])
    #            print(df.head().T)
            
            if choice1 == 'e':
                break

    if choice == 6:
        choice2 ='a'
        while(choice2 !='f'):
            print('a.Maximum Absolute Scaling')
            print('b.Min-Max Feature Scaling')
            print('c.Z-Score method')
            print('d.Robust Scaling')
            print('e.Using Sci-kit learn')
            print('f.Go back')
            choice2=input('Enter your choice for normalization:')
            df=pd.read_csv('Placement_Data_Full_Class.csv')
            print('Placement dataset is again successfully loaded into Dataframe-------')
            
            print(df.head().T)

            if choice2 == 'a':
                df['salary']=df['salary']/df['salary'].abs().max()
                print(df.head().T)
            if choice2 == 'b':
                df['salary']=(df['salary']-df['salary'].min()/(df['salary'].max()-
                df['salary'].min()))
                print(df.head().T)
            if choice2 == 'c':
                df['salary'] = (df['salary'] - df['salary'].mean()) /(df['salary'].std())
                print('\n z score is \n\n')
                print(df['salary'].head().T)
            if choice2 == 'd':
                df['salary'] = (df['salary'] - df['salary'].mean())/(df['salary'].quantile(0.75) - df['salary'].quantile())
                print(df['salary'].head().T)
            if choice2 == 'e':
                from sklearn.preprocessing import MaxAbsScalerabs_scaler=MaxAbsScaler()
                df[['salary']]=abs_scaler.fit_transform(df[['salary']])
                print('\n Maximum absolute Scaling method normalization -1 to 1 \n\n')
                print(df['salary'].head().T)
            if choice2=='f':
                break
    
    if choice == 7:
        df=pd.read_csv('autodata.csv')
        print('Placement dataset is again successfully loaded into Dataframe-------')
        print(df.head().T)

    if choice == 10:
        break

