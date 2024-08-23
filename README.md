# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import the standard Libraries.

3.Set variables for assigning dataset values.

4.Import linear regression from sklearn.

5.Assign the points for representing in the graph.

6.Predict the regression for marks by using the representation of the graph.

7.Compare the graphs and hence we obtained the linear regression for the given datas.

8.Stop te program.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Preethi S
RegisterNumber:  212223230157

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()

df.tail()

# segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

Y_test

# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:

![image](https://github.com/user-attachments/assets/15fd9d29-73e8-4e7c-a29a-b5918d893932)

![image](https://github.com/user-attachments/assets/3320a129-37da-419b-b128-d0343c7e18fd)

![image](https://github.com/user-attachments/assets/c727fa2f-3db8-4083-9ee7-912bba92f44d)

![image](https://github.com/user-attachments/assets/f854fcac-b6d7-44f3-8b4b-4662113935b3)

![image](https://github.com/user-attachments/assets/461632b9-0cab-4f5a-a7a5-35899b1e928a)

![image](https://github.com/user-attachments/assets/afb53d67-56f5-42ac-8b05-98be12bd2ff2)

![image](https://github.com/user-attachments/assets/4b75f721-35fe-4e9f-95e5-334544efe225)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
