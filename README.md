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

![image](https://github.com/user-attachments/assets/b8493b37-d5ca-4fca-b42d-008fbdf63cd1)

df.tail()

![image](https://github.com/user-attachments/assets/38f21ff2-0b8f-4ad8-a604-76fa8c83c247)


# segregating data to variables
X=df.iloc[:,:-1].values
X

![image](https://github.com/user-attachments/assets/d113b1f9-da8d-4ec0-a9b9-c40a7098f16c)

Y=df.iloc[:,1].values
Y

![image](https://github.com/user-attachments/assets/ceffb91b-fe27-4af9-b3a1-8d05f87b7f68)


# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

![image](https://github.com/user-attachments/assets/bd6639d5-7d7d-46c6-b85c-1fff0b2032b4)

Y_test

![image](https://github.com/user-attachments/assets/b9737814-530f-4610-a1b4-a747acc81d14)


# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

![image](https://github.com/user-attachments/assets/959e2421-a4da-4a27-b852-b6373c94ad3b)

# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

![image](https://github.com/user-attachments/assets/6d238757-1741-4080-aad2-c1365bf421f6)


mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

![image](https://github.com/user-attachments/assets/2abacaf1-182c-476d-9bc3-e9b439ac5944)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

![image](https://github.com/user-attachments/assets/69c2fab7-a292-4f75-9238-8518cb3f1f5d)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

![image](https://github.com/user-attachments/assets/3d6fa33a-122a-4f83-85fd-284ec46c58dd)

```

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
