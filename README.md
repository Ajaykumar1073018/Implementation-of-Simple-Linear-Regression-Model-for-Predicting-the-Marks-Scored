# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ajay kumar T
RegisterNumber:  212223047001
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
DATASET:
![image](https://github.com/user-attachments/assets/58cebe7e-d735-4b3f-98cc-f88b2ee4b369)
HEAD VALUES :
![image](https://github.com/user-attachments/assets/b74cae84-1ec3-4e6d-bb53-ccd9d1e9f992)
TAIL VALUES:
![image](https://github.com/user-attachments/assets/c3a63246-f2d3-4cad-a19b-a5a3e084e4ed)
X AND Y VALUES:
![image](https://github.com/user-attachments/assets/0daf0bce-ac53-4413-b20b-544b3e16aa50)
PREDICTION VALUES OF X AND Y :
![image](https://github.com/user-attachments/assets/48b018e7-5d51-4762-84cf-abad69700135)
MSE,MAE,RMSE VALUES :
![image](https://github.com/user-attachments/assets/c84a5aff-f295-4f86-8eb8-64a5caf52849)
TRAINING SET :
![image](https://github.com/user-attachments/assets/b55abe0a-13a2-426a-b48d-5cc9024efc0c)
TESTING SET:
![image](https://github.com/user-attachments/assets/11540108-1f99-41ec-b599-2241ce4ceb20)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
