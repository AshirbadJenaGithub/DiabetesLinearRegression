from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
diabetes=load_diabetes()
data=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
x=data.drop('age',axis=1)
y=data['age']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=lr()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
evaluate=mean_absolute_error(prediction,y_test)
plt.scatter(prediction,y_test)
