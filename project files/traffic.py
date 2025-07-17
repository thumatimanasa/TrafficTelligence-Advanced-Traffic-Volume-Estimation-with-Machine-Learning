# importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import metrics
from collections import Counter
import pickle
import xgboost

#importing the dataset
data=pd.read_csv(r"C:\Users\Tumat\OneDrive\Desktop\traffic_volume\traffic_volume.csv") 

#analyze the data
data.head() 
data.describe()
data.info()

#handling missing values
data.isnull().sum()  #display null values of data

#filling the missing values
data['temp'] = data['temp'].fillna(data['temp'].mean())
data['rain'] = data['rain'].fillna(data['rain'].mean())
data['snow'] = data['snow'].fillna(data['snow'].mean())
data['weather'] = data['weather'].fillna('Clouds')



# Data Visualization
# correlation matrix
corr_matrix = data.select_dtypes(include=['float64', 'int64']).corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features Only)')
plt.show()


# Pair plot 
sns.pairplot(data)
plt.show()

# Box plot
plt.figure(figsize=(6, 4))
sns.boxplot(x=data['temp'])
plt.title('Box Plot of Temperature')
plt.show()

# Convert datetime
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['Time'], format='%d-%m-%Y %H:%M:%S')

data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['weekday'] = data['datetime'].dt.weekday

#splitting the dataset into independent and dependent variables
y = data['traffic_volume']
x = data.drop(columns=['traffic_volume', 'datetime'])  

#feature scaling
# Handle categoricals
x = pd.get_dummies(x)

# Scale the features using StandardScaler
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)


# split dataset into test and train
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_scaled, y, test_size=0.2, random_state=0)

#model building
#training and testing the model

lin_reg = linear_model.LinearRegression()
Dtree = tree.DecisionTreeRegressor()
Rand = ensemble.RandomForestRegressor()
svr = svm.SVR()
XGB = xgboost.XGBRegressor()

lin_reg.fit(x_train, y_train)
Dtree.fit(x_train, y_train)
Rand.fit(x_train, y_train)
svr.fit(x_train, y_train)
XGB.fit(x_train, y_train)

# Predict on training data
p1 = lin_reg.predict(x_train)
p2 = Dtree.predict(x_train)
p3 = Rand.predict(x_train)
p4 = svr.predict(x_train)
p5 = XGB.predict(x_train)

# model Evaluation
print("Linear R2:", metrics.r2_score(y_train, p1))
print("Decision Tree R2:", metrics.r2_score(y_train, p2))
print("Random Forest R2:", metrics.r2_score(y_train, p3))
print("SVR R2:", metrics.r2_score(y_train, p4))
print("XGBoost R2:", metrics.r2_score(y_train, p5))


# Predict on testing data
p1 = lin_reg.predict(x_test)
p2 = Dtree.predict(x_test)
p3 = Rand.predict(x_test)
p4 = svr.predict(x_test)
p5 = XGB.predict(x_test)

# Evaluation on testing data
print("Linear R2:", metrics.r2_score(y_test, p1))
print("Decision Tree R2:", metrics.r2_score(y_test, p2))
print("Random Forest R2:", metrics.r2_score(y_test, p3))
print("SVR R2:", metrics.r2_score(y_test, p4))
print("XGBoost R2:", metrics.r2_score(y_test, p5))

# RMSE
print("Random Forest RMSE:", np.sqrt(metrics.mean_squared_error(y_test, p3)))

# Save the Model

with open("model.pkl", "wb") as model_file:
    pickle.dump(Rand, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("columns.pkl", "wb") as columns_file:
    pickle.dump(x.columns.tolist(),columns_file)



