import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#These are tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Load our data
df = pd.read_csv("c:\\Users\\tyagi\\Downloads\\house_data.csv")
print("Here are 5 house in data :")
print(df.head())
print("Column names and their types :")
print(df.dtypes)

#EDA
print("Basic statistics of our data :")
print(df.describe())

print("Missing Values :")
print(df.isnull().sum())

#Cleaning the data

df["selling_price"] = df["selling_price"].fillna(df["selling_price"].median())
print("Checking that our empty values are fixed")
print(df["selling_price"].isnull().sum())
print("All empty cells are now filled")

#Fix Outliers
plt.boxplot(df["selling_price"])
plt.title("Box plot for checking outliers")
plt.show()

Q1 = df["selling_price"].quantile(0.25)
Q3 = df["selling_price"].quantile(0.75)
IQR = Q3 -Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[(df["selling_price"]>=lower_limit) & (df["selling_price"]<= upper_limit)]
print(f"Data after cleaning : {df.shape[0]} rows , {df.shape[1]} columns")

#Encoding

df["area_encoded"] = df["area"].map({"Rural" : 0, "Urban" : 1})

#Choose Features and Target

X = df[["actual_making_cost", "age_of_house", "area_encoded", "no_of_rooms"]]
y = df["selling_price"]

#Split Data for Train and Test

X_train,X_Test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Scale

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_Test_scaled = scaler.transform(X_Test)

#Train the Model

model = LinearRegression()
model.fit(X_train_scaled,y_train)

print("Model Intercept :", model.intercept_)
print("Coef :",model.coef_)

#Make Prediction

predictions = model.predict(X_Test_scaled)

#Check the model (Metrics)

r2   = r2_score(y_test, predictions)
mae  = mean_absolute_error(y_test, predictions)
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("R2 Score :",r2)
print("MAE :",mae)
print("MSE :",mse)
print("RMSE :",rmse)

#Draw graphs
#Graph 1:
feature_names  = ["Making Cost", "Age", "Area", "Rooms"]
importance     = np.abs(model.coef_)
importance_pct = importance / importance.sum() * 100

plt.figure(figsize=(8, 5))
plt.bar(feature_names, importance_pct, color=["blue", "red", "green", "orange"])
plt.title("Which Feature Matters Most?")
plt.xlabel("Feature")
plt.ylabel("Importance (%)")
plt.show()

#Graph 2:

actual = y_test.values[:20]
pred   = predictions[:20]
houses = range(1, 21)

plt.figure(figsize=(12, 5))
plt.plot(houses, actual / 100000, marker="o", label="Actual Price",    color="blue")
plt.plot(houses, pred   / 100000, marker="s", label="Predicted Price", color="orange")
plt.title("Actual Price vs Predicted Price (20 Houses)")
plt.xlabel("House Number")
plt.ylabel("Price (in Lakhs)")
plt.legend()
plt.show()

#Graph 3:

errors = actual - pred

plt.figure(figsize=(12, 5))
plt.bar(houses, errors / 100000, color="green")
plt.axhline(0, color="red", linestyle="--", label="Zero Error (Perfect)")
plt.title("How Wrong Was Our Model? (Error per House)")
plt.xlabel("House Number")
plt.ylabel("Error (in Lakhs)")
plt.legend()
plt.show()

#Predict a New House Selling Price

new_house = pd.DataFrame({
    "actual_making_cost": [2000000], 
    "age_of_house":       [10],        
    "area_encoded":       [1],        
    "no_of_rooms":        [4],         
})

new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)[0]

print("Predicted Selling Price :",predicted_price)