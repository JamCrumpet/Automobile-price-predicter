# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sets how pandas prints dataframes
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("Automobile price data _Raw_.csv")
print(df.head())

feature_names = ["make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels",
                 "engine-location","wheel-base","length","width","height","curb-weight","engine-type",
                 "num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio",
                "horsepower","peak-rpm","city-mpg","highway-mpg"]
#feature_names = ["wheel-base","length","width","height","curb-weight","engine-size",
#                 "bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]

target_name = "price"

X = df[feature_names]
y = df[target_name]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression()

linear_regression_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
# Gen the prediction of the model for the data it has not seen (for testing)
y_pred_test = linear_regression_model.predict(X_test)
# All the metrics compare in some way how close are the predict vs the actual values
error_metric = mean_squared_error(y_pred_test, y_true=y_test)
print("Mean Square Error: ", error_metric)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_test)
ax.plot(y_test,y_test, color = "red")
ax.set_xlabel("Testing target values")
ax.set_ylabel("Predicted target values")
ax.set_title("Predicted vs. Actual Values")