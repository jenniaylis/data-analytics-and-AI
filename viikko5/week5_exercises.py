# %% Exercise 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# read data from csv
df = pd.read_csv('salary.csv')

X = df.loc[:, ['YearsExperience']]
y = df.loc[:, ['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'r2: {r2}')
print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.show()

print(f'Uuden työntekijän palkka 7v kokemuksella on {model.predict([[7]])}')
# %%
