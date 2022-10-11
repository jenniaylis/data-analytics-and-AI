# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('housing.csv')

df.plot(kind='scatter', x='median_income', y='median_house_value')
plt.show()

X = df.iloc[:, [7]]
y = df.iloc[:, [8]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

residuals = y_pred - y_test
plt.hist(residuals)
plt.show()

plt.scatter(X,y)
plt.plot(X_test, y_pred, color='red')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nMetriikka test datalla:')
print (f'mae: {mae}')
print (f'mse: {mse}')
print (f'rmse: {rmse}')
print(f'R2: {r2}') 

new_house = pd.DataFrame(data={'median_income':[3]})
print (f'Kotitalouden talon arvo on: {regr.predict(new_house)}')

# %%
