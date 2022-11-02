# %% Exercise 1
import pandas as pd
import matplotlib.pylab as plt

# creating dataframe
X = [1, 2, 3, 4, 6, 7, 8]
y = [5, 7, 9, 11, 15, 17, 19]
df = pd.DataFrame(X, columns=['x'])
df['y'] = y
print(df.head())

# visualizing
plt.scatter(X, y, color='blue')
plt.plot(X, y, color='black')

#####################################################################

# %% Exercise 2
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt

# creating dataframe
x = [1, 2, 3, 4, 6, 7, 8]
y = [5, 7, 9, 11, 15, 17, 19]
df = pd.DataFrame(x, columns=['x'])
df['y'] = y

x = df.loc[:, ['x']]
y = df.loc[:, ['y']]

model = LinearRegression()
model.fit(x, y)

# predicting x = 5 and visualizing it
y_pred = model.predict([[5]])
plt.scatter(x, y)
plt.scatter(5, y_pred, color ='red')
plt.plot(x, y)
plt.show()

# equation for the model
coef = model.coef_
inter = model.intercept_
print ('Suoran yhtälö on: ')
print (f'y = {round(coef[0][0],2)} * x + {round(inter[0],2)}')


#####################################################################
# %% Exercise 3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# never do this again:
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# never do that again!

# read data from csv
df = pd.read_csv('salary.csv')

X = df.loc[:, ['YearsExperience']]
y = df.loc[:, ['Salary']]

# visualizing data with scatterplot
print('scatter plot:')
plt.scatter(X, y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print('heatmap')
sns.heatmap(df.corr(), annot=True)
plt.show()

# splitting data to training dataset and test data 70/30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

coef = model.coef_
inter = model.intercept_
print ('Suoran yhtälö on: ')
print (f'y = {round(coef[0][0],2)} * x + {round(inter[0],2)}')

# prediction with test data
y_pred = model.predict(X_test)

# seaborn regplot
print('seaborn regplot:')
sns.regplot(X_test, y_test, ci=None)
plt.show()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'r2: {r2}')
print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

print('Test: red, prediction: blue')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Kokemus')
plt.ylabel('Palkka')
plt.title('Palkka vs Kokemus')
plt.show()

# print employees salary with 7 years experience
print(f'Uuden työntekijän palkka 7v kokemuksella on {model.predict([[7]])}')


#####################################################################

# %% Exercise 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# read data from csv
df = pd.read_csv('housing.csv')
print(df.head())

X = df.loc[:, ['median_income']]
y = df.loc[:, ['median_house_value']]

# visualizing data with scatterplot
print('scatter plot:')
plt.scatter(X, y)
plt.xlabel('median income')
plt.ylabel('house value')
plt.show()

# correlation heatmap
print('heatmap')
sns.heatmap(df.corr(), annot=True)
plt.show()

# splitting data to training dataset and test data 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

# prediction with test data
y_pred = model.predict(X_test)

# seaborn regplot
print('seaborn regplot:')
sns.regplot(X_test, y_test, ci=None)
plt.show()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# print values
print(f'r2: {r2}') # determination value, r square
print(f'mae: {mae}') # mean absolute error
print(f'mse: {mse}') # mean squared error
print(f'rmse: {rmse}') # root mean squared error

# create plot
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Value of house')
plt.ylabel('Yearly income')
plt.title('Value of house by yearly income')
plt.show()

print(f'Kotitalouden talon arvo, kun vuositulot on 30 000 dollaria: {model.predict([[30000]])}')
# %%
