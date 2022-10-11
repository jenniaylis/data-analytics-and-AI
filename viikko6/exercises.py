# %%
# EXERCISE 1
############################################################
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# read data from csv, create dataframe
df = pd.read_csv('startup.csv')

X = df.iloc[:, :-1] # selittävät muuttujat kulut ja osavaltio
y = df.iloc[:, [-1]] # selitettävä muuttuja profit

# state dummies
dummies = pd.get_dummies(df, drop_first=True)
#print('DUMMIES: \n', dummies)

#X_org = X
# state dummies with one-hot-encoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')

X = ct.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# teaching model
model = LinearRegression()
model.fit(X_train, y_train)

# creating prediction
y_pred = model.predict(X_test)
print(f'prediction: ',y_pred)

# printing values
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'r2: {r2}')
print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

# scale data, after split
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)

# EXERCISE 3 part 1 (startup_train.py):
# tallennetaan malli levylle
with open('startup-model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
# tallennetaan encoder
with open('startup-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
    
# tallennetaan skaaleri x
with open('startup-scaler-x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)

# tallennetaan skaaleri y
with open('startup-scaler-y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)

# %%
# EXERCISE 2
############################################################
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# read data from csv, create dataframe
df = pd.read_csv('new_company.csv')

dummies = pd.get_dummies(df, drop_first=True)
print(dummies)

# read data from csv, create dataframe
df_ct = pd.read_csv('new_company_ct.csv')

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')

# %%
# EXERCISE 3
############################################################
import pandas as pd

# %%
# EXERCISE 4
############################################################
import pandas as pd
