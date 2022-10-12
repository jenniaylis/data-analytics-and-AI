import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle #save encoder

# lue data sekä jaa X ja y
df = pd.read_csv('housing.csv')

X = df.iloc[:, [2,3,4,7, 9]]
print (f'Onko null arvoja: {X.isnull().sum()}\n')
X = X.fillna(0)

print (f'Onko null arvoja: {X.isnull().sum()}\n')
y = df.iloc[:, [8]]


X_org = X
# dummyt
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['ocean_proximity'])], remainder='passthrough')
X = ct.fit_transform(X) # ensimmäisellä kerralla fit_transform

# opetusdata ja testidata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# opetetaan usean muuttujan lineaarinen regressio
model = LinearRegression()
model.fit(X_train, y_train)

# ennustetaan testidatalla
y_pred = model.predict(X_test)

# metriikat
mae=mean_absolute_error(y_test, y_pred) 
r2=r2_score(y_test, y_pred)
mea = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mea)

print(f'r2:  {round(r2,4)}')
print(f'mae: {round(mae,4)}')
print(f'rmse: {round(rmse,4)}')


# tallennetaan malli levylle
with open('housing-model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
# tallennetaan encoderi
with open('housing-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)




