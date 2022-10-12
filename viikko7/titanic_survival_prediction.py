#%% EXERCISE 1, 2 and 3
########################################################
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# read data and create dataframe
df = pd.read_csv('titanic-class-age-gender-survived.csv')

# x and y datasets
X = df.iloc[:, :-1] # x is age, gender and class
y = df.iloc[:, [-1]] # y is survived

############## ↓ Exercise2and3Addition ↓ ####################
# dummies
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender', 'PClass'])], remainder='passthrough')

X = ct.fit_transform(X)
########### ↑ Exercise2and3Addition ↑ #####################

# split test data and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# create model
model = LogisticRegression()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'confusion matrix: {cm}')
print(f'accuracy: {acc}')
print(f'precision: {precision}')
print(f'recall: {recall}')

# create heatmap visualization
sns.heatmap(cm, annot=True, fmt='g')

# predict survival rate on new dataset
xnew = pd.read_csv('titanic-new.csv')

xnew = ct.transform(xnew) # dummies

y_pred_new = model.predict(xnew)
y_pred_new_pros = model.predict_proba(xnew)

print('prediction for Jack and Kate:')
print(y_pred_new_pros)

# %%
