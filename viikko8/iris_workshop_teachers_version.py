# %%
# this was teachers version on workshop
import pandas as pd
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('iris.csv')
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='Species', data=df)
plt.show()

X = df.iloc[:, 0:4]
y = df.iloc[:, [4]]
columns = X.columns

# splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# training the decision tree classification model
model = tree.DecisionTreeClassifier(max_depth=5, criterion="gini")
model.fit(X_train, Y_train)

# training the random forest classifier model
# model = ensemble.RandomForestClassifier(max_depth=5, criterion="entropy")
# model.fit(X_train, Y_train)

# predicting the test set results
y_pred = model.predict(X_test)
y_pred_pros = model.predict_proba(X_test)

# making the confusion matrix and 
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(f'accuracy: {acc}')

ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_title('DT')
plt.show()

# create dot file for graphviz visualization
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names = columns,
    class_names = df['Class'].unique(),
    filled = True,
    rounded = True
)

graph = graphviz.Source(dot_data)
graph.render(filename='iris_workshop', format='png')

# %%
