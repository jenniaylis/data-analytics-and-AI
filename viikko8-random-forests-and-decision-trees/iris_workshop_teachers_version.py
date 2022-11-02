# %%
# this was teachers version on workshop
import pandas as pd
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn import tree, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import graphviz

# lue iris.csv dataset pandasin dataframeen
df = pd.read_csv('iris.csv')
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='Species', data=df)
plt.show()

# luo X ja y (species)
X = df.iloc[:, [0, 1, 2, 3]]
y = df.iloc[:, [4]]
columns = X.columns

# jaa data training ja test setteihin (75 - 25 %)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# opeta DecisionTreeClassifier
model = tree.DecisionTreeClassifier(max_depth=3, criterion="gini")
model.fit(X_train, Y_train)

# tee ennusteet testiaineistolla
y_pred = model.predict(X_test)
y_pred_pros = model.predict_proba(X_test)

# luo confusion matrix (Huom:  nyt meillä on kolme luokkaa, 
# joten tämä ei ole binäärinen tapaus), ja visualisoi seabornin heatmapilla (annot=True)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(f'accuracy: {acc}')

ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_title('DT')
plt.show()

# visualisoi päätöspuu käyttäen graphviz-moduulia ja tallenna kuva  .png formaatissa
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

# ennusta tulokset oheisella uudella datalla new-iris.csv 
xnew = pd.read_csv('new-iris.csv')

y_pred_new = model.predict(xnew)
y_pred_new_pros = model.predict_proba(xnew)

print(f'prediction: {y_pred_new}')
print(f'prediction percentage: \n {y_pred_new_pros}')

# %%
