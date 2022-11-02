# %%
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# read the csv file
titanicData = pd.read_csv('titanic.csv')

# bar chart of travellers by age groups
bins= [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
labels = ['0-5','5-10','10-15','15-20', '20-25', '25-30', '30-35','35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75']
pd.cut(titanicData['Age'], bins=bins, labels=labels).value_counts()[['0-5','5-10','10-15','15-20', '20-25', '25-30', '30-35','35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75']].plot(kind='bar', title='Age groups on Titanic')

# I think this could be done more clever without those bins and labels, somehow. Code doesnt look pretty.
# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt

# read the csv file
titanicData = pd.read_csv('titanic.csv')

# pie chart, percentage of survived men and women
print(titanicData.groupby(['Gender', 'Survived'])['Survived'].count())

titanicData.query('Survived == 1').groupby('Gender').size().plot(kind='pie', autopct='%.1f%%', colors=['violet', 'lightblue'])

# change labels and titles
plt.ylabel("Survivors")
plt.title('Passengers: ' 
             + str(titanicData['id'].count()) 
             + '\n Survived women: '
             + str(titanicData.query('Survived == 1').groupby('GenderCode').value_counts()[1].count())
             + '\n Survived men: '
             + str(titanicData.query('Survived == 1').groupby('GenderCode').value_counts()[0].count())
)

# I tried to make 2 suptitles but ended up using line breaks instead...
# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

# read the csv file
titanicData = pd.read_csv('titanic.csv')

# delete class *
titanicData = titanicData.drop(titanicData[titanicData.PClass == '*'].index)

# add new column called Saved
def label_saved (row):
    if row['Survived'] == 1:
      return 'yes'
    else :
      return 'no'

titanicData['Saved'] = titanicData.apply(lambda row: label_saved(row), axis=1)

# seaborn box and whisker plot representing classes, survival rate and age
sns.boxplot(data=titanicData, x="PClass", y="Age", hue="Saved")
plt.title('Titanic survivors')
# %%
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

# read the csv file
titanicData = pd.read_csv('titanic.csv')

# delete class *
titanicData = titanicData.drop(titanicData[titanicData.PClass == '*'].index)

# add new column called Saved
def label_saved (row):
    if row['Survived'] == 1:
      return 'yes'
    else :
      return 'no'

titanicData['Saved'] = titanicData.apply(lambda row: label_saved(row), axis=1)

# seaborn swarmplot representing classes, survival rate and age
sns.swarmplot(data=titanicData, x="PClass", y="Age", hue="Saved")
plt.title('Titanic survivors')
# %%
