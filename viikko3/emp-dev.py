# %%
import pandas as pd
import matplotlib.pylab as plt

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

# scatter chart with age on axis x and salary on axis y
employeesData.plot.scatter(x='age', y='salary', c='DarkBlue')

# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

# bar chart how many employees per department
employeesData['dname'].value_counts().plot(kind='bar')

# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

# turning vertical plot to horizontal
employeesData['dname'].value_counts().plot(kind='barh')

# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

# bar chart of employees by age groups
bins= [30, 40, 50, 60, np.inf]
labels = ['30-40','40-50','50-60','60+']
pd.cut(employeesData['age'], bins=bins, labels=labels).value_counts()[['30-40','40-50','50-60','60+']].plot(kind='bar', title='Age groups')

# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

#changing the header name of the row
employeesData['gender'] = employeesData['gender'].replace(0, 'male')
employeesData['gender'] = employeesData['gender'].replace(1, 'female')

# pie chart, percentage of men and women
employeesData.groupby('gender').size().plot(kind='pie', autopct='%.1f%%', colors=['violet', 'lightblue'], title='Genders')

# remove y label
plt.ylabel("")
# %%
# #############################################################################
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# read the csv file
employeesData = pd.read_csv('emp-dep.csv')

# changing the header name of the row
employeesData['gender'] = employeesData['gender'].replace(0, 'male')
employeesData['gender'] = employeesData['gender'].replace(1, 'female')

# multiple bar diagram of employees by age groups and gender
counts = employeesData.groupby(['age_group', 'gender']).age.count().unstack()
counts.plot(kind='bar')

# changing titles
plt.title('Employees by age')
plt.xlabel('Age group')
plt.ylabel('Amount')

# %%
