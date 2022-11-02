# %%
from email import header
import pandas as pd
import numpy as np

titanicData = pd.read_csv('Titanic_data.csv')
namesData = pd.read_csv('Titanic_names.csv')
print('TITANIC INFO')
titanicData.info()
print('TITANIC DESCRIBE')
print(titanicData.describe())
print('')

print('NAMES INFO')
namesData.info()
print('NAMES DESCRIBE')
print(namesData.describe())
print('')

titanicData.hist(bins=4) # showing the histogram

print('merged data')
df = titanicData.merge(namesData,how='inner',on=["id"]) # merging two dataframes into one
print(df.head(10)) # testing that data prints out correctly
print('')

print('There were', len(df.index), 'persons on Titanic.')
print(df['GenderCode'].value_counts()[0], 'were men.')
print(df['GenderCode'].value_counts()[1], 'were women.')

print('Age average was', round(df['Age'].mean(),2))
print('There were', df['Age'].value_counts()[0], 'zero years old')

df['Age'] = df['Age'].replace(0, np.NaN)
print('Age average excluding zero values is', round(df['Age'].mean(),2))
print(' ')
dups = df.pivot_table(index = ['PClass'], aggfunc ='size')
print(dups)

print('')
# finding the person traveling in class '*', printing just the name (and row index)
print('There is one person travelling on * class. Person is:', df.query("PClass == '*'")
      .filter(items=['Name'])
      .rename(columns={'Name':''}))
print(' ')

print(df['Survived'].value_counts()[0], "didn't survive.")
print(df['Survived'].value_counts()[1], 'survived.')
print(' ')

df['Survived'] = df['Survived'].replace(0, 'Drowned') #changing the header name of the row
df['Survived'] = df['Survived'].replace(1, 'Survived')
df1 = (df['Survived'].value_counts(normalize=True) #counting how many duplicate values
                .mul(100) #multiplied by 100
                .rename_axis('Survival') # renaming column headers
                .reset_index(name='percentage')) # renaming column headers
print(df1)
print('')

survival = df.groupby(['Survived', 'Gender'])
print(survival.size())
# %%
