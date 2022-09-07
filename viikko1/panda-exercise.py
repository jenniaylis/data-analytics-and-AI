# %%
import pandas as pd
import matplotlib.pylab as plt

#load data from csv
healthData = pd.read_csv('diabetes.csv')
print(healthData.describe()) # print all (max, min, average, std etc)

print('Count values:')
print(healthData.count()) # print count of every column of the dataset
print('Count of pregnancies:', healthData.Pregnancies.count()) # print count of pregnancies

print('Max values:')
print(healthData.max()) # print max values of every column
print('Max BMI:',healthData.BMI.max()) # print max BMI

print('Min values:')
print(healthData.min()) # print minimum values of every column

print('Mean values:')
print(healthData.mean()) # average values of given dataset

print('Std:') # standard deviation values of given dataset
print(healthData.std())
# %%
import pandas as pd
import matplotlib.pylab as plt
healthData = pd.read_csv('diabetes.csv')

healthData.hist()
healthData.hist('Pregnancies')

# %%
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

healthData = pd.read_csv('diabetes.csv')
plt.figure(figsize=(16, 6)) # Increase the size of the heatmap.
heatmap = sns.heatmap(healthData.corr(), vmin=-1, vmax=1, annot=True) 
heatmap.set_title('Correlation Heatmap - Diabetes', fontdict={'fontsize':12}, pad=12)

sns.heatmap(healthData.corr())
# %%
import pandas as pd

healthData = pd.read_csv('diabetes.csv')
dups = healthData.pivot_table(index = ['Age'], aggfunc ='size')
print(dups)

# %%
import pandas as pd

healthData = pd.read_csv('diabetes.csv')
dups = healthData.pivot_table(index = ['Outcome'], aggfunc ='size')
print(dups)
# %%
import pandas as pd

healthData = pd.read_csv('diabetes.csv')
count_of_nan = healthData.isna().sum()
print('COUNT NAN VALUES')
print(count_of_nan)
healthData.notnull()
# %%
