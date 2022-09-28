# %% tehtävä 1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read data from excel
dfs = pd.read_excel("tt.xlsx")
# create the crosstab
dg_df = pd.crosstab(index=dfs['koulutus'], columns=['nro']).reset_index()
# re-naming columns and rows
dg_df.columns = ['koulutus','Lukumäärä']
dg_df['koulutus'] = ['Peruskoulu', '2. aste', 'Korkeakoulu', 'Ylempi korkeakoulu']

# calculating percentages and adding them to crosstab
p_m = dg_df['Lukumäärä'].sum()
tot = p_m
dg_df['%'] = (dg_df['Lukumäärä'] / tot * 100)
print(dg_df)

# creating barplot
dg_df.plot(kind='barh', x='koulutus', y='Lukumäärä', title='Koulutustausta')
plt.show()






# %% tehtävä 2
import pandas as pd

dfs = pd.read_excel("tt.xlsx")
# create the crosstab
dg_df = pd.crosstab(index=dfs['koulutus'], columns=dfs['sukup']).reset_index()
# re-naming columns and rows
dg_df.columns = ['koulutus', 'mies', 'nainen']
dg_df['koulutus'] = ['Peruskoulu', '2. aste', 'Korkeakoulu', 'Ylempi korkeakoulu']


print(dg_df)





# %% tehtävä 3
from scipy.stats import chi2_contingency
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# read data from excel
df = pd.read_excel("tt.xlsx")
# create the crosstab
dg_df = pd.crosstab(index=df['koulutus'], columns=df['sukup']).reset_index()
# re-naming columns and rows

dg_df.columns = ['koulutus', 'mies', 'nainen']
dg_df['koulutus'] = ['Peruskoulu', '2. aste', 'Korkeakoulu', 'Ylempi korkeakoulu']

#chi square
p = stats.chi2_contingency(dg_df[['mies', 'nainen']])[1]
if (p > 0.05):
    print(f'Riippuvuus ei ole tilastollisesti merkitsevä, p= {p}')

if (p < 0.05):
    print('Riippuvuus on tilastollisesti merkitsevä, p= {p}')
    
print('Chisquare output: ',chi2_contingency(dg_df[['mies', 'nainen']]))
# output: chi2, p value, degrees of freedom, array

# creating barplot
dg_df.plot(kind='barh', x='koulutus', title='Koulutustausta')
plt.xlabel('Lukumäärä')
plt.show()






# %% tehtävä 4
import matplotlib.pyplot as plt
import seaborn as sns

dfs = pd.read_excel("tt.xlsx")
# creating new dataframe with fewer columns
df_corr = dfs[['sukup', 'ikä', 'perhe', 'koulutus', 'palkka']].copy()
# exploring correlation
print(df_corr.corr())
# printing correlation heatmap
sns.heatmap(df_corr.corr(), annot=True)
# %%
