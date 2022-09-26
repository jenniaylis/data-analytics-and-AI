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

dfs = pd.read_excel("tt.xlsx")
# create the crosstab
dg_df = pd.crosstab(index=dfs['koulutus'], columns=dfs['sukup']).reset_index()
# re-naming columns and rows
dg_df.columns = ['koulutus', 'mies', 'nainen']
dg_df['koulutus'] = ['Peruskoulu', '2. aste', 'Korkeakoulu', 'Ylempi korkeakoulu']

#chi square
obs = np.array(dfs['koulutus'], dtype=np.float32)
exp = np.array(dfs['sukup'], dtype=np.float32)
print(obs)
print(exp)
chi2_contingency(obs, exp)
print("chi2: %f\np_value: %f" % stats.chisquare(obs, exp))
# output: chi2, p value, degrees of freedom, array

# creating barplot
dg_df.plot(kind='barh', x='koulutus', title='Koulutustausta')
plt.xlabel('Lukumäärä')
plt.show()

# %% tehtävä 4

dfs = pd.read_excel("tt.xlsx")
# creating new dataframe with fewer columns
df_corr = dfs[['sukup', 'ikä', 'perhe', 'koulutus', 'palkka']].copy()
# exploring correlation
print(df_corr.corr())
# printing correlation heatmap
sns.heatmap(df_corr.corr(), annot=True)
# %%
