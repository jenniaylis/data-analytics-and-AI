# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
from plotly.offline import plot

# creating dataframe from csv
df = pd.read_csv('country_vaccinations.csv')

# printing  first 5 lines of dataframe
print(f'head: \n {df.head()}')

# averages etc
print(f'describe: \n {df.describe()}')

# merging England, Scotland, Wales and Northern Ireland all to be United Kingdom
df = df[df.country.apply(lambda x: x not in ["England", "Scotland", "Wales", "Northern Ireland"])]

print('Data point starts from ',df.date.min())
print('Data point ends at ',df.date.max())
print('Total number of countries in the data set ',len(df.country.unique()))
print('Total number of unique vaccines in the data set ',len(df.vaccines.unique()))

# checking how many times each country occurs in the csv-dataset, and creating new csv-file out of that to make reading easier
country_counts = df.country.value_counts()
country_counts.to_csv('country_counts.csv')

# most used vaccine
vaccine = df["vaccines"].value_counts().reset_index()
vaccine.columns = ['Vaccines','Number of Country']
vaccines = vaccine.nlargest(5,"Number of Country")
vaccines.to_csv('vaccinations.csv') # checking data
vaccines.plot(y='Number of Country', kind='pie', labels=vaccines['Vaccines'], autopct='%.1f%%', figsize=(15,10))
plt.ylabel('')
plt.xlabel('Most used vaccines')
# figu = px.pie(vaccine, values='Number of Country', names='Vaccines', title='Most used vaccinations')
# figu.show()

# bar plot: top 10 countries with highest total vaccinations
vaccine = df.groupby(["country"])['people_fully_vaccinated'].max().nlargest(10).reset_index()
vaccine.columns = ["country", "Fully vaccinated"]
fig = px.bar(vaccine, x='country', y='Fully vaccinated', title='Top 10 countries with highest count of fully vaccinated')
fig.show()

# world map showing vaccination groups around the world
map = px.choropleth(data_frame=df, 
                    locations='country', 
                    locationmode='country names', 
                    color='vaccines', 
                    title='vaccines in different countries', 
                    color_discrete_sequence = px.colors.qualitative.Alphabet)
map.update_layout(height=400, margin={"r":0,"t":0,"l":0,"b":0})
plot(map)


# %%
