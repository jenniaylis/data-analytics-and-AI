import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

healthData = pd.read_csv('diabetes.csv')
plt.figure(figsize=(16, 6)) # Increase the size of the heatmap.
heatmap = sns.heatmap(healthData.corr(), vmin=-1, vmax=1, annot=True) 
heatmap.set_title('Correlation Heatmap - Diabetes', fontdict={'fontsize':12}, pad=12)

sns.heatmap(healthData.corr())