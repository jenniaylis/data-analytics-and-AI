# %%
from email.utils import parsedate_to_datetime
import pandas as pd
import matplotlib.pylab as plt
from datetime import date


employeesData = pd.read_csv('employees.csv', parse_dates=['bdate'])
del employeesData['image']

departmentsData = pd.read_csv('departments.csv')

merged_data = employeesData.merge(departmentsData,how='inner',on=["dep"])
print('-------------------------------------------')
print('How many employees in dataframe:',merged_data.id.count())
print('-------------------------------------------')
print('How many women in dataframe:', merged_data['gender'].value_counts()[1])
print('How many men in dataframe:', merged_data['gender'].value_counts()[0])
print('-------------------------------------------')
merged_data['gender'] = merged_data['gender'].replace(0, 'male') #changing the header name of the row
merged_data['gender'] = merged_data['gender'].replace(1, 'female')
df1 = (merged_data['gender'].value_counts(normalize=True) #counting how many duplicate values
                .mul(100) #multiplied by 100
                .rename_axis('gender') # renaming column headers
                .reset_index(name='percentage')) # renaming column headers
print(df1)
print('-------------------------------------------')
print('Maximum salary of employees:', merged_data.salary.max())
print('Minimum salary of employees:', merged_data.salary.min())
print('Average salary of employees:', merged_data.salary.mean())
print('-------------------------------------------')

print('Average salary by departments:', merged_data.groupby('dname').agg({'salary':'mean'}))
print('-------------------------------------------')
print('Count of employees without second phone number:', merged_data['phone2'].isna().sum())
print('-------------------------------------------')

# function
def calc_age(bd: pd.Series) -> pd.Series:
    today = pd.to_datetime(date.today())  # convert today to a pandas datetime
    return round((today - bd) / pd.Timedelta(days=365.25),0)  # divide by days to get years, round decimals

# call function and assign the values to a new column in the dataframe
merged_data['Age'] = calc_age(merged_data.bdate)
merged_data.head(15)

# %%
from email.utils import parsedate_to_datetime
import pandas as pd
import matplotlib.pylab as plt
from datetime import date
import seaborn as sns

employeesData = pd.read_csv('employees.csv', parse_dates=['bdate'])
del employeesData['image']

departmentsData = pd.read_csv('departments.csv')

merged_data = employeesData.merge(departmentsData,how='inner',on=["dep"])

def calc_age(bd: pd.Series) -> pd.Series:
    today = pd.to_datetime(date.today())  # convert today to a pandas datetime
    return round((today - bd) / pd.Timedelta(days=365.25),0)  # divide by days to get years, round decimals

# call function and assign the values to a new column in the dataframe
merged_data['Age'] = calc_age(merged_data.bdate)

new_Dataframe = merged_data[['salary', 'gender', 'Age']].copy()
new_Dataframe.head(10)
plt.figure(figsize=(16, 6)) # Increase the size of the heatmap.
heatmap = sns.heatmap(new_Dataframe.corr(), vmin=-1, vmax=1, annot=True) 
heatmap.set_title('Correlation Heatmap - Salaries', fontdict={'fontsize':12}, pad=12)

sns.heatmap(new_Dataframe.corr())
# %%
