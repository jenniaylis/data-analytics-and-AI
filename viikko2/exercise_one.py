# %%
import pandas as pd
import matplotlib.pylab as plt


employeesData = pd.read_csv('employees.csv')
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
merged_data.insert('age')
# %%
