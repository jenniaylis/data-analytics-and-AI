# %%
import pandas as pd
import pickle # load encoder

# load model
with open('startup-model.pickle', 'rb') as f:
    model = pickle.load(f)
    
# load encoder
with open('startup-ct.pickle', 'rb') as f:
    ct = pickle.load(f)

# prediction with new data
Xnew = pd.read_csv('new_company.csv')
Xnew_org = Xnew
Xnew = ct.transform(Xnew)
ynew = model.predict(Xnew)

for i in range (len(ynew)):
    print(f'{Xnew_org.iloc[i]}\n Voitto: {ynew[i][0]}\n')

coef = model.coef_
inter = model.intercept_
# %%
