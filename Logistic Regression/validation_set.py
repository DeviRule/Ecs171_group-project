import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

val_set = pd.read_csv('../Data/validation.csv', header=0)

X = val_set.iloc[:,0:-1].copy()
Y = val_set.iloc[:, -1].copy()
rus = RandomUnderSampler(sampling_strategy=1)
X_res, Y_res = rus.fit_resample(X, Y)
print(pd.DataFrame(X_res))
print(Y_res)
val_under = pd.DataFrame(X_res).assign(Class = Y_res)
val_under.columns = ['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','Amount','Class']
val_under = val_under.sample(frac=1).reset_index(drop=True)
print(val_under)
val_under.to_csv(r'../Data/validation_under.csv', index = None, header=True)

val_set = pd.read_csv('../Data/validation_stand.csv', header=0)

X = val_set.iloc[:,0:-1].copy()
Y = val_set.iloc[:, -1].copy()
rus = RandomUnderSampler(sampling_strategy=1)
X_res, Y_res = rus.fit_resample(X, Y)
print(pd.DataFrame(X_res))
print(Y_res)
val_under = pd.DataFrame(X_res).assign(Class = Y_res)
val_under.columns = ['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','Amount','Class']
val_under = val_under.sample(frac=1).reset_index(drop=True)
print(val_under)
val_under.to_csv(r'../Data/validation_stand_under.csv', index = None, header=True)