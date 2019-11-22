import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

val_set = pd.read_csv('../Data/validation.csv', header=0)

X = val_set.iloc[:,0:-1].copy()
Y = val_set.iloc[:, -1].copy()

sm = SMOTE(random_state=42)

X_res, Y_res = sm.fit_resample(X, Y)
#print(pd.DataFrame(X_res))
#print(len(Y_res))
val_smote = pd.DataFrame(X_res).assign(Class = Y_res)
val_smote.columns = ['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','Amount','Class']
val_smote = val_smote.sample(frac=1).reset_index(drop=True)
#print(val_smote)
val_smote.to_csv(r'../Data/validation_smote.csv', index = None, header=True)


