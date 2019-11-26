import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../Data/creditcard.csv', header=0)

# Comment the following block to remove standarlization
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print(df)

droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
df = df.drop(droplist, axis = 1)

frauddf = df[df['Class'] == 1]
notfrauddf = df[df['Class'] == 0]

tra_frauddf, val_frauddf = train_test_split(frauddf, test_size=0.35, random_state=0)
tra_notfrauddf, val_notfrauddf = train_test_split(notfrauddf, test_size=0.35, random_state=0)

train_set = tra_frauddf.append(tra_notfrauddf, ignore_index=True)
val_set = val_frauddf.append(val_notfrauddf, ignore_index=True)

train_set = train_set.sample(frac=1).reset_index(drop=True)
val_set = val_set.sample(frac=1).reset_index(drop=True)

train_set.to_csv(r'../Data/train_stand.csv', index = None, header=True)
val_set.to_csv(r'../Data/validation_stand.csv', index = None, header=True)