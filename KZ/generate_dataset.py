import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Data/creditcard.csv', header=0)

frauddf = df[df['Class'] == 1]
notfrauddf = df[df['Class'] == 0]

tra_frauddf, val_frauddf = train_test_split(frauddf, test_size=0.35)
tra_notfrauddf, val_notfrauddf = train_test_split(notfrauddf, test_size=0.35)

train_set = tra_frauddf.append(tra_notfrauddf, ignore_index=True)
val_set = val_frauddf.append(val_notfrauddf, ignore_index=True)

train_set = train_set.sample(frac=1).reset_index(drop=True)
val_set = val_set.sample(frac=1).reset_index(drop=True)

train_set.to_csv(r'../Data/train.csv', index = None, header=True)
val_set.to_csv(r'../Data/validation.csv', index = None, header=True)