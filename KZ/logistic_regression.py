import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Data/creditcard.csv', header=0)
print(pd.value_counts(df['Class'], normalize = True))