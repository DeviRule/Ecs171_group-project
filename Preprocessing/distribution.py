import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

data  = pd.read_csv('creditcard.csv')
droplist = ['Time', 'Amount']
data = data.drop(droplist, axis = 1)


v_features = data.iloc[:, :-1].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class == 1], bins=50)
    sns.distplot(data[cn][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.savefig("before_sampling_distribution.png")

attribute_names = data.columns
X = data.iloc[:,0:-1]
y = data.iloc[:, -1]
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)
y = np.matrix(y)
y = np.transpose(y)
data = np.concatenate((X, y), axis=1)
data = pd.DataFrame(data=data[:,:], index=[i for i in range(data.shape[0])], columns=attribute_names)

v_features = data.iloc[:, :-1].columns
fig2 = plt.figure()
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class == 1], bins=50)
    sns.distplot(data[cn][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.savefig("after_smote_distribution.png")

attribute_names = data.columns
X = data.iloc[:,0:-1]
y = data.iloc[:, -1]
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X, y = nm.fit_sample(X, y)
y = np.matrix(y)
y = np.transpose(y)
print("X dimension", X.shape[0], X.shape[1])
print("y dimension", X.shape[0], X.shape[1])
data = np.concatenate((X, y), axis=1)
data = pd.DataFrame(data=data[:,:], index=[i for i in range(data.shape[0])], columns=attribute_names)

v_features = data.iloc[:, :-1].columns
fig3 = plt.figure()
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class == 1], bins=50)
    sns.distplot(data[cn][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.savefig("after_nearmiss_distribution.png")

