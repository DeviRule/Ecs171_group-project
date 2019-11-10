import pandas as pd
from sklearn.metrics import roc_auc_score
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('../Data/train.csv', header=0)
X = df.loc[:, 'Time':'Amount'].copy()
Y = df.loc[:, 'Class'].copy()

#scaler = StandardScaler()
#X = scaler.fit_transform(X)
class_weight = {0: 998, 1: 2}
lr = LogisticRegression(random_state=0, class_weight=class_weight, solver='lbfgs')
classifier = lr.fit(X, Y)

df = pd.read_csv('../Data/validation.csv', header=0)
X_valid = df.loc[:, 'Time':'Amount'].copy()
Y_valid = df.loc[:, 'Class'].copy()
#scaler = StandardScaler()
#X_valid = scaler.fit_transform(X_valid)

#classifier_probs = classifier.predict_proba(X_valid)
print("Classifer with balanced class weight only: ")
Y_predit = classifier.predict(X_valid)
print(classification_report(Y_valid, Y_predit))
roc_auc_score(Y_valid, Y_predit)

#Handle the dataset with undersampling strategy
rus = RandomUnderSampler(sampling_strategy=0.8)
X_res, Y_res = rus.fit_resample(X, Y)

print(pd.value_counts(Y_res))

class_weight = {0: 5, 1: 4}
lr = LogisticRegression(random_state=0, class_weight=class_weight, solver='lbfgs')
classifier = lr.fit(X_res, Y_res)
print("Classifer with undersampling dataset: ")
Y_predit = classifier.predict(X_valid)
print(classification_report(Y_valid, Y_predit))
roc_auc_score(Y_valid, Y_predit)

#Handle the dataset with oversampling strategy
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X, Y)

print(pd.value_counts(Y_resampled))

lr = LogisticRegression(random_state=0, solver='lbfgs')
classifier = lr.fit(X_resampled, Y_resampled)
print("Classifer with undersampling dataset: ")
Y_predit = classifier.predict(X_valid)
print(classification_report(Y_valid, Y_predit))
roc_auc_score(Y_valid, Y_predit)
