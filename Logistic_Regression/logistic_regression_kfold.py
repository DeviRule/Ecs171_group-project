
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler

def FiveFoldROC(estimator, X, y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    cv = StratifiedKFold(n_splits=5)
    for train, test in cv.split(X, y):
    # Compute ROC curve and area the curve
        #print(train)
        #print(test)
        estimator.fit(X[train], y[train])
        y_score = estimator.predict_proba(X[test])[:,1]
        fpr, tpr, thresholds = roc_curve(y[test], y_score, pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# test on original dataset
df = pd.read_csv('../Data/train.csv', header=0)
X = df.iloc[:,0:-1].copy()
Y = df.iloc[:, -1].copy()

df = pd.read_csv('../Data/validation_under.csv', header=0)
X_valid = df.iloc[:,0:-1].copy()
Y_valid = df.iloc[:, -1].copy()

# Handle the dataset with SMOTE
SM = SMOTE(random_state=0)
X_smote, Y_smote = SM.fit_sample(X, Y)

class_weight = {0: 33, 1: 67}
lr = LogisticRegression(random_state=0, class_weight=class_weight, solver='lbfgs')
lr.fit(X_smote, Y_smote)
Y_predit = lr.predict(X_valid)
Y_train = lr.predict(X_smote)
print(classification_report(Y_smote, Y_train))
print(classification_report(Y_valid, Y_predit))

df = pd.read_csv('../Data/creditcard.csv', header=0)
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,0:-1].copy()
y = df.iloc[:, -1].copy()
rus = RandomUnderSampler(sampling_strategy=1)
X_res, y_res = rus.fit_resample(X, y)

class_weight = {0: 33, 1: 67}
lr = LogisticRegression(random_state=0, class_weight=class_weight, solver='lbfgs')
FiveFoldROC(lr, X_res, y_res)
