from sklearn import tree
import pandas as pd
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def get_ROC_5fold_plot(classifier, X, y):
    cv = StratifiedKFold(n_splits=5)
    fig = plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    return fig, aucs

fat_test = pd.read_csv("../Data/validation.csv")
y_fat = fat_test["Class"]
x_fat = fat_test.drop(['Class', 'Amount'], axis = 1)

train = pd.read_csv("../Data/train.csv")
y_orig = train["Class"]
X_orig = train.drop(['Class', 'Amount'], axis = 1)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_orig, y_orig)

test = pd.read_csv("../Data/validation_under.csv")
y_test = train["Class"]
X_test = train.drop(['Class', 'Amount'], axis = 1)

"""
parameters = {'max_depth' : [1, 5, 10, 15, 20], 'min_samples_leaf' : [10, 25, 50]}
clf = tree.DecisionTreeClassifier(class_weight={0:98, 1:2})
grid = GridSearchCV(clf, parameters, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
"""

def create_report(clf, X_train, Y_train, X_test, Y_test):
    target_names = ['normal', 'fraud']
    print("TRAIN\n")
    print(classification_report(Y_train, clf.predict(X_train), target_names=target_names))
    print("TEST\n")
    print(classification_report(Y_test, clf.predict(X_test), target_names=target_names))

print("Here comes the default classifier performance:")
default = tree.DecisionTreeClassifier()
default.fit(X_train, y_train)
create_report(default, X_train, y_train, x_fat, y_fat)
print("As you can see: default classifier has serious overfitting issues.")

print("Now proceed to parameter analysis")
max_depths = np.linspace(1, 30, num=15)
min_samples_leaves = np.linspace(1, 100, num=20)

f1_lists  = []
for depth in max_depths:
    clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10, class_weight={0: 98, 1: 2})
    # best_model = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    f1_for_fruad = f1_score(y_fat, clf.predict(x_fat), average=None)[1]
    f1_lists.append(f1_for_fruad)
plt.plot(max_depths, f1_lists)
plt.xlabel('max #Depths')
plt.ylabel('f1 score')
plt.show()

f1_lists  = []
for leaf in min_samples_leaves:
    clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=int(leaf), class_weight={0: 98, 1: 2})
    # best_model = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    f1_for_fruad = f1_score(y_fat, clf.predict(x_fat), average=None)[1]
    f1_lists.append(f1_for_fruad)
plt.plot(min_samples_leaves, f1_lists)
plt.xlabel('min #leaves')
plt.ylabel('f1 score')
plt.show()

best_model = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight={0: 98, 1: 2})
best_model.fit(X_train, y_train)
print("Improved model Result:")
create_report(best_model, X_train, y_train, x_fat, y_fat)
fig2, aucs = get_ROC_5fold_plot(best_model, x_fat.to_numpy(), y_fat.to_numpy())
fig2.suptitle('ROC with 5-fold cross-validation')
plt.savefig('ROC Decision Tree.png')

