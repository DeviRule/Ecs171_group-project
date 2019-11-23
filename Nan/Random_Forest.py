import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,classification_report, matthews_corrcoef, accuracy_score, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

transactions = pd.read_csv('../Data/train.csv')
x_train = transactions.drop(labels='Class', axis=1)
y_train = transactions.loc[:,'Class']

# Handle the dataset with undersampling strategy
rus = RandomUnderSampler(sampling_strategy=0.8)
X_res, Y_res = rus.fit_resample(x_train, y_train)

# Handle the dataset with oversampling strategy
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X, Y)

# Handle the dataset with SMOTE
SM = SMOTE(random_state=0)
X_smote, Y_smote = SM.fit_sample(X, Y)

num_folds = 5
MCC_scorer = make_scorer(matthews_corrcoef)

rf = RandomForestClassifier(n_jobs=-1, random_state=1)

n_estimators = [50, 75, 500]  #default = 50;
# ,50, 60, 90, 105, 120, 500, 1000
min_samples_split = [2, 5] # default=2
# , 5, 10, 15, 100
min_samples_leaf = [1, 5]  # default = 1
# , 2, 5, 8

param_grid_rf = {'n_estimators': n_estimators,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_split,
                }

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,cv=num_folds,scoring = MCC_scorer, 
                       n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1, return_train_score=False)
grid_rf_undersample = GridSearchCV(estimator=rf, param_grid=param_grid_rf,cv=num_folds,scoring = MCC_scorer, 
                       n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1, return_train_score=False)
grid_rf_oversample = GridSearchCV(estimator=rf, param_grid=param_grid_rf,cv=num_folds,scoring = MCC_scorer, 
                       n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1, return_train_score=False)
grid_rf_smote = GridSearchCV(estimator=rf, param_grid=param_grid_rf,cv=num_folds,scoring = MCC_scorer, 
                       n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1, return_train_score=False)

#origin
grid_rf.fit(x_train, y_train)
grid_rf.best_score_
grid_rf.best_params_

evaluation = pd.read_csv('../Data/validation.csv')
x_eval = evaluation.drop(labels='Class', axis=1)
y_eval = evaluation.loc[:,'Class']
def Random_Forest_eval(estimator, x_test, y_test):
    
    y_pred = estimator.predict(x_test)

    print('Classification Report')
    print(classification_report(y_test, y_pred))
    if y_test.nunique() <= 2:
        try:
            y_score = estimator.predict_proba(x_test)[:,1]
        except:
            y_score = estimator.decision_function(x_test)
        print('AUPRC', average_precision_score(y_test, y_score))
        print('AUROC', roc_auc_score(y_test, y_score))

Random_Forest_eval(grid_rf, x_eval, y_eval)

#undersample
grid_rf_undersample.fit(X_res, Y_res)
grid_rf_undersample.best_score_
grid_rf_undersample.best_params_
Random_Forest_eval(grid_rf_undersample, x_eval, y_eval)

#oversample
grid_rf_oversample.fit(X_resampledm, Y_resampled)
grid_rf_oversample.best_score_
grid_rf_oversample.best_params_
Random_Forest_eval(grid_rf_oversample, x_eval, y_eval)

#oversample
grid_rf_smote.fit(X_resampledm, Y_resampled)
grid_rf_smote.best_score_
grid_rf_smote.best_params_
Random_Forest_eval(grid_rf_smote, x_eval, y_eval)
