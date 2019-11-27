# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:03:22 2019

@author: Zekun Chen,Yuqi Sha,Rongfei Li
"""

# AdaBoost_main_under.py is used as the main driver for the AdaBoostClassifer
# for data with undersampling treatment

## PLEASE PUT THESE PYTHON FILES WITH THE DATA IN THE SAME FOLDER
## SO THE CODE CAN RUN SMOOTHLY!

from AdaBoost_funcs import Draw_PR
from AdaBoost_funcs import make_predictions
from AdaBoost_funcs import get_decision_ROC
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier


#### Classification Model (AdaBoost) Build UP & Parameter Tuning 
#### Based on Validation Data with undersampling treaments ######


# Define predictor_str and target 

target = 'Class'

# Use this trigger which data
# to intake. If use the
# max-min nomralized data
# set the booleaning flag 
# to True. Otherwise, set it
# as False

is_max_min_normalized = True

if is_max_min_normalized:
  train_data = pd.read_csv('train_stand.csv')

else:
  train_data = pd.read_csv('train.csv')


if train_data.shape[1]==30:

  predictors_str = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 
                    'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16',\
                    'V17', 'V18', 'V19','V20', 'V21', 'V22', 'V23', 'V24', 
                    'V25', 'V26', 'V27', 'V28','Amount']
else:

# Feature used for predictions is shrinked to 18. Please see report for more 
# details.

  predictors_str = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10',\
       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19',\
       'Amount']


# Use smaller RANDOM_STATE for 
# undersample data set beacause
# it is smaller in size

RANDOM_STATE = 12 
NUM_ESTIMATORS = 150
lr = 1

# Construct model

clf = AdaBoostClassifier(random_state=RANDOM_STATE,algorithm='SAMME.R',
              learning_rate=lr,n_estimators=NUM_ESTIMATORS)

# Perform training

clf.fit(train_data[predictors_str], train_data[target].values)

# Make predictions for later classification report

predictions_train = make_predictions(clf,predictors_str,train_data)

# Trace estimator weight and estimator error

is_show_model_details = False
if is_show_model_details:
  print('Estimator Weight: ',clf.estimator_weights_)
  print('\n')
  print('Estimator type: ',clf.estimators_[0])
  print('\n')
  print('Estimator ErrorL ',clf.estimator_errors_)

######## Classification Performance on Validation Data with Undersampling######


# Load in validation data

if is_max_min_normalized:
  validation_under_data = pd.read_csv('validation_stand_under.csv')

else:
  validation_under_data = pd.read_csv('validation_under.csv')

# Make prediciton based on the load in data

predictions_valid_under = make_predictions(clf,predictors_str,validation_under_data)

# ROC curve

prob_valid_under = clf.predict_proba(validation_under_data[predictors_str])[:,1]
print('\n')

# PR/ROC curve

get_decision_ROC(clf, validation_under_data[predictors_str].values, validation_under_data[target].values)
Draw_PR(prob_valid_under, predictions_valid_under, validation_under_data[target].values, 
        is_draw_dot = True,model_name = 'AdaBoost',is_save_fig=True)