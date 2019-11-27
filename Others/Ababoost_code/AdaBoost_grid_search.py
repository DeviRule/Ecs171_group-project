# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:40:31 2019

@author: Zekun Chen,Yuqi Sha,Rongfei Li
"""

# AdaBoost_grid_search.py is used to grid search optimze parameters
# for AdaBoostClassifer


from AdaBoost_funcs import make_predictions
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


###################### Grid Search for AdaBoost Classfifier ###################

# Define target 

target = 'Class'

# Use this trigger which data
# to intake. If use the
# max-min normalized data
# set the boolean flag 
# to True. Otherwise, set it
# as False

is_max_min_normalized = True

# If is_build_model_for_under is True
# Please run "Classification Performance on Validation Data with Undersampling" 
# cell. Otherwise, run "Classification Performance on alidation Set "

is_build_model_for_under = False

if not is_max_min_normalized:
  train_data = pd.read_csv('train.csv')

if is_max_min_normalized:
   train_data = pd.read_csv('train_stand.csv')

if train_data.shape[1]==30:

  predictors_str = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 
                    'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16',\
                    'V17', 'V18', 'V19','V20', 'V21', 'V22', 'V23', 'V24', 
                    'V25', 'V26', 'V27', 'V28','Amount']
else:
  predictors_str = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10',\
       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19',\
       'Amount']

# Define two parameters for Grid Search

lr_arr = np.arange(0.1,1.1,0.1)
NUM_ESTIMATORS_ARR = np.array([50,100,150,200,250,300])

if not is_max_min_normalized and not is_build_model_for_under:
  validation_data = pd.read_csv('validation.csv')
  RANDOM_STATE = 42

# Bulid a model for non-unsampling data, aim for high f1 score
  
  f1_score_list_tra = []
  f1_score_list_val = []

if is_max_min_normalized and not is_build_model_for_under:
  validation_data = pd.read_csv('validation_stand.csv')
  RANDOM_STATE = 42
  f1_score_list_tra = []
  f1_score_list_val = []


if not is_max_min_normalized and is_build_model_for_under:
  validation_data = pd.read_csv('validation_under.csv')
  RANDOM_STATE = 12

# Bulid a model for unsampling data, aim for high roc_auc_score
  
  roc_auc_score_list_tra = []
  roc_auc_score_list_val = []

if is_max_min_normalized and is_build_model_for_under:
  validation_data = pd.read_csv('validation_stand_under.csv')
  RANDOM_STATE = 12

  roc_auc_score_list_tra = []
  roc_auc_score_list_val = []


# Define number of estimators and construct model

for n in NUM_ESTIMATORS_ARR:
  for l in lr_arr:

    clf = AdaBoostClassifier(random_state=RANDOM_STATE,algorithm='SAMME.R',
              learning_rate=l,n_estimators=n)

    # Perform training

    print('Learing rate: ',l,' Num of esimators: ',n)
    print('\n')
    clf.fit(train_data[predictors_str], train_data[target].values)

    # Make predictors based on training and testing data

    preds_train = make_predictions(clf,predictors_str,train_data)
    preds_valid = make_predictions(clf,predictors_str,validation_data)
    

    # Perform grid search accordinly

    if is_build_model_for_under:

       roc_auc_score_from_tra = roc_auc_score(train_data[target].values,preds_train)
       roc_auc_score_from_valid = roc_auc_score(validation_data[target].values,preds_valid)
       roc_auc_score_list_tra.append(roc_auc_score_from_tra)
       roc_auc_score_list_val.append(roc_auc_score_from_valid)


    if not is_build_model_for_under:

       f1_from_tra = f1_score(train_data[target].values,preds_train)
       f1_from_valid = f1_score(validation_data[target].values,preds_valid)
       f1_score_list_tra.append(f1_from_tra)
       f1_score_list_val.append(f1_from_valid)

# Print out index fof the best combination

if not is_build_model_for_under:

  f1_from_tra_arr = np.array(f1_score_list_tra)
  f1_from_val_arr = np.array(f1_score_list_val)

  index_highest_f1_tra = np.where(f1_from_tra_arr == np.max(f1_from_tra_arr))
  index_highest_f1_val = np.where(f1_from_val_arr == np.max(f1_from_val_arr))

  if index_highest_f1_tra != index_highest_f1_val:

    # Use validation set as basis if best combination is different
    # between training and testing

    best_index = index_highest_f1_val[0][0]
  
  else:
     best_index = index_highest_f1_tra[0][0]
    
    

  print('Best combination @ trial %d : '%best_index)
  print('f1 score from training set: %.3f '%f1_from_tra_arr[best_index])
  print('f1 score from validation set: %.3f'%f1_from_val_arr[best_index])

if is_build_model_for_under:

  roc_auc_score_from_tra = np.array(roc_auc_score_list_tra)
  roc_auc_score_from_val = np.array(roc_auc_score_list_val)

  index_highest_roc_auc_tra = np.where(roc_auc_score_from_tra == np.max(roc_auc_score_from_tra))
  index_highest_roc_auc_val = np.where( roc_auc_score_from_val == np.max(roc_auc_score_from_val))

  if index_highest_roc_auc_tra != index_highest_roc_auc_val: 

    best_index = index_highest_roc_auc_val[0][0]
  
  else:
     best_index = index_highest_roc_auc_tra[0][0]
    
    
  print('Best combination @ trial %d : '%best_index)
  print('Roc auc score from training set: %.3f '%roc_auc_score_from_tra[best_index])
  print('Roc auc score from validation set: %.3f'%roc_auc_score_from_val[best_index])