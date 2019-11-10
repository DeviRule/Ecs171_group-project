# Ecs171_group-project
handle credit card fraud dataset

https://www.kaggle.com/mlg-ulb/creditcardfraud

required package:
imbalanced-learn
sklearn

#### Currently, the f1 score indicate that neither sampling strategies works.
Classifer with balanced class weight only: 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     99511
           1       0.90      0.05      0.10       173

    accuracy                           1.00     99684
   macro avg       0.95      0.53      0.55     99684
weighted avg       1.00      1.00      1.00     99684

0    398
1    319
dtype: int64

Classifer with undersampling dataset: 
              precision    recall  f1-score   support

           0       1.00      0.95      0.98     99511
           1       0.03      0.92      0.06       173

    accuracy                           0.95     99684
   macro avg       0.52      0.94      0.52     99684
weighted avg       1.00      0.95      0.97     99684

1    184804
0    184804
dtype: int64

Classifer with oversampling dataset:
              precision    recall  f1-score   support

           0       1.00      0.96      0.98     99511
           1       0.04      0.93      0.08       173

    accuracy                           0.96     99684
   macro avg       0.52      0.95      0.53     99684
weighted avg       1.00      0.96      0.98     99684