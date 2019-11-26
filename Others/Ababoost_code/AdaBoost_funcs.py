# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:59:14 2019

@author: Zekun Chen,Yuqi Sha,Rongfei Li
"""
# AdaBoost_funcs.py is used as the function driver for the AdaBoostClassifer

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import interp
plt.style.use('seaborn-poster')

############################### Function declaration ##########################

def make_predictions(clf_object,predictors_str,data_source):

  """make_predictions comes up with predictions
     from given input data

      Input:

              clf_object
              object
              constructed classification model

              predictors_str
              nd str array
              string array containing names
              of predictors

              data_source
              ndarray
              source of data
              either from valid
              or test


      Output:
              preds
              ndarray
              prediction classes based on
              given input data

  """
  preds = clf_object.predict(data_source[predictors_str])
  return preds


###############################################################################

def Draw_ROC(Y_prob, Y_observed, model_name = 'Model',is_save_fig = True):

    """Draw_ROC is the utility function to plot Receiver Operating
       Characteristic (ROC) curve

      Input:

              Y_prob
              ndarray
              Class probabilities from input features

              Y_observed
              ndarray
              Ground truth observation from two classes


              model_name
              str
              Name of the algorithim

              is_save_fig
              boolean
              Boolean insturction on wheter to save the generated figure



    """

    ns_probs = [0 for _ in range(len(Y_observed))]

    # calculate scores
    lr_auc = roc_auc_score(Y_observed, Y_prob)

    # calculate roc curves

    ns_fpr, ns_tpr, _ = roc_curve(Y_observed, ns_probs, pos_label=1)
    lr_fpr, lr_tpr, _ = roc_curve(Y_observed, Y_prob, pos_label=1)

    # plot the roc curve for the model

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Chance')
    plt.plot(lr_fpr, lr_tpr, '-o',ms=5, label=model_name + ': $ROC_{auc}$ = ' + str(
        np.round(lr_auc,3)))

    plt.title('Receiver operating characteristic curve')
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate',  fontsize=15, fontweight='bold')
    plt.legend()
    if is_save_fig:
      plt.savefig('ROC Curve AdaBoost.png',dpi = 300)
    plt.show()


###############################################################################

def Draw_PR(Y_prob, Y_predicted,
            Y_observed,
            is_draw_dot = True,
            model_name = 'Model',
            is_save_fig = True):

    """Draw_PR is the utility function to plot precision and recall curve

      Input:

              Y_prob
              ndarray
              Class probabilities from input features

              Y_predicted
              ndarray
              Class predictions from input features

              Y_observed
              ndarray
              Ground truth observation from two classes

              is_draw_dot
              boolen
              Boolean insturction on wheter to draw dots in the PR curve


              model_name
              str
              Name of the algorithim

              is_save_fig
              boolean
              Boolean insturction on wheter to save the generated figure

    """

    # predict class values

    lr_precision, lr_recall, _ = precision_recall_curve(Y_observed,
                                                        Y_prob, pos_label=1)
    lr_f1, lr_auc = f1_score(Y_observed, Y_predicted), auc(lr_recall,
                                                           lr_precision)

    # plot the precision-recall curves

    no_skill = len(Y_observed[Y_observed==1]) / len(Y_observed)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Chance')
    if is_draw_dot:
      plt.plot(lr_recall, lr_precision, '-o' , ms=5,
             label=model_name + ': $f_{1}$ =%.3f, $PR_{auc}$=%.3f' % (lr_f1, lr_auc))

    else:
      plt.plot(lr_recall, lr_precision, '-', ms=8,
             label=model_name + ': $f_{1}$ =%.3f, $PR_{auc}$=%.3f' % (lr_f1, lr_auc))

    plt.title('2-class Precision-Recall curve')
    plt.xlabel('Recall', fontsize=15, fontweight='bold')
    plt.ylabel('Precision', fontsize=15, fontweight='bold')

    plt.legend()
    if is_save_fig:
      plt.savefig('PR AdaBoost.png',dpi = 300)
    plt.show()

###############################################################################

def Draw_PR_ROC(Y_prob, Y_predicted,
            Y_observed,
            is_draw_dot = True,
            model_name = 'Model',
            is_save_fig = True):

    """Draw_PR_ROC is the utility function to plot precision and recall curve
       in a subplot fashion

      Input:

              Y_prob
              ndarray
              Class probabilities from input features

              Y_predicted
              ndarray
              Class predictions from input features

              Y_observed
              ndarray
              Ground truth observation from two classes

              is_draw_dot
              boolen
              Boolean insturction on wheter to draw dots in the PR curve


              model_name
              str
              Name of the algorithim

              is_save_fig
              boolean
              Boolean insturction on wheter to save the generated figure

    """

    ns_probs = [0 for _ in range(len(Y_observed))]

    # calculate roc curves

    ns_fpr, ns_tpr, _ = roc_curve(Y_observed, ns_probs, pos_label=1)
    lr_fpr, lr_tpr, _ = roc_curve(Y_observed, Y_prob, pos_label=1)

     # predict class values

    lr_precision, lr_recall, _ = precision_recall_curve(Y_observed,
                                                        Y_prob, pos_label=1)
    lr_f1, lr_auc2 = f1_score(Y_observed, Y_predicted), auc(lr_recall,
                                                            lr_precision)
    no_skill = len(Y_observed[Y_observed==1]) / len(Y_observed)

    # constructure subplot

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Chance')
    ax1.set_xlabel('Recall', fontsize=23, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=23, fontweight='bold')
    if is_draw_dot:
      ax1.plot(lr_recall, lr_precision, '-o' , ms=8,
             label=model_name + ': $f_{1}$ =%.3f'%lr_f1)

    else:
      ax1.plot(lr_recall, lr_precision, '-', ms=8,
             label=model_name + ': $f_{1}$ =%.3f'%lr_f1)

    ax1.set_title('2-class Precision-Recall curve')
    ax1.legend(loc=10,prop={'size': 15})
    plt.subplots_adjust(hspace=0.6)

    ax2.plot(ns_fpr, ns_tpr, linestyle='--', label='Chance')
    ax2.plot(lr_fpr, lr_tpr, '-o',ms=8, label=model_name + ': $ROC_{auc}$ = ' + str(
        np.round(lr_auc2,3)))

    ax2.set_title('Receiver operating characteristic curve')
    ax2.set_xlabel('False Positive Rate', fontsize=23, fontweight='bold')
    ax2.set_ylabel('True Positive Rate',  fontsize=23, fontweight='bold')
    ax2.legend(prop={'size': 15})
    if is_save_fig:
      plt.savefig('PR_ROC AdaBoost.jpg',dpi = 300)
    plt.tight_layout()
    plt.show()
###############################################################################

def get_decision_ROC(classifier, X_test, y_test):

    """get_decision_ROC the utility function to plot precision and recall curve
       out of a 5-fold validation

      Input:

              classifer
              Classifer object

              X_test
              ndarray
              Features from the test set input

              y_test
              ndarray
              predictions from the test set

    """

    cv = StratifiedKFold(n_splits=5)
    fig = plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    y = label_binarize(y_test, classes=[0, 1])
    plt.figure()
    i = 0
    for train, test in cv.split(X_test, y_test):
        y_score = classifier.predict_proba(X_test[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], y_score[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3,
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
    plt.legend(loc="lower right")
    plt.savefig('5-fold-validation ROC AdaBoost.pdf',dpi=300)
    plt.show()
    return fig, aucs
