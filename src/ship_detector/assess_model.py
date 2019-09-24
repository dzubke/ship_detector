# standard libraries
from typing import Callable, Tuple
import time

# non-standard libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve


def count_time(*args, print_values: bool = True):
    """Not sure if this will work like this, but idealy this function that take in a function with all of the necessary arguments and would run the function
        inside of it and count the amount of time it toook to run the function

    """
    
    start_time = time.time()


    function_outputs = args # call the function

    stop_time = time.time()
    duration = stop_time - start_time
    
    if print_values: 
        print(f"The function took: {duration} seconds")

    return duration, function_outputs


def roc_assess(model_fit, Xinput: np.ndarray,  ylabel: np.ndarray, print_values: bool = True) -> None:
    """
    Parameters
    ----------
    model_fit: sklearn.model.fit
        the object object type it not entirely accurate because sklearn doesn't have a 'model' members. it has 'linear_model' and others, but the general
        ides is that the fit of a sklearn.model needs to be passed as the model_fit argument. 

    Xinput: a 2d np.ndarray
        the input pixel values

    ylabels: a 1d np.ndarray
        the true labels that the y_hat predictions will be compared to

    print_values: bool = True
        if True, the method will print the values of the confusion matrix. If false, it will not print the values. 

    """

    y_hat = model_fit.predict(Xinput)   # column vextor predicted y-values. shape = # samples x 0 

    # predicted_proba outputs the probability of the prediction being zero or one with shape = # samples x 2
    y_prob = model_fit.predict_proba(Xinput)[:,1]   # we only want the probability that y_hat is equal to one, so we only take the right column. 
    
    fpr, tpr, threshold  = roc_curve(ylabel, y_prob)   # roc_curve (Receiver Operating Characteristic) outputs the false positive rates and true positive rates

    roc_auc = auc(fpr, tpr) # area under the cure (auc) of the roc curve

    if print_values: 
        # all of the print statements below were used to help me understand what roc_curve() and auc() were doing
        # print(f"y_hat sample: {y_hat[:15]}, y_hat shape: {y_hat.shape}")
        # print(f"type: {type(y_prob)}, y_score shape: {y_prob.shape}, y_score: {y_prob[:15]}")
        # print(f"fpr type: {type(fpr)}, fpr shape: {fpr.shape}, fpr sample: {fpr}")
        # print(f"tpr type: {type(tpr)}, tpr shape: {tpr.shape}, tpr sample: {tpr}")
        # print(f"thres type: {type(threshold)}, shape: {threshold.shape}, sample: {threshold}")

        print(f"area under the roc cure: {roc_auc}")
    
        plt.figure()
        
        # Plotting our Baseline..
        plt.plot([0,1],[0,1])
        plt.plot(fpr,tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')


def F1score_assess(model_fit, Xinput: np.ndarray,  ylabel: np.ndarray, print_values: bool = True) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    model_fit: sklearn.model.fit
        the object object type it not entirely accurate because sklearn doesn't have a 'model' members. it has 'linear_model' and others, but the general
        ides is that the fit of a sklearn.model needs to be passed as the model_fit argument. 

    Xinput: a 2d np.ndarray
        the input pixel values

    ylabels: a 1d np.ndarray
        the true labels that the y_hat predictions will be compared to

    print_values: bool = True
        if True, the method will print the values of the confusion matrix. If false, it will not print the values. 

    """

    y_hat = model_fit.predict(Xinput)   # column vextor predicted y-values. shape = # samples x 0 

    tn, fp, fn, tp = confusion_matrix(ylabel, y_hat).ravel()  # calculates the confusion matrix, which is the unravelled matrix [[true neg, false pos], [false neg, true pos]]
    precision = tp/ (tp + fp)
    recall = tp/ (tp + fn)
    F1_score = 2* precision * recall / (precision + recall)

    if print_values: 
        print (f"F1 score: {F1_score}")
        print (f"Precision: {precision}")
        print (f"Recall: {recall}")

    return F1_score, precision, recall


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')