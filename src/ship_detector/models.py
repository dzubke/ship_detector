# standard libraries
from typing import Callable

# non-standard libraries
from sklearn.linear_model import LogisticRegression
import numpy as np


# def evaluate_model(model: Callable, xtrain: np.ndarray, ytrain: np.ndarray, xtest: np.ndarray, ytest: np.ndarray):

def run_model(model: Callable, xtrain: np.ndarray, xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray):
    """

    Parameters
    ----------


    Returns
    -------


    """

    model_fit = model.fit(xtrain, ytrain)

    train_acc=model_fit.score(xtrain, ytrain)
    test_acc=model_fit.score(xtest,ytest)
    print("Training Data Accuracy: %0.2f" %(train_acc))
    print("Test Data Accuracy:     %0.2f" %(test_acc))

    # return LR_Fit

    # lr1 = test_model(lr, xtrain, ytrain, xtest, ytest)



