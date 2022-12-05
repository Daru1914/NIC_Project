import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import make_classification
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import math 
import pickle
from sklearn import metrics

features, label = make_classification(n_samples=1000, n_features=20, n_classes=4, random_state = 18,
                                     class_sep=2.0, n_informative=8)


X_train, X_test, y_train, y_test = train_test_split(features, label, 
                                            test_size = 0.2, random_state=42)


def train_tune(model, param_grid, filename, X = X_train, y = y_train):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid = GridSearchCV(model, param_grid, refit = True, verbose = 1,n_jobs=-1, cv=cv)
    grid.fit(X, np.ravel(y, order='C'))
    print(grid.best_params_)
    pickle.dump(grid.best_estimator_, open(filename, 'wb'))
    return model


def predict_with_saved_model(filename, X_test = X_test, y_test = y_test, metric = 'accuracy'):
    saved_model = pickle.load(open(filename, 'rb'))
    y_pred = saved_model.predict(X_test)
    if (metric == 'accuracy'):
        result = metrics.accuracy_score(y_test, y_pred)
    elif (metric == 'balanced_accuracy'):
        result = metrics.balanced_accuracy_score(y_test, y_pred)
    elif (metric == 'f1'):
        result = metrics.f1_score(y_test, y_pred)
    elif (metric == 'precision'):
        result = metrics.precision_score(y_test, y_pred)
    elif (metric == 'recall'):
        result = metrics.recall_score(y_test, y_pred)
    return result

score = predict_with_saved_model('knn_clf', metric = 'recall')
print(score)



