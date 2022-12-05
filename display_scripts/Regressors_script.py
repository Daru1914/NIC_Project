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
from sklearn import metrics
import math 
import pickle

df = pd.read_excel('final_dataset.xlsx', index_col=0)
ord = OrdinalEncoder()
df = df.drop('name', axis=1)
df['code'] = ord.fit_transform(df[['code']])

label = df[['pop_growth']]
features = df.drop('pop_growth', axis=1)


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state=42)


def train_tune(model, param_grid, filename, X = X_train, y = y_train):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid = GridSearchCV(model, param_grid, refit = True, verbose = 1,n_jobs=-1, cv=cv)
    grid.fit(X, np.ravel(y, order='C'))
    print(grid.best_params_)
    pickle.dump(grid.best_estimator_, open(filename, 'wb'))
    return model


def predict_with_saved_model(filename, X_test = X_test, y_test = y_test, metric = 'mean_squared_error'):
    saved_model = pickle.load(open(filename, 'rb'))
    y_pred = saved_model.predict(X_test)
    if (metric == 'mean_squared_error'):
        result = metrics.mean_squared_error(y_test, y_pred)
    elif (metric == 'mean_absolute_error'):
        result = metrics.mean_absolute_error(y_test, y_pred)
    elif (metric == 'r2'):
        result = metrics.r2_score(y_test, y_pred)
    elif (metric == 'mean_absolute_percentage_error'):
        result = metrics.mean_absolute_percentage_error(y_test, y_pred)
    return result


score = predict_with_saved_model('ransac_reg', metric = 'mean_squared_error')
print(math.sqrt(score))