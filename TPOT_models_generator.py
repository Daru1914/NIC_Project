from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

features, label = make_classification(n_samples=1000, n_features=20, n_classes=4, random_state = 18,
                                     class_sep=2.0, n_informative=8)


X_train, X_test, y_train, y_test = train_test_split(features, label, 
                                            test_size = 0.2, random_state=42)


pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_clf.py')


df = pd.read_excel('final_dataset.xlsx', index_col=0)
ord = OrdinalEncoder()
df = df.drop('name', axis=1)
df['code'] = ord.fit_transform(df[['code']])

Y = df['pop_growth']
X = df.drop('pop_growth', axis=1)


X_trn, X_tst, y_trn, y_tst = train_test_split(X, Y, test_size = 0.2, random_state=42)

pipeline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)

pipeline_optimizer.fit(X_trn, y_trn)
pipeline_optimizer.export('tpot_reg.py')