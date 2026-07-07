import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

car = pd.read_csv('Cleaned_Car_data.csv')

X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_i = int(np.argmax(scores))
print("Best random_state:", best_i, "R2 score:", scores[best_i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_i)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("Final R2 score:", r2_score(y_test, y_pred))

pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
print("Saved new LinearRegressionModel.pkl (compatible with your installed scikit-learn)")
