import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from skearn.metrics import mean_squared_error


def data_preprocessing():
    data = pd.read_csv('Data/PHM08/PHM08.csv')

    # Train and test split of the data with remaining RUL y values. 75/25 split.
    train, test = data[data['id'] <= 163], data[data['id'] >= 163]

    y_train, y_test = train['RUL'], test['RUL']
    x_train, x_test = train.drop(['RUL'], axis=1), test.drop(['RUL'], axis=1)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = data_preprocessing()
print(x_train.head())
def data_visualisation():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
    for col in data.columns:
        for i in data['id'].unique():
            fig, axes = plt.subplots(1, 1, figsize=(10, 4))
            axes.scatter('RUL', col, data=data[data['id'] == i], alpha=0.5,s=1)
            fig.savefig('Figures/PHM08/' + col + '.pdf')
data_visualisation()

def train():
    model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
    model.fit(x_train, y_train)

    return model

def evaluate(model):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: ', mse)

