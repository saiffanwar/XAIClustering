import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle as pck
from chilli import CHILLI
from LocalLinearRegression import LocalLinearRegression
from LinearClustering import LinearClustering
import os, glob


def data_preprocessing():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
    features = data.columns.tolist()
    features.remove('RUL')
    # Train and test split of the data with remaining RUL y values. 75/25 split.
    train, test = data[data['id'] <= 163], data[data['id'] > 163]

    y_train, y_test = train['RUL'], test['RUL']
    x_train, x_test = train.drop(['RUL'], axis=1), test.drop(['RUL'], axis=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test, features


x_train, x_test, y_train, y_test, features = data_preprocessing()
def data_visualisation():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
    for col in data.columns:
#        for i in data['id'].unique():
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes.scatter('RUL', col, data=data, alpha=0.5,s=1)
        fig.savefig('Figures/PHM08/' + col + '.pdf')
#data_visualisation()

def train():
    model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
    model.fit(x_train, y_train)

    return model

def evaluate(model):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    fig, ax = plt.subplots(13, 2, figsize=(10, 24))
    axes = fig.get_axes()
    for i in range(len(x_test[0])):
#        for i in data['id'].unique():
        col_xs = x_test[:, i]
        axes[i].scatter(col_xs, y_test, color='blue', alpha=0.5,s=1)
        axes[i].scatter(col_xs, y_pred, color='green', alpha=0.5,s=1)
        axes[i].set_ylabel(features[i])
    fig.savefig('Figures/PHM08/predictions.pdf')
    print('MSE: ', mse)

train_model = False
if train_model:
    model = train()
    with open('model.pck', 'wb') as file:
        pck.dump(model, file)
with open('model.pck', 'rb') as file:
    model = pck.load(file)
#evaluate(model)


def explain():
    chilliExplainer = CHILLI(model, x_train, y_train, x_test, y_test, features)
    explainer = chilliExplainer.build_explainer(mode='regression')
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = chilliExplainer.make_explanation(explainer, instance=35)
    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    with open('explanation.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)
#explain()


def plot_single_perturbation():
    with open('explanation.pck', 'rb') as file:
        exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = pck.load(file)
    perturbations = np.array(perturbations)
    plt.scatter(perturbations[:, features.index('s15')], model_perturbation_predictions, s=5)
    plt.show()

def make_linear_ensemble(xdata, ydata):


    print('Performing Local Linear Regression')
    # Perform LocalLinear regression on fetched data
    LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
    w1, w2, w, MSE = LLR.calculateLocalModels()
    xrange = np.linspace(min(xdata), max(xdata), 100)
    print('Calculating Distances')
#
    distance_weights = [1,0.75,0]

    D, xDs= LLR.compute_distance_matrix(w, MSE, distance_weights=distance_weights)
    print('Doing K-medoids-clustering')
    # Define number of medoids and perform K medoid clustering.

    LC = LinearClustering(xdata, ydata, D, xDs)

    K = 20

    files = glob.glob(f'Figures/Clustering/OptimisedClusters/*')
    for f in files:
        os.remove(f)
#    clustered_data, medoids, linear_params, clustering_cost = LLR.adapted_clustering(K,D, xDs)
    clustered_data, medoids, linear_params, clustering_cost = LC.adapted_clustering(K)


    return clustered_data, linear_params

with open('explanation.pck', 'rb') as file:
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = pck.load(file)
perturbations = np.array(perturbations)
xdata, ydata = perturbations[:, features.index('s14')], model_perturbation_predictions
make_linear_ensemble(xdata, ydata)
