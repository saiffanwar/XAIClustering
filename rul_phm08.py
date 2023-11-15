import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs.lib import generate_slices
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
import pickle as pck
from chilli import CHILLI
from LocalLinearRegression import LocalLinearRegression
from LinearClustering import LinearClustering
import os, glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import sys
from pprint import pprint
#import jenkspy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from run_linear_clustering import GlobalLinearExplainer

plt.rcParams['text.usetex'] = True

def data_preprocessing():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
#    features.remove('RUL')


    X = data.drop(['RUL', 'id'], axis=1)
    Y = data['RUL']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    features = x_train.columns.tolist()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test, features

def data_visualisation():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
    for col in data.columns:
#        for i in data['id'].unique():
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes.scatter(col, 'RUL', data=data, alpha=0.5,s=1)
        fig.savefig('Figures/PHM08/' + col + '.png')
#data_visualisation()

def train(x_train, y_train):
    model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
    model.fit(x_train, y_train)
    with open('saved/model.pck', 'wb') as file:
            pck.dump(model, file)
    return model

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: ', mse)
    return y_pred

def chilli_explain(model, instance=25, automated_locality=False, newMethod=True, kernel_width=5):
    chilliExplainer = CHILLI(model, x_train, y_train, x_test, y_test, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction = chilliExplainer.make_explanation(explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    with open(f'saved/explanation_{instance}.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)

    return explanation_error, instance_prediction, exp_instance_prediction

def plot_results(kernel_widths=None):
    colours = ['r', 'b', 'g', 'black']
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.4)
    ax = fig.get_axes()
    for i, kernel_width in enumerate(kernel_widths):
        with open(f'saved/results_{kernel_width}.pck', 'rb') as file:
            results = pck.load(file)
        new_results = {}
        for k,v in results.items():
            if v != []:
                new_results[k] = v
        results = new_results
        for j in range(3):
            ys = [results[instance][j] for instance in results.keys() if results[instance] != []]
            if j == 0:
                avg_newchilli_error = np.mean(ys)
            if j == 1:
                avg_chilli_error = np.mean(ys)
            if j ==2:
                avg_lime_error = np.mean(ys)
            if max(ys) > 1000:
                ax[i].set_yscale('log')
            xs = [x for x in range(len(ys))]
            ax[i].scatter(xs, ys, c=colours[j], s=20, marker='x')
        ax[i].vlines(xs,[results[instance][0] for instance in results.keys()], [results[instance][1] for instance in results.keys()], color='gray', label='_nolegend_')
        ax[i].set_title(r'$\sigma$ = '+str(kernel_width)+f'\n New CHILLI Average MSE: {avg_newchilli_error:.2f}, \n CHILLI Average MSE: {avg_chilli_error:.2f}, \n LIME Average MSE: {avg_lime_error:.2f}')
        ax[i].set_xlabel('Instance')
        ax[i].set_ylabel('Explanation error (MSE)')
    fig.legend(['CHILLI with automated locality', 'CHILLI without automated locality', 'LIME'],loc='center', bbox_to_anchor=(0.5,0.98), ncols=3)
    fig.savefig('Figures/Results.pdf', bbox_inches='tight')


def plot_results2():
    with open(f'saved/PHM08_results.pck', 'rb') as file:
        results = pck.load(file)
    model_predictions, chilli_predictions, llc_predictions = results[0], results[1], results[2]
    fig, axes = plt.subplots(1,1, figsize=(8, 4))
    for instance in range(len(model_predictions)):
        axes.scatter([instance], [model_predictions[instance]], c='r', s=30, marker='x')
        axes.scatter([instance], [chilli_predictions[instance]], c='b', s=20, marker='x')
        axes.scatter([instance], [llc_predictions[instance]], c='g', s=20, marker='x')

        axes.vlines([instance],[min(model_predictions[instance], chilli_predictions[instance], llc_predictions[instance])], [max(model_predictions[instance], chilli_predictions[instance], llc_predictions[instance])], color='gray', label='_nolegend_')
    fig.legend(['Model predictions', 'CHILLI predictions', 'LLC predictions'],loc='center', bbox_to_anchor=(0.5,0.98), ncols=3)
    fig.savefig('Figures/Results.pdf', bbox_inches='tight')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, features = data_preprocessing()

    discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    with open(f'saved/PHM08_model.pck', 'rb') as file:
        model = pck.load(file)

    R = np.random.RandomState(42)
    random_samples = R.randint(2, len(x_test), 5000)

    x_train = x_train[random_samples]
    y_train = y_train[random_samples]
#    model = train(x_train, y_train)
    y_pred = evaluate(model, x_test, y_test)

    x_test = x_test[random_samples]
    y_pred = y_pred[random_samples]
    y_test = y_test[random_samples]
#    GLE.plot_all_clustering()
#    GLE.multi_layer_clustering(discrete_features)
#    instance = 100
    instances = [100]

#    parameter_search = {
#                        'sparsity': [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
#                        'coverage': [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
#                        'starting_k': [1,5,10,20],
#                        'neighbourhood': [0.01, 0.05, 0.1, 0.25, 0.5, 1],
#                        }

    parameter_search = {
                        'sparsity': [0, 0.05, 0.1, 0.25, 0.5, 1],
                        'coverage': [0, 0.05, 0.1, 0.25, 0.5, 1],
                        'starting_k': [1,5,10,20],
                        'neighbourhood': [0.01, 0.05, 0.1, 0.25, 0.5, 1],
                        }
    model_predictions = []
    chilli_predictions = []
    llc_predictions = []

#    instances = random.sample(range(len(x_test)), 10)
    instances = [4292, 4942, 3164, 2133, 4468, 2858, 4789, 2266, 3833, 873]
    for sparsity_threshold in parameter_search['sparsity']:
        for coverage_threshold in parameter_search['coverage']:
            for starting_k in parameter_search['starting_k']:
                for neighbourhood_threshold in parameter_search['neighbourhood']:
                    print('-----------------')
                    print(f'Sparsity threshold = {sparsity_threshold}')
                    print(f'Coverage threshold = {coverage_threshold}')
                    print(f'Starting k = {starting_k}')
                    print(f'Neighbourhood threshold = {neighbourhood_threshold}')

                    GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, dataset='PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)

                    GLE.multi_layer_clustering(discrete_features)
                    for instance in instances:
                        try:
                            print(f'--------- Instance  = {instance} ----------')
                            _,_, chilli_prediction = chilli_explain(model, instance=instance)
                            llc_prediction, fig = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
                            model_predictions.append(y_pred[instance])
                            chilli_predictions.append(chilli_prediction)
                            llc_predictions.append(llc_prediction)

                        except:
                            pass
                        with open(f'saved/PHM08_results_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck', 'wb') as file:
                            pck.dump([model_predictions, chilli_predictions, llc_predictions], file)
#    with open('saved/PHM08_results.pck', 'wb') as file:
#        pck.dump([model_predictions, chilli_predictions, llc_predictions], file)
#    plot_results2()


#    with open('saved/feature_ensembles.pck', 'rb') as file:
#        feature_ensembles = pck.load(file)
#    plot_all_clustering(feature_ensembles)






