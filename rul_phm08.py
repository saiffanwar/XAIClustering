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

import multiprocessing
#import logging
#import threading
#import time
#import concurrent.futures

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
    with open('saved/PHM08_model.pck', 'wb') as file:
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

def run_clustering(model, x_test,y_pred, features, discrete_features, search_num, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold):
    print('Starting thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)
    GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, dataset='PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=False)

    GLE.multi_layer_clustering(search_num, discrete_features)
    print('finishing thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)


def compare_parameters(parameter_search):
#    sparsitys = []
#    coverages = []
#    starting_ks = []
#    neighbourhoods = []
    rmses = []
    results = []
    missing_results = []
    i=0
    sparsitys = {v: [] for v in parameter_search['sparsity']}
    coverages = {v: [] for v in parameter_search['coverage']}
    starting_ks = {v: [] for v in parameter_search['starting_k']}
    neihgbourhoods = {v: [] for v in parameter_search['neighbourhood']}

    for sparsity_threshold in parameter_search['sparsity']:
        for coverage_threshold in parameter_search['coverage']:
            for starting_k in parameter_search['starting_k']:
                for neighbourhood_threshold in parameter_search['neighbourhood']:

                    if not os.path.exists(f'saved/feature_ensembles/PHM08_feature_ensembles_full_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck'):
                        missing_results.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
                    else:
                        with open(f'saved/results/PHM08_results_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck', 'rb') as file:
                            model_predictions, llc_predictions, rmse = pck.load(file)
                            i+=1
#                        sparsitys.append(sparsity_threshold)
#                        coverages.append(coverage_threshold)
#                        starting_ks.append(starting_k)
#                        neighbourhoods.append(neighbourhood_threshold)
                        rmses.append(rmse)
                        results.append([[sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold], rmse])
    selected_params = {v: [] for v in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']}
    for fix1 in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']:
        for fix2 in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']:
            comparing_params = [t for t in ['sparsity', 'coverage', 'starting_k', 'neighbourhood'] if t not in [fix1, fix2]]
            if fix1 != fix2:
                for i in parameter_search[fix1]:
                    selected_params[fix1] = i
                    for j in parameter_search[fix2]:
                        selected_params[fix2] = j
                        results = []
                        for k in parameter_search[comparing_params[0]]:
                            row = []
                            selected_params[comparing_params[0]] = k
                            for l in parameter_search[comparing_params[1]]:
                                selected_params[comparing_params[1]] = l
                                with open(f"saved/results/PHM08_results_{selected_params['sparsity']}_{selected_params['coverage']}_{selected_params['starting_k']}_{selected_params['neighbourhood']}.pck", 'rb') as file:
                                    model_predictions, llc_predictions, rmse = pck.load(file)
                                    row.append(rmse)
                            results.append(row)

                        print(comparing_params[0], comparing_params[1], np.shape(np.array(results)))








    print(len(missing_results), i)
    with open('saved/missing_results.pck', 'wb') as file:
        pck.dump(missing_results, file)


if __name__ == '__main__':
    mode = sys.argv[2]


    parameter_search = {'1':{
                        'sparsity': [0],
                        'coverage': [0, 0.05, 0.5, 1],
                        'starting_k': [1,5,10],
                        'neighbourhood': [0.05, 0.1, 0.5],
                        },
#                        'sparsity': [0],
#                        'coverage': [0, 0.05, 0.5, 1],
#                        'starting_k': [5],
#                        'neighbourhood': [0.05],
#                        },
                        '2':{
                        'sparsity': [0.05],
                        'coverage': [0, 0.05, 0.5, 1],
                        'starting_k': [1,5,10],
                        'neighbourhood': [0.05, 0.1, 0.5],
                        },
                        '3':{
                        'sparsity': [0.5],
                        'coverage': [0, 0.05, 0.5, 1],
                        'starting_k': [1,5,10],
                        'neighbourhood': [0.05, 0.1, 0.5],
                        },
                        '4':{
                        'sparsity': [1],
                        'coverage': [0, 0.05, 0.5, 1],
                        'starting_k': [1,5,10],
                        'neighbourhood': [0.05, 0.1, 0.5],
                        }}
    parameter_search = {'sparsity': [0, 0.05, 0.5, 1],
                        'coverage': [0, 0.05, 0.5, 1],
                        'starting_k': [1,5,10],
                        'neighbourhood': [0.05, 0.1, 0.5],
                        }

    if mode == 'results':
        compare_parameters(parameter_search)
    elif mode == 'ensembles' or 'explain':
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

        parameter_search_list = []
        for sparsity_threshold in parameter_search['sparsity']:
            for coverage_threshold in parameter_search['coverage']:
                for starting_k in parameter_search['starting_k']:
                    for neighbourhood_threshold in parameter_search['neighbourhood']:
                        parameter_search_list.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
        num_nodes = 4
        tasks_per_node = len(parameter_search_list)/num_nodes
#        parameter_search_list = parameter_search_list[sys.argv[1]-1*tasks_per_node:sys.argv[1]*tasks_per_node]
        with open('saved/missing_results.pck', 'rb') as file:
            parameter_search_list = pck.load(file)
        print(parameter_search_list, len(parameter_search_list))

        for params in parameter_search_list:
            sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = params

            if mode == 'ensembles':
                process = multiprocessing.Process(target=run_clustering, args=(model, x_test, y_pred, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold))
                process.start()

            elif mode=='explain':
                #    instances = [2773, 4123, 2006, 3291, 902, 2173, 3043, 967, 883, 3187]
                instances = random.choices(np.arange(len(x_test)), k=10)
                if os.path.exists(f'saved/feature_ensembles/PHM08_feature_ensembles_full_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck'):
                    model_predictions = []
                    chilli_predictions = []
                    llc_predictions = []
#                try:
                    GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, dataset='PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)
                    for instance in instances:
                            print(f'--------- Instance  = {instance} ----------')
#                            _,_, chilli_prediction = chilli_explain(model, instance=instance)
                            llc_prediction, fig = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
                            model_predictions.append(y_pred[instance])
#                            chilli_predictions.append(chilli_prediction)
                            llc_predictions.append(llc_prediction)

                    with open(f'saved/results/PHM08_results_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck', 'wb') as file:
                        pck.dump([model_predictions, llc_predictions, mean_squared_error(model_predictions, llc_predictions, squared=False)], file)
                else:
                    missing_results.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
                    print('No file for: ', sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)





#


