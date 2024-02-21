import numpy as np
import math
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
from tqdm import tqdm

import multiprocessing
#import logging
#import threading
import time
#import concurrent.futures

plt.rcParams['text.usetex'] = True

def data_preprocessing(dataset='PHM08'):

    if dataset=='PHM08':
        data = pd.read_csv('Data/PHM08/PHM08.csv')

        features = data.drop(['RUL','cycle', 'id'], axis=1).columns.tolist()

        # ----------------------------------------------------------
        # RERUN ALL EXPERIMENTS USING THE FOLLOWING DATA SPLIT METHOD

        train_df = data[data['id'] <= 150]
        test_df = data[data['id'] > 150]

        scaler = StandardScaler()
#        x_train = scaler.fit_transform(x_train)
#        x_test = scaler.fit_transform(x_test)
        x_train = scaler.fit_transform(train_df.drop(['RUL','cycle', 'id'], axis=1).values)
        x_test = scaler.fit_transform(test_df.drop(['RUL', 'cycle','id'], axis=1).values)
        y_train = train_df['RUL'].values
        y_test = test_df['RUL'].values

        # ----------------------------------------------------------

#        X = data.drop(['RUL', 'id'], axis=1)
#        Y = data['RUL']
##
#        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#
#        scaler = StandardScaler()
#        x_train = scaler.fit_transform(x_train)
#        x_test = scaler.fit_transform(x_test)
#        y_train = y_train.values
#        y_test = y_test.values

    if dataset == 'CMAPS':
        with open('processed_cmaps.pck', 'rb') as f:
            x_train, y_train, x_test, y_test, features = pck.load(f)

    return x_train, x_test, y_train, y_test, features

def data_visualisation():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
    for col in data.columns:
#        for i in data['id'].unique():
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes.scatter(col, 'RUL', data=data, alpha=0.5,s=1)
        fig.savefig('Figures/PHM08/' + col + '.png')
#data_visualisation()

def train(x_train, y_train, feature_names):
    model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
    model.fit(x_train, y_train)
    with open('saved/PHM08_model.pck', 'wb') as file:
            pck.dump(model, file)
#    feature_importance = model.feature_importances_
#    sorted_idx = np.argsort(feature_importance)
#    pos = np.arange(sorted_idx.shape[0]) + 0.5
#    fig = plt.figure(figsize=(12, 6))
#    plt.subplot(1, 2, 1)
#    plt.barh(pos, feature_importance[sorted_idx], align="center")
#    plt.yticks(pos, np.array(feature_names)[sorted_idx])
#    plt.title("Feature Importance (MDI)")
#
#    fig.tight_layout()
#    plt.show()
    return model

def evaluate(model, X_train, X_test, y_train, y_test):
    y_hat_train = model.predict(X_train)
    print('training RMSE: ',mean_squared_error(y_train, y_hat_train),)
    y_hat_test = model.predict(X_test)
    print('test RMSE: ',mean_squared_error(y_test, y_hat_test))

    return y_hat_train, y_hat_test

def chilli_explain(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=25, automated_locality=False, newMethod=True, kernel_width=0.1, noisey_instance=None, categorical_features=None, neighbours=None):
    chilliExplainer = CHILLI('PHM08', model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction = chilliExplainer.make_explanation(model, explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL', neighbours=neighbours)
    with open(f'saved/explanation_{instance}.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)
    plotting_data = [x_test[instance], instance, perturbations, model_perturbation_predictions, y_test[instance], y_train[instance], exp_instance_prediction, exp_perturbation_predictions, exp]

    return explanation_error, instance_prediction, exp_instance_prediction, plotting_data


def run_clustering(model, x_test,y_pred, dataset, features, discrete_features, search_num, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold, preload_explainer=False):
    print('Starting thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)
    GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=preload_explainer)
    if preload_explainer:
        GLE.plot_all_clustering()
    else:
        GLE.multi_layer_clustering(search_num, discrete_features)
    print('finishing thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)





if __name__ == '__main__':
    mode = sys.argv[2]

    dataset = 'PHM08'
    x_train, x_test, y_train, y_test, features = data_preprocessing(dataset)

    discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    discrete_features = features
    categorical_features = [features.index(feature) for feature in discrete_features]
    with open(f'saved/models/{dataset}_model.pck', 'rb') as file:
        model = pck.load(file)

    R = np.random.RandomState(42)
    random_samples = R.randint(2, len(x_test), 5000)

    x_train = x_train[random_samples]
    y_train = y_train[random_samples]
    tic = time.time()
#        model = train(x_train, y_train, features)
    y_train_pred, y_test_pred = evaluate(model, x_train, x_test, y_train, y_test)
    toc = time.time()
    print('Time taken to train and evaluate model: ', toc-tic)

    x_test = x_test[random_samples]
    y_pred = y_test_pred[random_samples]
    y_test = y_test[random_samples]
    print(f'Training samples: {len(x_train)}')
    print(f'Test samples: {len(x_test)}')

    # -------- PARAMETER SEARCH -------- #
#
#        parameter_search_list = []
#        for sparsity_threshold in parameter_search['sparsity']:
#            for coverage_threshold in parameter_search['coverage']:
#                for starting_k in parameter_search['starting_k']:
#                    for neighbourhood_threshold in parameter_search['neighbourhood']:
#                        parameter_search_list.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
#        num_nodes = 4
#        tasks_per_node = len(parameter_search_list)/num_nodes
##        parameter_search_list = parameter_search_list[sys.argv[1]-1*tasks_per_node:sys.argv[1]*tasks_per_node]

    # ---------------------------------- #

    # Write list of LLC parameters here
    # [sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold]
    parameter_search_list = [[0.05, 0.1, 10, 0.1]]
#        parameter_search_list = [[0.5, 0.05, 5, 0.5]]

    for params in parameter_search_list:
        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = params

        if mode == 'ensembles':

            # If doing parameter search then use multiprocessing

#                process = multiprocessing.Process(target=run_clustering, args=(model, x_test, y_pred, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold))
#                process.start()

            run_clustering(model, x_test, y_pred, dataset, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)

        elif mode=='explain':
            GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, discrete_features=discrete_features, dataset='PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)
#                instances = [2773, 4123, 2006, 3291, 902, 2173, 3043, 967, 883, 3187]

#                instances = [random.randint(0, len(x_test)) for i in range(1)]
            instances = [4444]
#                [4444, 3012, 3492, 1955, 1206, 4858, 2120, 1717, 4470, 958, 3446, 4831, 4403, 4527, 4519, 2576, 4613, 3377, 2067, 548, 347, 1397, 1894, 479, 2663]

            instance = instances[0]

            if sys.argv[3] == 'similar':
                llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])

                data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data

                relevant_instances = GLE.new_evaluation(instance, instance_explanation_model, x_test[instance], importance_threshold = 1e10)
                instances = relevant_instances[:10]
#                distances = np.linalg.norm(x_test-x_test[instance], axis=1)
##                distances = combinedFeatureDistances(calcAllDistances(x_test[instance], x_test, features))
#
#                closest_instances = np.argsort(distances)[:10]
#                instances = closest_instances
            elif sys.argv[3] == 'same':
                instances = [instance for i in range(10)]




            # ---- Similar Instances ----
#                instances24matches = [3325, 68, 81, 92, 206, 253, 254, 302, 350, 425, 499, 621, 660, 669, 786, 846, 962, 975, 1011, 1012, 1076, 1339, 1584, 1604, 1614, 1636, 1656, 1677, 1930, 2131, 2132, 2246, 2295, 2335, 2343, 2723, 2997, 3116, 3146, 3156, 3176, 3286, 3297, 3325, 3413, 3419, 3421, 3495, 3509, 3523, 3566, 3581, 3582, 3589, 3642, 3817, 3958, 3992, 4141, 4165, 4247, 4282, 4285, 4314, 4334, 4354, 4397, 4409, 4417, 4668, 4693, 4735, 4851, 4953, 498][:10]
#                instances23matches = [3325, 1, 3, 29, 51, 55, 67, 68, 81, 83, 92, 94, 111, 117, 153, 178, 179, 192, 198]

            # ---- Same Instances ----
#                instances = [100 for i in range(10)]

            kw = 0.1
#                    chilli_predictions = {kernel_width: [] for kernel_width in kernel_widths}
#                for k in kernel_widths:
            model_predictions = []

            lime_predictions = []
            lime_models = []
            chilli_predictions = []
            chilli_models = []
            llc_predictions = []
            llc_explanation_models = []

            similar_explanation_data = []
            chilli_exp = True
            lime_exp = True
            chilli_deviations = []
            for instance in tqdm(instances):
                    GLE.plot_all_clustering()
                    print(f'################# Instance  = {instance} ###################')

#                        distances = [math.dist(x_test[instance], x) for x in x_test]
#                        print('Min Distance: ', min(distances))
#                        closest_instances = np.argsort(distances)[1:11]

                    # ------ BASE MODEL ------
                    print(f'Ground Truth: {y_test[instance]}')
                    print(f'Model Prediction: {y_pred[instance]}')

                    model_predictions.append(y_pred[instance])

                    # ---- LIME EXPLANATION -------
                    if chilli_exp:
                        _,_, lime_prediction, lime_plotting_data = chilli_explain(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=instance, newMethod=False, kernel_width=kw, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, lime_exp = lime_plotting_data

                        lime_predictions.append(exp_instance_prediction)
                        lime_models.append(lime_exp.as_list())

                    # ---- CHILLI EXPLANATION -------
                    if chilli_exp:
                        _,_, chilli_prediction, chilli_plotting_data = chilli_explain(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=instance, newMethod=True, kernel_width=kw, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, chilli_exp = chilli_plotting_data
                        chilli_predictions.append(exp_instance_prediction)
                        chilli_models.append(chilli_exp.as_list())

                    # ---- LLC EXPLANATION -------
                    llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])

                    data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data

                    GLE.interactive_exp_plot(data_instance, instance_index, instance_prediction, y_test_pred, exp_instance_prediction, instance_explanation_model.coef_, local_x, local_x_weights, local_y_pred, exp_local_y_pred, 'RUL')

                    llc_predictions.append(llc_prediction)
                    print(instance_explanation_model.coef_)
                    llc_explanation_models.append(instance_explanation_model.coef_)

#                        similar_explanation_data.append([lime_exp.as_list(), chilli_exp.as_list(), instance_explanation_model.coef_])

        with open(f'saved/results/{dataset}_{sys.argv[3]}_instances_kw={kw}.pck', 'wb') as f:
            pck.dump([[lime_predictions, lime_models], [chilli_predictions, chilli_models], [llc_predictions, llc_explanation_models], instances, model_predictions ], f)

#                with open('saved/results/same_instances.pck', 'wb') as f:
#                    pck.dump(similar_explanation_data, f)



#


