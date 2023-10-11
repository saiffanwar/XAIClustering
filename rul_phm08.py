import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs.lib import generate_slices
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle as pck
from chilli import CHILLI
from LocalLinearRegression import LocalLinearRegression
from LinearClustering import LinearClustering
import os, glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import sys

plt.rcParams['text.usetex'] = True


def data_preprocessing():
    data = pd.read_csv('Data/PHM08/PHM08.csv')
#    features.remove('RUL')


    X = data.drop(['RUL', 'id'], axis=1)
    Y = data['RUL']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print(len(x_train), len(x_test))
    # Train and test split of the data with remaining RUL y values. 75/25 split.
#    train, test = data[data['id'] <= 163], data[data['id'] > 163]
#
#    y_train, y_test = train['RUL'], test['RUL']
#    x_train, x_test = train.drop(['RUL', 'id'], axis=1), test.drop(['RUL','id'], axis=1)
    features = x_train.columns.tolist()

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
    return y_pred


def explain(model, instance=25, automated_locality=True, newMethod=True, kernel_width=10):
    chilliExplainer = CHILLI(model, x_train, y_train, x_test, y_test, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = chilliExplainer.make_explanation(explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    with open('saved/explanation.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)

    return explanation_error

def plot_single_perturbation():
    with open('explanation.pck', 'rb') as file:
        exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = pck.load(file)
    perturbations = np.array(perturbations)
    plt.scatter(perturbations[:, features.index('s15')], model_perturbation_predictions, s=5)
    plt.show()

def make_linear_ensemble(x_test, y_pred):

    feature_ensembles = {}
    R = np.random.RandomState(42)
    random_samples = R.randint(2, len(x_test), 2000)
    x_test = x_test[random_samples]
    y_pred = y_pred[random_samples]



    discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
#    files = glob.glob(f'Figures/Clustering/OptimisedClusters/*')
#    for f in files:
#        os.remove(f)
    for i in tqdm(range(len(features))):
#    for i in tqdm(range(2)):

        if features[i] in discrete_features:
            K=1
        else:
            K=50
#    for i in tqdm(range(0,5)):
        feature = features[i]
        xdata = x_test[:, features.index(feature)]
        ydata = y_pred


        print('Performing Local Linear Regression')
        # Perform LocalLinear regression on fetched data
        LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
        w1, w2, w, MSE = LLR.calculateLocalModels()
        print('Calculating Distances')
#
        distance_weights = [1,1,0]
        D, xDs= LLR.compute_distance_matrix(w, MSE, distance_weights=distance_weights)
        print('Doing K-medoids-clustering')
        # Define number of medoids and perform K medoid clustering.

        LC = LinearClustering(xdata, ydata, D, xDs, features[i], K)

        clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()
        fig.savefig(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{len(clustered_data)}.pdf')


        cluster_x_ranges = [[min(clustered_data[i][0]), max(clustered_data[i][0])] for i in range(len(clustered_data))]
        feature_ensembles[features[i]] = [clustered_data, linear_params, cluster_x_ranges]

        with open(f'saved/feature_ensembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pck', 'wb') as file:
            pck.dump(feature_ensembles, file)

    return feature_ensembles


def plot_ensembles(K=10, distance_weights=[1,1,0]):
    with open(f'saved/feature_ensembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pck', 'rb') as file:
        feature_ensembles = pck.load(file)

    fig, axes = plt.subplots(13, 2, figsize=(10, 28))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.4)
    ax = fig.get_axes()
    colours = [np.random.rand(1,3) for i in range(20)]

    for f, key in enumerate(feature_ensembles.keys()):
        clustered_data, linear_params, cluster_x_ranges = feature_ensembles[key]
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
#            colours.append(colour)
            colour = colours[i]
            ax[f].scatter(clustered_data[i][0], clustered_data[i][1], s=1, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
            ax[f].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -20, 20, color=colour, label='_nolegend_')
            ax[f].plot(cluster_range, w*cluster_range+b, linewidth=1, c=colour)

        ax[f].set_title(key)
#        try:
#            ax[f].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2)
#        except:
#            pass
    fig.savefig(f'Figures/Clustering/AllFeatureEnsembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pdf', bbox_inches='tight')

train_model = False
if train_model:
    model = train()
    with open('saved/model.pck', 'wb') as file:
        pck.dump(model, file)
with open('saved/model.pck', 'rb') as file:
    model = pck.load(file)

y_pred = evaluate(model)
make_linear_ensemble(x_test, y_pred)
plot_ensembles()

kernel_widths = [0.1, 0.25, 0.5, 0.75, 1, 5]

if sys.argv[1] == 'genexp':
    instances = random.sample(range(0, len(x_test)), 15)
#    instances = [11198, 8640, 4571, 1955, 4335, 2851, 7010, 1965, 10964, 653]
#    instances = instances[:3]
    kernel_widths = [5]
    instances = [8646]
    for kernel_width in kernel_widths:
        results = {instance: [] for instance in instances}
        for instance in instances:
#            try:
                print(f'-------------------{instance}-------------------')
                for newMethod in [True, False]:
                    for automated_locality in [True, False]:
                        explanation_error = explain(model, instance=instance, automated_locality=automated_locality, newMethod=newMethod, kernel_width=kernel_width)
                        results[instance].append(explanation_error)
                        print(f'Automated locality: {automated_locality}, newMethod: {newMethod}, explanation error: {explanation_error}')
                print('-----------------------------------------------\n')
#            except:
#                pass
#        with open(f'saved/results_{kernel_width}.pck', 'wb') as file:
#            pck.dump(results, file)
#
#explanation_error = explain(model, instance=1000, automated_locality=False, newMethod=True)

#with open('saved/explanation.pck', 'rb') as file:
#    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = pck.load(file)

def plot_results(kernel_widths=kernel_widths):
    colours = ['r', 'b', 'g', 'black']
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.4)
    ax = fig.get_axes()
    for i, kernel_width in enumerate(kernel_widths):
        with open(f'saved/results_{kernel_width}.pck', 'rb') as file:
            results = pck.load(file)
        new_results = {}
        print(results.keys())
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

if sys.argv[1] == 'plot':
    print('plotting')
    plot_results()

