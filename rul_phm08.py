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
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly

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
        axes.scatter('RUL', col, data=data, alpha=0.5,s=1)
        fig.savefig('Figures/PHM08/' + col + '.pdf')
#data_visualisation()

def train():
    model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
    model.fit(x_train, y_train)
    with open('saved/model.pck', 'wb') as file:
            pck.dump(model, file)
    return model

def evaluate(model):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: ', mse)
    return y_pred

def chilli_explain(model, instance=25, automated_locality=True, newMethod=True, kernel_width=10):
    chilliExplainer = CHILLI(model, x_train, y_train, x_test, y_test, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = chilliExplainer.make_explanation(explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    with open(f'saved/explanation_{instance}.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)

    return explanation_error

def plot_ensembles(K=20, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}):
    with open(f'saved/feature_ensembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pck', 'rb') as file:
        feature_ensembles, y_pred = pck.load(file)

    fig, axes = plt.subplots(13, 2, figsize=(10, 28))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.4)
    ax = fig.get_axes()
    colours = [np.random.rand(1,3) for i in range(35)]

    for f, key in enumerate(feature_ensembles.keys()):
        clustered_data, linear_params, cluster_x_ranges = feature_ensembles[key]
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
#            colours.append(colour)
            colour = colours[i]
            ax[f].scatter(clustered_data[i][0], clustered_data[i][1], s=0.5, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
            ax[f].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -20, 20, color=colour, label='_nolegend_')
            ax[f].plot(cluster_range, w*cluster_range+b, linewidth=1, c='black')

        ax[f].set_title(key)
#        try:
#            ax[f].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2)
#        except:
#            pass
    fig.savefig(f'Figures/Clustering/AllFeatureEnsembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pdf', bbox_inches='tight')


def plot_results(kernel_widths=None):
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




def linear_clustering(xdata, ydata, feature, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
, K=20):
    print('Performing Local Linear Regression')
    # Perform LocalLinear regression on fetched data
    print(type(xdata))
    LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
    w1, w2, w = LLR.calculateLocalModels()
    print('Calculating Distances')
    #

    D, xDs= LLR.compute_distance_matrix(w, distance_weights=distance_weights)
    print('Doing K-medoids-clustering')
    # Define number of medoids and perform K medoid clustering.

    LC = LinearClustering(xdata, ydata, D, xDs, feature, K,
                          sparsity_threshold = 0.01,
                          coverage_threshold=0.01*abs(max(xdata)-min(xdata)),
                          gaps_threshold=100,
                          similarity_threshold=0.1)



    clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()
#    fig.savefig(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{len(clustered_data)}.pdf')

    return clustered_data, medoids, linear_params, clustering_cost

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
    distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
    for i in tqdm(range(len(features))):
        #    for i in tqdm(range(1,18)):
        if features[i] in discrete_features:
            K=1
        else:
            K=20
        #    for i in tqdm(range(0,5)):
        feature = features[i]
        xdata = x_test[:, features.index(feature)]
        print(xdata)
        ydata = y_pred


        clustered_data, medoids, linear_params, clustering_cost = linear_clustering(xdata, y_pred, feature, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}, K=K)
        cluster_x_ranges = [[min(clustered_data[i][0]), max(clustered_data[i][0])] for i in range(len(clustered_data))]
        feature_ensembles[features[i]] = [clustered_data, linear_params, cluster_x_ranges]

        with open(f'saved/feature_ensembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pck', 'wb') as file:
            pck.dump([feature_ensembles, ydata], file)

    return feature_ensembles


def feature_space_clustering(xdata, ydata):
    data = xdata
    # Create a kernel density estimation model
    kde = KernelDensity(bandwidth=0.1*max(xdata))  # You can adjust the bandwidth
    kde.fit(np.array(data).reshape(-1, 1))

    # Create a range of data points for evaluation
    x_eval = np.linspace(min(data), max(data), 1000)
    log_dens = kde.score_samples(x_eval.reshape(-1, 1))
    dens = np.exp(log_dens)

    # Find local maxima in the density curve
    peaks, _ = find_peaks(dens)
    cluster_centers = x_eval[peaks]

    # Assign data points to clusters based on proximity to cluster centers
    print(len(cluster_centers))
    cluster_assignments = []
    for data_point in data:
        distances = np.abs(cluster_centers - data_point)
        nearest_cluster = np.argmin(distances)
        cluster_assignments.append(nearest_cluster)

    return cluster_assignments

def main():
        #model = train()
    with open('saved/model.pck', 'rb') as file:
        model = pck.load(file)

    y_pred = evaluate(model)
#    evaluate_LLC_explanation(x_test,y_pred, 100)

    make_linear_ensemble(x_test, y_pred)
    plot_ensembles()

    kernel_widths = [0.1, 0.25, 0.5, 0.75, 1, 5]

    if sys.argv[1] == 'genexp':
        instances = random.sample(range(0, len(x_test)), 15)
#    instances = [11198, 8640, 4571, 1955, 4335, 2851, 7010, 1965, 10964, 653]
        kernel_widths = [5]
        instances = [100]
        for kernel_width in kernel_widths:
            results = {instance: [] for instance in instances}
            for instance in instances:
#            try:
                    print(f'-------------------{instance}-------------------')
                    for newMethod in [True]:
                        explanation_error = explain(model, instance=instance, automated_locality=True, newMethod=newMethod, kernel_width=kernel_width)
                        results[instance].append(explanation_error)
                        print(f'Automated locality: {False}, newMethod: {newMethod}, explanation error: {explanation_error}')
                    print('-----------------------------------------------\n')
#            except:
#                pass
#        with open(f'saved/results_{kernel_width}.pck', 'wb') as file:
#            pck.dump(results, file)
    if sys.argv[1] == 'plot':
        print('plotting')
        plot_results()
#

def plot_final_clustering(clustered_data, linear_params):
#    cost = self.calculate_clustering_cost(clustered_data)
    fig = go.Figure()
    for cluster in range(len(clustered_data)):
        colour = np.random.rand(3)
        colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
        xs, ys = clustered_data[cluster]
        w, b = linear_params[cluster]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=4, color=colour)))
        cluster_range = np.linspace(min(xs), max(xs), 100)
        fig.add_trace(go.Scatter(x=cluster_range, y=w*cluster_range+b, mode='lines', line=dict(color=colour, width=5)))
#    fig.update_layout(title=f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {cost:.2f}')
    return fig

def plot_all_clustering(feature_ensembles):
    num_rows=int(np.ceil(len(features)/4))
    num_cols=4

    fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[1/num_rows for row in range(num_rows)], specs = [[{}, {}, {}, {}] for i in range(num_rows)], subplot_titles=features, horizontal_spacing=0.05, vertical_spacing=0.05)

    axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]

    for feature, value in feature_ensembles.items():
        i = features.index(feature)
        clustered_data, linear_params = value
#        if i==0:
#            showlegend=True
#        else:
        showlegend=False
        for cluster, params in zip(clustered_data, linear_params):
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
            fig.add_trace(go.Scatter(x=cluster[0],y=cluster[1],
                                     mode='markers', marker = dict(size=3, opacity=0.9, color=colour),
                                     showlegend=showlegend),
                          row=axes[i][0], col=axes[i][1])

            fig.add_trace(go.Scatter(x=cluster[0],y=[params[0]*x+params[1] for x in cluster[0]],
                                     mode='lines', marker = dict(size=3, opacity=0.9, color=colour),
                                     showlegend=showlegend),
                          row=axes[i][0], col=axes[i][1])

    fig.update_layout(legend=dict(yanchor="top", y=1.1, xanchor="right"),
                      height=350*num_rows, )
    fig.write_html(f'Figures/PHM08/all_feature_clustering.html', auto_open=False)

def multi_layer_clustering(x_test):

    with open('saved/model.pck', 'rb') as file:
        model = pck.load(file)
    y_pred = evaluate(model)

    R = np.random.RandomState(42)
    random_samples = R.randint(2, len(x_test), 5000)
    x_test = x_test[random_samples]
    y_pred = y_pred[random_samples]
    feature_ensembles = {feature: [] for feature in features}

    discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    # XDs, WDs, neighbourhoodDs
    distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
    fig, ax = plt.subplots(13, 2, figsize=(10, 24))
    axes = fig.get_axes()
    for i in range(len(features)):
        print(f'--------{features[i]}---------')
        all_feature_clusters = []
        all_feature_linear_params = []
        feature_xs = x_test[:, i]
        if features[i] not in discrete_features:
            cluster_assignments = feature_space_clustering(feature_xs, y_pred)
            K=5
        else:
            cluster_assignments = np.zeros(len(feature_xs))
            K=1

#            print(feature_xs)
        for super_cluster in np.unique(cluster_assignments):
            print(f'--------{super_cluster} out of {len(np.unique(cluster_assignments))}---------')
            super_cluster_x_indices = np.array(np.argwhere(cluster_assignments == super_cluster)).flatten()
            super_cluster_xs = feature_xs[super_cluster_x_indices]
            super_cluster_y_pred = y_pred[super_cluster_x_indices]

            clustered_data, medoids, linear_params, clustering_cost = linear_clustering(super_cluster_xs, super_cluster_y_pred, features[i], distance_weights, K=K)
            [all_feature_clusters.append(cluster) for cluster in clustered_data]
            [all_feature_linear_params.append(linear_param) for linear_param in linear_params]

        fig = plot_final_clustering(all_feature_clusters, all_feature_linear_params)
        fig.write_html(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{K}.html')
        feature_ensembles[features[i]] = [all_feature_clusters, all_feature_linear_params]
        with open(f'saved/feature_ensembles.pck', 'wb') as file:
            pck.dump(feature_ensembles, file)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, features = data_preprocessing()
    #    main()
#    multi_layer_clustering(x_test)
    with open('saved/feature_ensembles.pck', 'rb') as file:
        feature_ensembles = pck.load(file)
    plot_all_clustering(feature_ensembles)






