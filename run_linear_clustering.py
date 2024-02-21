from LinearClustering import LinearClustering
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly
import plotly.express as px
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from LocalLinearRegression import LocalLinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pickle as pck
import random
from tqdm import tqdm
import time
import math
import os
from itertools import combinations
from mpl_toolkits import mplot3d

class GlobalLinearExplainer():

    def __init__(self, model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=0.5, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.5, preload_explainer=True):
        self.model = model
        self.features = features
        self.dataset = dataset
        self.x_test = x_test
        self.y_pred = y_pred
        self.plotting=True
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.ploting=False
        self.discrete_features = discrete_features

        if preload_explainer:
            print('Loading', f'saved/feature_ensembles/{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck')
            with open(f'saved/feature_ensembles/{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'rb') as file:
#            with open(f'saved/{self.dataset}_feature_ensembles_full.pck', 'rb') as file:
                self.feature_ensembles = pck.load(file)
        else:
            self.feature_ensembles = None

    def feature_space_clustering(self, feature_xs):
        # Create a kernel density estimation model
        kde = KernelDensity(bandwidth=0.25*max(feature_xs))  # You can adjust the bandwidth
        kde.fit(np.array(feature_xs).reshape(-1, 1))

        # Create a range of data points for evaluation
        x_eval = np.linspace(min(feature_xs), max(feature_xs), 1000)
        log_dens = kde.score_samples(x_eval.reshape(-1, 1))
        dens = np.exp(log_dens)

        # Find local maxima in the density curve
        peaks, _ = find_peaks(dens)
        cluster_centers = x_eval[peaks]

        # Assign data points to clusters based on proximity to cluster centers
        cluster_assignments = []
        for data_point in feature_xs:
            distances = np.abs(cluster_centers - data_point)
            nearest_cluster = np.argmin(distances)
            cluster_assignments.append(nearest_cluster)

        return cluster_assignments

    def plot_final_clustering(self, clustered_data, linear_params):
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

    def plot_clustering_matplotlib(self, clustered_data, linear_params):
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

    def linear_clustering(self, super_cluster_num, xdata, ydata, feature, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
    , K=20):
#        print('Performing Local Linear Regression')
        # Perform LocalLinear regression on fetched data
        LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
        w1, w2, w = LLR.calculateLocalModels(self.neighbourhood_threshold)

        D, xDs= LLR.compute_distance_matrix(w, distance_weights=distance_weights)
#        print('Doing K-medoids-clustering')
        # Define number of medoids and perform K medoid clustering.

        LC = LinearClustering(self.dataset, super_cluster_num, xdata, ydata, D, xDs, feature, K,
                              sparsity_threshold = self.sparsity_threshold,
                              coverage_threshold=self.coverage_threshold,
                              gaps_threshold=0.1,
                              )

        clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()

        return clustered_data, medoids, linear_params, clustering_cost

    def multi_layer_clustering(self, search_num):
        x_test = self.x_test
        y_pred = self.y_pred
        features = self.features
        if self.model == None:
            with open(f'saved/models/{self.dataset}_model.pck', 'rb') as file:
                model = pck.load(file)
        print(f'{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}')

#       Check if ensemble exists already and can edit that.
        if os.path.isfile(f'saved/ensembles/{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck'):
            with open(f'saved/ensembles/{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'rb') as file:
                self.feature_ensembles = pck.load(file)
        else:
            self.feature_ensembles = {feature: [] for feature in features}
#        R = np.random.RandomState(42)
#        random_samples = R.randint(2, len(x_test), 5000)
#        x_test = x_test[random_samples]
#        y_pred = y_pred[random_samples]

        # XDs, WDs, neighbourhoodDs
        distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
        for i in range(len(features)):
#        for i in range(0,1):
            self.feature_ensembles[features[i]] = []
#        for i in range():
            tic = time.time()
            print(f'{search_num}: {i} out of {len(features)}')
#            print(f'--------{features[i]} ({i} out of {len(features)})---------')
            all_feature_clusters = []
            all_feature_linear_params = []
            feature_xs = x_test[:, i]
            if features[i] not in self.discrete_features:
                cluster_assignments = self.feature_space_clustering(feature_xs)
                K=self.starting_k
            else:
                cluster_assignments = np.zeros(len(feature_xs))
                K=1


            for super_cluster in np.unique(cluster_assignments):
                super_clsuter = int(super_cluster)
                print(f'--------{super_cluster} out of {len(np.unique(cluster_assignments))-1}---------')
                super_cluster_x_indices = np.array(np.argwhere(cluster_assignments == super_cluster)).flatten()
                super_cluster_xs = feature_xs[super_cluster_x_indices]
                super_cluster_y_pred = y_pred[super_cluster_x_indices]

                clustered_data, medoids, linear_params, clustering_cost = self.linear_clustering(super_cluster, super_cluster_xs, super_cluster_y_pred, features[i], distance_weights, K=K)
                [all_feature_clusters.append(cluster) for cluster in clustered_data]
                [all_feature_linear_params.append(linear_param) for linear_param in linear_params]

            if self.plotting==True:
                K=len(all_feature_clusters)
                fig = self.plot_final_clustering(all_feature_clusters, all_feature_linear_params)
                fig.write_html(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{K}.html')
            lens = []
            for j in range(len(all_feature_clusters)):
                lens = np.sum([len(cluster[0]) for cluster in all_feature_clusters])

            self.feature_ensembles[features[i]] = [all_feature_clusters, all_feature_linear_params]
#            with open(f'saved/{self.dataset}_feature_ensembles_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'wb') as file:
#                pck.dump(self.feature_ensembles, file)
#            if i == len(features)-1:
            with open(f'saved/feature_ensembles/{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'wb') as file:
                pck.dump(self.feature_ensembles, file)
            toc = time.time()
            print(f'Time taken for one feature: {toc-tic}')
        return self.feature_ensembles

    def find_cluster_matches(self, data_instance, cluster_ranges):
#        print('\n')
#        print('######################')
        cluster_matches = []
        local_x, local_y_pred = [], []
        matched_instances = []
        for i, x in enumerate(self.x_test):
#            print('\n')
            num_matches = 0
            for f in range(len(self.features)):
#                print(f, data_instance[f], x[f])
#                print(cluster_ranges[f])
                if cluster_ranges[f][0] <= x[f] <= cluster_ranges[f][1]:
                    num_matches += 1
            cluster_matches.append(num_matches)
        match_threshold = len(self.features)-1
        while len(matched_instances) < 2:
            matched_instances = np.argwhere(np.array(cluster_matches) >= match_threshold).flatten()
#            matched_instances = self.x_test[matched_instances]
            match_threshold -= 1

        local_x = np.array([self.x_test[i] for i in matched_instances])
        local_y_pred = np.array([self.y_pred[i] for i in matched_instances])

        return cluster_matches, local_x, local_y_pred, matched_instances


    def generate_explanation(self, data_instance, instance_index, instance_prediction=None, ground_truth=None):

        instance_cluster_models = {}
        exp_prediction = 0
        instance_cluster_ranges = {}
        for feature in self.features:
            value = self.feature_ensembles[feature]
            feature_index = self.features.index(feature)
            i = self.features.index(feature)
            clustered_data, linear_params = value
            for cluster, params in zip(clustered_data, linear_params):
                    cluster_min = min(cluster[0])
                    cluster_max = max(cluster[0])
                    if data_instance[feature_index] >= cluster_min and data_instance[feature_index] <= cluster_max:
                        instance_cluster_ranges[i] = [cluster_min, cluster_max]
                        instance_cluster_models[i] = params

#            if not range_found:
        cluster_matches, local_x, local_y_pred, matched_instances = self.find_cluster_matches(data_instance, instance_cluster_ranges)
        print(f'Number of local points: {len(matched_instances)}')
#        if self.plotting==True:
#            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
#            axes.hist(cluster_matches, bins=range(1, len(self.features)))
#            axes.set_xlabel('Number of cluster matches')
#            axes.set_ylabel('Number of data instances')
#            fig.savefig(f'Figures/Clustering/{self.dataset}_{instance_index}_cluster_matches.png')
#            plt.close(fig)
        norm_cluster_matches = [m/max(cluster_matches) for m in cluster_matches]
#        weights = [norm_cluster_matches[m] for m in matched_instances]
#        weights = [np.linalg.norm(x-data_instance) for x in local_x]
        weights = [1 for x in local_x]
        instance_explanation_model = LinearRegression()
        instance_explanation_model.fit(local_x, local_y_pred, sample_weight=weights)
        exp_local_y_pred = instance_explanation_model.predict(local_x)
        exp_instance_prediction = instance_explanation_model.predict([data_instance])[0]
#        print(f'Ground Truth: {ground_truth}')
#        print(f'Model Prediction: { instance_prediction }')
        print(f'LLC Explanation prediction: { exp_instance_prediction }')

        plotting_data = [data_instance, instance_index, local_x, weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models]
#        fig = self.plot_data(plotting_data)
        return exp_instance_prediction, plotting_data, matched_instances

    def evaluate_explanation(self, exp_model, instance, method, instance_index):
        ''' This function measures the robustness of an explanation by adding noise to different features to monitor
        the change in the model prediction. If the variance in the prediction decreases as less important features
        are permuted, then the explanation is robust'''
        feature_contributions = exp_model.coef_
        contribution_magnitudes = [abs(item) for item in feature_contributions]

        original_prediction = self.model.predict([instance])[0]

        ordered_contribution_size_index = np.argsort(contribution_magnitudes)
        deviations = []
        for f in ordered_contribution_size_index:
            feature_noise = np.random.normal(0, 0.001, 1)
            noise = np.zeros(len(instance))
            noise[f] = feature_noise

            noisey_instance = instance + noise
            for f in range(len(noisey_instance)):
                poss_values = np.unique(self.x_test[:,f])
                noisey_instance[f] = poss_values[np.argmin(np.abs(poss_values - noisey_instance[f]))]
            noisy_prediction = self.model.predict([noisey_instance])[0]
            prediction_deviation = abs(original_prediction - noisy_prediction)
            deviations.append(prediction_deviation)

#        deviations = [item/max(deviations) for item in deviations]
#        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
#        axes.plot([i for i in range(len(deviations))], deviations)
##        axes.set_yscale('log')
#        fig.savefig(f'Figures/Robustness{method}_{instance_index}.pdf')

        return deviations

    def new_evaluation(self, instance_index, exp_model, primary_instance, importance_threshold=5):
        important_features = [f for f in self.features if abs(exp_model.coef_[self.features.index(f)]) > importance_threshold]
        print(len(important_features), exp_model.coef_)
        only_important_x = self.x_test[:,[self.features.index(f) for f in important_features]]

        for i in range(len(only_important_x)):
            np.append(only_important_x[i], self.y_pred[i])

        silhouette_scores = []
        max_clusters = 10

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(only_important_x)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(only_important_x, labels))

        optimal_clusters = np.argmax(silhouette_scores) + 2  # Add 2 to account for starting from 2 clusters
        kmeans = KMeans(n_clusters=20)
        kmeans.fit(only_important_x)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        if len(important_features) > 2:

            for pairing in list(combinations(important_features,2)):
                f1 = pairing[0]
                f2 = pairing[1]

                x_f1 = only_important_x[:,important_features.index(f1)]
                x_f2 = only_important_x[:,important_features.index(f2)]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(only_important_x[:,important_features.index(f1)], only_important_x[:,important_features.index(f2)], self.y_pred, c=labels, marker='o', s=5)
                ax.set_xlabel(f1)
                ax.set_ylabel(f2)
                ax.set_zlabel('Prediction')
#                plt.show()

        instance_cluster = labels[instance_index]
#        instance_cluster_x = only_important_x[labels == instance_cluster]
        instance_cluster_x = np.argwhere(labels == instance_cluster).flatten()

        return instance_cluster_x

    def plot_all_clustering(self,instance=None, features_to_plot=None, instances_to_show=None):
        if features_to_plot == None:
            features_to_plot = self.features
        if len(features_to_plot) > 4:
            num_cols=4
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/4))

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[1/num_cols for i in range(num_cols)], row_heights =[1/num_rows for row in range(num_rows)], specs = [[{} for c in range(num_cols)] for i in range(num_rows)], subplot_titles=features_to_plot, horizontal_spacing=0.05, vertical_spacing=0.05)


        axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]

        if instances_to_show != None:
            matched_instance_colours = [px.colors.qualitative.Alphabet[i % 10] for i in range(len(instances_to_show))]
            instances_to_show_matches = []
            for inst in instances_to_show:
                exp_instance_prediction, plotting_data, matched_instances = self.generate_explanation(self.x_test[inst], instance_index=inst)
                instances_to_show_matches.append(matched_instances)

        for feature in features_to_plot:
            value = self.feature_ensembles[feature]
            feature_index = self.features.index(feature)
            i = features_to_plot.index(feature)
            clustered_data, linear_params = value
            showlegend=False
            instanceFound = True
            instance_clusters = []
            all_feature_points = []
            if instance != None:
                fig.add_trace(go.Scatter(x=[self.x_test[instance][feature_index]],y=[self.y_pred[instance]],
                                         mode='markers', marker = dict(size=10, opacity=0.9, color='black'),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])
            for cluster, params in zip(clustered_data, linear_params):
                colour = np.random.rand(3)
                colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
                fig.add_trace(go.Scatter(x=cluster[0],y=cluster[1],
                                         mode='markers', marker = dict(size=3, opacity=0.6, color=colour),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])

                fig.add_trace(go.Scatter(x=cluster[0],y=[params[0]*x+params[1] for x in cluster[0]],
                                         mode='lines', marker = dict(size=3, opacity=0.9, color=colour),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])

#                instances_to_show = None
                if instances_to_show != None:

                    for num, inst in enumerate(instances_to_show):
#                            colour = np.random.rand(3)
                            matched_instances = instances_to_show_matches[num]
                            if inst == instances_to_show[0]:
                                showlegend=True
                            else:
                                showlegend=False
                            fig.add_trace(go.Scatter(x=[self.x_test[inst,feature_index]],y=[self.y_pred[inst]],
                                         mode='markers', marker = dict(size=20, opacity=0.9, color=[matched_instance_colours[num]]),
                                         showlegend=False, name='Other Instances', legendgroup='Other Instances'),
                                          row=axes[i][0], col=axes[i][1])

                            fig.add_trace(go.Scatter(x=[self.x_test[x,feature_index] for x in matched_instances],y=[self.y_pred[x] for x in matched_instances],
                                         mode='markers', marker = dict(size=10, opacity=0.9, color=[matched_instance_colours[num] for x in matched_instances]),
                                         showlegend=False, name='Other Instances', legendgroup='_nolegend_'),
                                          row=axes[i][0], col=axes[i][1])



            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title='Predicted RUL',range=[min(self.y_pred)*1.1, max(self.y_pred)*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 500
        else:
            height=350*num_rows
        fig.update_layout(
                          height=height)
        fig.write_html(f'Figures/{self.dataset}/Clustering/{self.dataset}_clustering_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.html')
        self.plot_all_clustering

        return fig


    def plot_data(self, plotting_data=None, features_to_plot=None, instances_to_show=None):
        if features_to_plot == None:
            features_to_plot = self.features

        if len(features_to_plot) > 4:
            num_cols=4
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/4))
        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[1/num_cols for i in range(num_cols)], row_heights =[1/num_rows for row in range(num_rows)], specs = [[{} for c in range(num_cols)] for i in range(num_rows)], subplot_titles=features_to_plot, horizontal_spacing=0.1, vertical_spacing=0.05)

        axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]


        for feature in features_to_plot:
            value = self.feature_ensembles[feature]
            i = features_to_plot.index(feature)
            feature_index = self.features.index(feature)
            if i == 0:
                showlegend=True
            else:
                showlegend=False
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))

            if plotting_data == None:
                fig.add_trace(go.Scatter(x=self.x_test[:,feature_index],y=self.y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='black'),
                                         showlegend=showlegend, name='Test Data', legendgroup='Test Data'),
                              row=axes[i][0], col=axes[i][1])
            else:
                data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data

                fig.add_trace(go.Scatter(x=self.x_test[:,feature_index],y=self.y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='lightgrey'),
                                         showlegend=showlegend, name='Test Data', legendgroup='Test Data'),
                              row=axes[i][0], col=axes[i][1])

                fig.add_trace(go.Scatter(x=[data_instance[feature_index]],y=[instance_prediction],
                                         mode='markers', marker = dict(size=30, opacity=0.9, color='black'),
                                         showlegend=showlegend, name='Instance Model Prediction', legendgroup='Instance Model Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=[data_instance[feature_index]],y=[exp_instance_prediction],
                                         mode='markers', marker = dict(size=30, opacity=0.9, color='orange'),
                                         showlegend=showlegend, name='Instance Explanation Prediction', legendgroup='Instance Explanation Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=local_y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='red'),
                                         showlegend=showlegend, name='Local Points Model Prediction', legendgroup='Local Points Model Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=exp_local_y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='blue'),
                                         showlegend=showlegend, name='Local Points Explanation Prediction', legendgroup='Local Points Explanation Prediction'),
                              row=axes[i][0], col=axes[i][1])
                if instances_to_show != None or []:
                    for inst in instances_to_show:
                        if inst == instances_to_show[0]:
                            showlegend=True
                        else:
                            showlegend=False
                        fig.add_trace(go.Scatter(x=[self.x_test[inst,feature_index]],y=[self.y_pred[inst]],
                                     mode='markers', marker = dict(size=20, opacity=0.9, color='green'),
                                     showlegend=showlegend, name='Other Instances', legendgroup='Other Instances'),
                                      row=axes[i][0], col=axes[i][1])


#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_cluster_models[i][0]*x+instance_cluster_models[i][1] for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='black'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_explanation.coef_[i]*x+instance_explanation.intercept_ for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='green'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
                fig.layout.annotations[i].update(text=f'{feature} : {round(data_instance[feature_index], 3)}')
            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title='Predicted RUL', range=[min(self.y_pred)*1.1, max(self.y_pred)*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 600
        else:
            height=350*num_rows
        fig.update_layout(legend=dict(yanchor="top", xanchor="auto", orientation='h', y=-0.25),
                          height=height)
#        fig.write_html(f'Figures/{self.dataset}/perturbations_{instance_index}.html', auto_open=False)
        return fig

    def plot_explanation(self, plotting_data=None, features_to_plot=None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Feature Contributions', 'Local Feature Significance'], horizontal_spacing=0.25)
        if features_to_plot != None:
            data_instance, instance_index, local_x, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = plotting_data
            explained_feature_indices = [f for f in range(len(self.features)) if self.features[f] in features_to_plot]
            explained_features = [self.features[e] for e in explained_feature_indices]
            feature_contributions = instance_explanation_model.coef_[explained_feature_indices]
            local_feature_significance = [instance_cluster_models[e][0] for e in explained_feature_indices]

            colours = ['green' if x>= 0 else 'red' for x in feature_contributions]
            fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=local_feature_significance, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=2)
            fig.update_layout(height=200+20*len(features_to_plot), title=dict(text=f'Ground Truth {ground_truth} <br> Model Prediction: {round(instance_prediction,2)} <br> Explanation Prediction: {round(exp_instance_prediction, 2)} <br>', y=0.95))
        else:
            fig.update_layout(height=10)

        return fig

    def interactive_exp_plot(self, instance, instance_index,  instance_model_prediction, y_test_pred, instance_exp_prediction, exp, local_points, local_weights, model_local_points_prediction, exp_local_points_prediction, targetFeature, neighbours=None):


        explanation_error = mean_squared_error(model_local_points_prediction, exp_local_points_prediction)

        num_rows=int(np.ceil(len(self.features)/4))+1
        num_cols=4

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[0.33]+[0.16]*(num_rows-1),
                            specs = [
                                [{'colspan':2}, None, {'colspan':2}, None],
                                ]+[[{}, {}, {}, {}] for i in range(num_rows-1)], subplot_titles=['Explanation Prediction Convergence', 'Feature Significance']+self.features,
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        colours = ['green' if x>= 0 else 'red' for x in exp]

        # Plot convergence of error as features are added
#        exp_convergence = []
#        for included_features in range(len(self.features)):
#            intercept = exp.intercept
#            for i in range(included_features):
#                intercept+= feature_contributions[i]*instance_x[i]
#
#            exp_convergence.append(intercept)
#
#        fig.add_trace(go.Scatter(x=[num for num in range(len(exp_convergence))], y=exp_convergence, mode='lines', marker = dict(color='orange', size=3), showlegend=False), row=1, col=1)


        # Plot explanation bar chart
        fig.add_trace(go.Bar(x=exp, y=self.features, marker_color=colours, orientation='h', showlegend=False), row=1, col=3)

        axes = [[row, col] for row in range(2,num_rows+1) for col in range(1,num_cols+1)]

#        for n in range(len(neighbours)):
#            neighbours[n] = [neighbours[n][i] for i in explained_feature_indices]
#
        for i in range(len(self.features)):
#        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='orange', size=3)), row=ax[0], col=ax[1])
            if i==0:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(x=self.x_test[:,i],y=y_test_pred,
                                     mode='markers', marker = dict(color='lightgrey', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Test data'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=local_points[:,i],y=model_local_points_prediction,
                                     mode='markers', marker = dict(color=local_weights, colorscale='Oranges', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Model (f) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=local_points[:,i],y=exp_local_points_prediction,
                                     mode='markers', marker = dict(color=local_weights, colorscale='Greens', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance[i]],y=[instance_model_prediction],
                                     mode='markers', marker = dict(color='red', size=20),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance[i]],y=[instance_exp_prediction],
                                     mode='markers', marker = dict(color='blue', size=10),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])


        fig.update_layout(title=dict(text = f' Explanation for instance {instance} <br> Explanation Error = {explanation_error:.2f} <br> Model Instance Prediction {instance_model_prediction} <br> Explanation Instance Prediction {instance_exp_prediction}', y=0.99, x=0),
                          font=dict(size=14),
                          legend=dict(yanchor="top", y=1.1, xanchor="right"),
                          height=300*num_rows, )
#        if self.automated_locality == True and self.newMethod == True:
#            suffix = 'LLC'
#        elif self.automated_locality == False and self.newMethod == True:
#            suffix = '_CHILLI'
#        elif self.automated_locality == False and self.newMethod == False:
#            suffix = '_LIME'
#        else:
#            suffix = ''
        suffix = 'LLC'
        fig.write_html(f'Figures/{self.dataset}/Explanations/instance_{instance_index}_{suffix}_explanation.html', auto_open=False)
