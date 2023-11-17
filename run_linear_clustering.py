from LinearClustering import LinearClustering
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from LocalLinearRegression import LocalLinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pickle as pck
import random
from tqdm import tqdm

class GlobalLinearExplainer():

    def __init__(self, model, x_test, y_pred, features, dataset, sparsity_threshold=0.1, coverage_threshold=0.1, starting_k=5, neighbourhood_threshold=0.05, preload_explainer=True):
        self.model = model
        self.features = features
        self.dataset = dataset
        self.x_test = x_test
        self.y_pred = y_pred
        self.sub_sample = 5000
        self.plotting=True
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.ploting=False

        if preload_explainer:
#            with open(f'saved/{self.dataset}_feature_ensembles_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'rb') as file:
#            with open(f'saved/{self.dataset}_feature_ensembles_full.pck', 'rb') as file:
                self.feature_ensembles = pck.load(file)
        else:
            self.feature_ensembles = None

    def feature_space_clustering(self, feature_xs):
        # Create a kernel density estimation model
        kde = KernelDensity(bandwidth=0.05*max(feature_xs))  # You can adjust the bandwidth
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

    def linear_clustering(self, xdata, ydata, feature, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
    , K=20):
#        print('Performing Local Linear Regression')
        # Perform LocalLinear regression on fetched data
        LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
        w1, w2, w = LLR.calculateLocalModels(self.neighbourhood_threshold)

        D, xDs= LLR.compute_distance_matrix(w, distance_weights=distance_weights)
#        print('Doing K-medoids-clustering')
        # Define number of medoids and perform K medoid clustering.

        LC = LinearClustering(xdata, ydata, D, xDs, feature, K,
                              sparsity_threshold = 0.25,
                              coverage_threshold=0.25*abs(max(xdata)-min(xdata)),
                              gaps_threshold=100,
                              similarity_threshold=0.1)

        clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()

        return clustered_data, medoids, linear_params, clustering_cost

    def multi_layer_clustering(self, search_num, discrete_features):
        x_test = self.x_test
        y_pred = self.y_pred
        features = self.features
        with open(f'saved/{self.dataset}_model.pck', 'rb') as file:
            model = pck.load(file)
        print(f'{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}')

        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), 5000)
        x_test = x_test[random_samples]
        y_pred = y_pred[random_samples]
        self.feature_ensembles = {feature: [] for feature in features}

        discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        # XDs, WDs, neighbourhoodDs
        distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
        for i in tqdm(range(len(features))):
            print(f'{search_num} ^')
#        for i in range(6,8):
#            print(f'--------{features[i]} ({i} out of {len(features)})---------')
            all_feature_clusters = []
            all_feature_linear_params = []
            feature_xs = x_test[:, i]
            if features[i] not in discrete_features:
                cluster_assignments = self.feature_space_clustering(feature_xs)
                K=self.starting_k
            else:
                cluster_assignments = np.zeros(len(feature_xs))
                K=1

            for super_cluster in np.unique(cluster_assignments):
#                print(f'--------{super_cluster} out of {len(np.unique(cluster_assignments))}---------')
                super_cluster_x_indices = np.array(np.argwhere(cluster_assignments == super_cluster)).flatten()
                super_cluster_xs = feature_xs[super_cluster_x_indices]
                super_cluster_y_pred = y_pred[super_cluster_x_indices]

                clustered_data, medoids, linear_params, clustering_cost = self.linear_clustering(super_cluster_xs, super_cluster_y_pred, features[i], distance_weights, K=K)
                [all_feature_clusters.append(cluster) for cluster in clustered_data]
                [all_feature_linear_params.append(linear_param) for linear_param in linear_params]

            if self.plotting==True:
                K=len(all_feature_clusters)
                fig = self.plot_final_clustering(all_feature_clusters, all_feature_linear_params)
                fig.write_html(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{K}.html')
            self.feature_ensembles[features[i]] = [all_feature_clusters, all_feature_linear_params]
#            with open(f'saved/{self.dataset}_feature_ensembles_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'wb') as file:
#                pck.dump(self.feature_ensembles, file)
            if i == len(features)-1:
                with open(f'saved/{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'wb') as file:
                    pck.dump(self.feature_ensembles, file)

        return self.feature_ensembles

    def find_cluster_matches(self, data_instance, cluster_ranges):
        cluster_matches = []
        local_x, local_y_pred = [], []
        for i, x in enumerate(self.x_test):
            num_matches = 0
            for f in range(len(self.features)):
                if cluster_ranges[f][0] <= x[f] <= cluster_ranges[f][1]:
                    num_matches += 1
            cluster_matches.append(num_matches)
            if num_matches >= 20:
                local_x.append(x)
                local_y_pred.append(self.y_pred[i])
        local_x = np.array(local_x)
        local_y_pred = np.array(local_y_pred)

        return cluster_matches, local_x, local_y_pred


    def generate_explanation(self, data_instance, instance_index, instance_prediction, ground_truth):
        instance_cluster_models = []
        exp_prediction = 0
        instance_cluster_ranges = []
        for i, item in enumerate(self.feature_ensembles.items()):
            feature, value = item
            clustered_data, linear_params = value
            cluster_ranges = [[min(cluster[0]), max(cluster[0])] for cluster in clustered_data]
            range_found = False
            for j in range(len(clustered_data)):
                if not range_found:
                    w,b = linear_params[j]
                    if data_instance[i] >= cluster_ranges[j][0] and data_instance[i] <= cluster_ranges[j][1]:
                        instance_cluster_models.append([w,b])
                        instance_cluster_ranges.append(cluster_ranges[j])
                        range_found = True


        cluster_matches, local_x, local_y_pred = self.find_cluster_matches(data_instance, instance_cluster_ranges)
#        if self.plotting==True:
#            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
#            axes.hist(cluster_matches, bins=range(1, len(self.features)))
#            axes.set_xlabel('Number of cluster matches')
#            axes.set_ylabel('Number of data instances')
#            fig.savefig(f'Figures/Clustering/{self.dataset}_{instance_index}_cluster_matches.png')



        instance_explanation = LinearRegression()
        instance_explanation.fit(local_x, local_y_pred)
        exp_local_y_pred = instance_explanation.predict(local_x)
        exp_instance_prediction = instance_explanation.predict([data_instance])[0]
        print(f'Ground Truth: {ground_truth}')
        print(f'Model Prediction: { instance_prediction }')
        print(f'Explanation prediction: { exp_instance_prediction }')

        plotting_data = [data_instance, instance_index, local_x, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models]
        fig = self.plot_data(plotting_data)
        return exp_instance_prediction, plotting_data


    def plot_all_clustering(self, features_to_plot=None):
        if features_to_plot == None:
            features_to_plot = self.features

        if len(features_to_plot) > 4:
            num_cols=4
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/4))

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[1/num_cols for i in range(num_cols)], row_heights =[1/num_rows for row in range(num_rows)], specs = [[{} for c in range(num_cols)] for i in range(num_rows)], subplot_titles=features_to_plot, horizontal_spacing=0.05, vertical_spacing=0.05)


        axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]

        for feature in features_to_plot:
            value = self.feature_ensembles[feature]
            feature_index = self.features.index(feature)
            i = features_to_plot.index(feature)
            clustered_data, linear_params = value
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

            fig.update_xaxes(title='Normalised Feature Value', range=[min(self.x_test[:,feature_index])*1.1, max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title='Predicted RUL',range=[min(self.y_pred)*1.1, max(self.y_pred)*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) == 2:
            height = 500
        elif len(features_to_plot) <= 4:
            height = 350
        else:
            height=350*num_rows
        fig.update_layout(legend=dict(yanchor="top", xanchor="auto", orientation='h'),
                          height=height, xaxis1_range=[-2,2], yaxis_range=[0,280])
        fig.write_html(f'Figures/Clustering/{self.dataset}_clustering_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.html')

        return fig


    def plot_data(self, plotting_data=None, features_to_plot=None):
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
                data_instance, instance_index, local_x, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data

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
#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_cluster_models[i][0]*x+instance_cluster_models[i][1] for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='black'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_explanation.coef_[i]*x+instance_explanation.intercept_ for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='green'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
                fig.layout.annotations[i].update(text=f'{feature} : {round(data_instance[feature_index], 3)}')
            fig.update_xaxes(title='Normalised Feature Value', range=[min(self.x_test[:,feature_index])*1.1, max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title='Predicted RUL', range=[min(self.y_pred)*1.1, max(self.y_pred)*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) == 2:
            height = 500
        elif len(features_to_plot) <= 4:
            height = 350
        else:
            height=350*num_rows
        fig.update_layout(legend=dict(yanchor="top", xanchor="auto", orientation='h', y=-0.25),
                          height=height, xaxis1_range=[-2,2], yaxis_range=[0,280])
#        fig.write_html(f'Figures/{self.dataset}/perturbations_{instance_index}.html', auto_open=False)
        return fig

    def plot_explanation(self, plotting_data=None, features_to_plot=None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Feature Contributions', 'Local Feature Significance'], horizontal_spacing=0.25)
        if features_to_plot != None:
            data_instance, instance_index, local_x, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data
            explained_feature_indices = [f for f in range(len(self.features)) if self.features[f] in features_to_plot]
            explained_features = [self.features[e] for e in explained_feature_indices]
            feature_contributions = instance_explanation.coef_[explained_feature_indices]
            local_feature_significance = [instance_cluster_models[e][0] for e in explained_feature_indices]

            colours = ['green' if x>= 0 else 'red' for x in feature_contributions]
            fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=local_feature_significance, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=2)
            fig.update_layout(height=200+20*len(features_to_plot), title=dict(text=f'Ground Truth {ground_truth} <br> Model Prediction: {round(instance_prediction,2)} <br> Explanation Prediction: {round(exp_instance_prediction, 2)} <br>', y=0.95))
        else:
            fig.update_layout(height=10)

        return fig

