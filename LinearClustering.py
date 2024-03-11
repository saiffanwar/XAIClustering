from os import wait
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import compress
from pandas.core.base import NoNewAttributesMixin
from sklearn.metrics import mean_squared_error
import statistics as stat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from cyclicRegression import CyclicRegression
from similaritychecker import calculate_line_similarity
from copy import deepcopy
import os
import glob
import time
import warnings
from pprint import pprint
import plotly.graph_objects as go
import plotly


warnings.filterwarnings("ignore")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})





class LinearClustering():

    ''' This class clusters a dataset into regions of linearity. It is provided a distance matrix computed in the LocalLinearRegression class.
    This distance matrix is based on the local linear models of each point as well as the raw distance values. '''

    def __init__(self, dataset, super_cluster, x, y, D, xDs, feature, K,
                sparsity_threshold,
                coverage_threshold,
                gaps_threshold,
                feature_type='Linear'
                ):

        self.x = x
        self.y = y
        self.D = D
        self.xDs = xDs
        self.feature_name = feature
        self.clustering_cost = None
        self.clustered_data = None
        self.medoids = None
        self.K = K
        self.init_K = K
        self.sparsity_threshold = sparsity_threshold
#        self.similarity_threshold = similarity_threshold
        self.coverage_threshold = coverage_threshold
#        self.gaps_threshold = gaps_threshold
        self.gaps_threshold=None
        self.dataset = dataset
        self.super_cluster = super_cluster
        self.feature_type = feature_type
        self.plotting=True

        # Clear folder for figures to show evolution of clustering.

        if os.path.isdir(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}'):
            pass
        else:
            os.makedirs(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}')

        files = glob.glob(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/*')
        for f in files:
            os.remove(f)

    def LR(self, x, y):
        if self.feature_type == 'Linear':
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_
        elif self.feature_type == 'Cyclic':
            CR = CyclicRegression(boundary=max(self.x))
            m, c = CR.cyclicRegression(x, y)
            return m, c

    def find_gaps(self, x=None):
        '''Find gaps in the data and do not cluster across these gaps'''
        if self.gaps_threshold == None:
            return []
        x = np.sort(x)
        if len(np.unique(x)) > 1:
            all_x_differences = [x[i+1]-x[i] for i in range(len(x)-1)]
            median_x_difference = stat.mean(all_x_differences)
            gap_indices = [i for i in range(len(all_x_differences)) if all_x_differences[i] > self.gaps_threshold*median_x_difference]
            gap_ranges = [[x[i], x[i+1]] for i in gap_indices]
        else:
            gap_ranges = []
        return gap_ranges

    def pick_medoids(self, K):
            sorted_x = np.sort(self.x)
            medoids = np.linspace(0, len(self.x)-1, K)
            medoids = [list(self.x).index(sorted_x[int(med)]) for med in medoids]
            return medoids

    def find_medoids(self, clustered_data):
        medoids = []
        for cluster in clustered_data:
            if cluster == []:
                medoids.append(None)
            else:
                distMatrix = np.zeros((len(cluster), len(cluster)))
                for i in range(len(cluster)):
                    for j in range(len(cluster)):
                        distMatrix[i,j] = self.D[cluster[i], cluster[j]]
                medoid = np.argmin(np.sum(distMatrix, axis=1))
                medoids.append(cluster[medoid])
        return medoids

    def adapted_clustering(self,init_clusters=True, clustered_data=None, medoids=None, clustering_cost=None, linear_params=None):

        '''
        This is the main clustering algorithm. It will start with some arbitrary number of K to generate clusters.
        It takes a K-medoids based approach to generate clusters using the provided distance matrix. It then checks if there
        are any clusters contained within other clusters. If so, it merges the child cluster into the parent cluster.
        Then it checks if there are any neighbouring clusters which have a similar linearity and merges those that do.
        Then based on the new number of clusters after merging, it will recluster the data to find the optimal clustering.
        This is done recursively until there are no new merges therefore an optimal value of K is found.
        '''

        K = self.K
        num_iterations = 1000
        iter = 0
        self.colours = [np.random.rand(1,3) for i in range(K)]

        new_clustering_cost = None
#        init_clusters = True
        recluster = True
#        self.gaps = self.find_gaps(self.x)
        print(f'Clustering {self.feature_name} with K={K} clusters.')
        if not self.plotting:
            fig=None
        while iter < num_iterations and recluster == True:

            if init_clusters == True:

                if self.medoids == None:
                    medoids = self.pick_medoids(K)
                else:
                    medoids = deepcopy(self.medoids)

                # Reset clustered data for new clustsering.
                clustered_data = [[med] for med in medoids]
                minimum_distances = []
                for index,i in enumerate(self.D):
                    if index not in medoids:
                        minimum_distances.append(min([i[j] for j in medoids]))
                        closest = np.argmin([i[j] for j in medoids])
                        clustered_data[closest].append(index)
                _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
                init_clusters = False

            else:
                # Recluster the data to find the optimal clustering.
                clustering_cost, linear_params, clustered_data, medoids, recluster = self.recluster(K, clustered_data, medoids, clustering_cost, linear_params)

            if len(clustered_data) == 1:
                if self.plotting:
                    fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
                    fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/final_{K}.pdf', bbox_inches='tight')
                clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
                return clustered_datapoints, medoids, linear_params, clustering_cost, fig
            iter+=1


        if self.plotting:
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/before_verifying_{K}.pdf', bbox_inches='tight')

        changes = [True]
        while any(changes):
            pre_verification_clustering = clustered_data
            clustered_data, medoids, linear_params, clustering_cost, changes = self.verify_clustering(clustered_data)

        self.clustered_data = deepcopy(clustered_data)
        self.medoids = deepcopy(medoids)
        self.linear_params = deepcopy(linear_params)
        self.clustering_cost = deepcopy(clustering_cost)

        if self.plotting:
            fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/after_verifying_{K}_merged_to_{len(self.clustered_data)}.pdf', bbox_inches='tight')

        # If the number of clusters has changed, pick new medoids from the same clusters and optimise further.
        if len(self.clustered_data) < K:
            K = len(self.clustered_data)
            self.K = K
            return self.adapted_clustering(False, self.clustered_data, self.medoids,self.clustering_cost, self.linear_params)
        else:
            if self.plotting:
                fig = self.plot_final_clustering(self.clustered_data, self.linear_params)
                fig.write_html(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/final_{K}.html')
                fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
                fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/final_{K}.pdf', bbox_inches='tight')
            clustered_datapoints = self.cluster_indices_to_datapoints(self.clustered_data)
            return clustered_datapoints, self.medoids, self.linear_params, self.clustering_cost, fig

        #  Check if there is any clusters contained within other clusters. If so, merge the child cluster into the parent cluster and recalculate the           new cluster model parameters.


    def verify_clustering(self, clustered_data):

        changes = []
        clustered_data = self.order_clusters(clustered_data)
        if len(clustered_data) != 1:
            pre_clustered_data = deepcopy(clustered_data)
            clustered_data = self.check_cluster_gaps(clustered_data)
            # Check if satisfying this constraint has changed the clustering.
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/after_separating_gaps_{time.time()}.pdf', bbox_inches='tight')


        if len(clustered_data) != 1:
            # Check if neighbouring clusters overlap.
            pre_clustered_data = deepcopy(clustered_data)
            clustered_data = self.check_cluster_overlap(clustered_data, linear_params, len(clustered_data))
            # Check if satisfying this constraint has changed the clustering.
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting == True:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/after_checking_overlaps_{time.time()}.pdf', bbox_inches='tight')

#            print('Post overlap Check K = ', len(clustered_data))

        if len(clustered_data) != 1:
        # Check size of clusters. If 10x smaller than neighbouring clusters, merge them.
            pre_clustered_data = deepcopy(clustered_data)
            clustered_data = self.check_cluster_sparsity(clustered_data, linear_params)
            # Check if satisfying this constraint has changed the clustering.
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting == True:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/after_checking_sparsity_{time.time()}.pdf', bbox_inches='tight')


        if len(clustered_data) != 1 and False:
        # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
            pre_clustered_data = deepcopy(clustered_data)
            clustered_data = self.check_cluster_similarity(clustered_data, linear_params)
            # Check if satisfying this constraint h{self.dataset}/as changed the clustering.
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting:
            fig = self.plotMedoids(clustered_data, None, linear_params, None)
            fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/after_similarity_merge_{time.time()}.pdf', bbox_inches='tight')

#        print('Post similarity Check K = ', len(clustered_data))

        clustering_cost = self.calculate_clustering_cost(clustered_data)

        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        medoids = self.find_medoids(clustered_data)

        return clustered_data, medoids, linear_params, clustering_cost, changes

    def gen_clustering(self, data, medoids):
        new_clustered_data = [[new_med] for new_med in medoids]
        for index,i in enumerate(self.D):
            closest = np.argmin([i[j] for j in medoids])
            new_clustered_data[closest].append(index)
        return new_clustered_data


    def recluster(self, K, clustered_data, medoids, clustering_cost, linear_params):

        '''
        This is a utility function used in the adapted clustering algorithm. It will randomly select a cluster
        and then randomly select a new medoid for the cluster and then assigning all points the appropriate cluster
        based on the new medoid. Linear Regression models are fit to all of the new clusters. A cost is calculated
        based on the combined error of all the new LR models. If this cost is lower than the previous cost, the new
        clustering is favoured. This function is called repeatedly for a specified number of iterations.
        '''

        random_cluster = random.choice(np.arange(0,K,1))
        while len(clustered_data[random_cluster]) == 1:
            random_cluster = random.choice(np.arange(0,K,1))
        if len(clustered_data[random_cluster]) != 1:
            prev_medoid = medoids[random_cluster]
            clustered_data[random_cluster].remove(prev_medoid)

            new_medoid = random.choice(clustered_data[random_cluster])
            new_medoids = deepcopy(medoids)
            new_medoids[random_cluster] = new_medoid
            clustered_data[random_cluster].append(prev_medoid)



        # Reset clustered data for new clustsering.
        new_clustered_data = [[new_med] for new_med in new_medoids]
        for index,i in enumerate(self.D):
            if index not in new_medoids:
                closest = np.argmin([i[j] for j in new_medoids])
                new_clustered_data[closest].append(index)

        _,new_linear_params, new_clustering_cost = self.calc_cluster_models(self.x, self.y, new_clustered_data)

        recluster = True
        if new_clustering_cost < clustering_cost:
#            print(f'New clustering cost: {new_clustering_cost} < {clustering_cost} = Old clustering cost')
            clustering_cost = new_clustering_cost
            linear_params = new_linear_params
            clustered_data = new_clustered_data
            medoids = new_medoids
        elif new_clustering_cost == clustering_cost:
            recluster = False
#        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
#        fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.pdf')
        return clustering_cost, linear_params, clustered_data, medoids, recluster

    '''--------------------------GAPS---------------------------'''

    def check_cluster_gaps(self, clustered_data):
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        new_clustering = []
        for cluster_num, cluster in enumerate(clustered_datapoints):
            cluster_xs = cluster[0]
            if len(cluster_xs) > 1:
                cluster_gaps = self.find_gaps(cluster_xs)
                if len(cluster_gaps) > 0:
                    separated_clusters = [[] for i in range(len(cluster_gaps)+1)]
                    remaining_xs = [i for i in range(len(cluster_xs))]
                    for i, gap in enumerate(cluster_gaps):
                        separated_clusters[i] = [x for x in remaining_xs if cluster_xs[x] <= gap[0]]
                        for x in separated_clusters[i]:
                            remaining_xs.remove(x)
                        separated_clusters[i+1] = [x for x in remaining_xs if x not in separated_clusters[i]]

                    for separated_cluster in separated_clusters:
                        new_clustering.append([clustered_data[cluster_num][x] for x in separated_cluster])
                else:
                    new_clustering.append(clustered_data[cluster_num])
            else:
                new_clustering.append(clustered_data[cluster_num])

        new_clustering = [i for i in new_clustering if i != []]
        return new_clustering


    '''--------------------------SIMILARITY---------------------------'''

    def check_cluster_similarity(self, clustered_data, linear_params):

        '''
        Utility function which checks the similarity between linear models of neighbouring clusters.
        If the gradient of neighbouring models is similar (based on some defined threshold), the clusters
        are merged.
        '''

#        new_clustering = deepcopy(clustered_data)
#        removed = []
        new_clustering = [clustered_data[0]]
        new_linear_params = [linear_params[0]]
        for i in range(len(clustered_data)-1):
#            print(len(new_clustering[i]), len(new_clustering[i+1]))
#            if i not in removed and (i+1) not in removed:
#                cluster_datapoints = self.cluster_indices_to_datapoints(new_clustering)
            current_cluster_datapoints = self.cluster_indices_to_datapoints(new_clustering[i])
            next_cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data[i+1])
            current_cluster_params = new_linear_params[i]
            next_cluster_params = linear_params[i+1]

            similarity, sim_ax = calculate_line_similarity(self.x, self.y,i, i+1, current_cluster_datapoints[0], next_cluster_datapoints[0], current_cluster_params, next_cluster_params, plotting=False, feature_name=self.feature_name)

#            if similarity < 20 or abs(current_cluster_params[0]-next_cluster_params[0]) < 0.2*(current_cluster_params[0]):
            if similarity < self.similarity_threshold:

                fig, axes = plt.subplots(1,3, figsize=(18*0.39,4*0.39))

                axes[0].scatter(self.x[new_clustering[i]], self.y[new_clustering[i]], s=1, color='red', label=f'Cluster {i}')
                m, c = self.LR(self.x[new_clustering[i]], self.y[new_clustering[i]])
                axes[0].plot(self.x[new_clustering[i]], m*self.x[new_clustering[i]]+c, color='red', linewidth=0.5)
                axes[0].scatter(self.x[clustered_data[i+1]], self.y[clustered_data[i+1]], s=1, color='blue', label=f'Cluster {i+1}')
                m, c = self.LR(self.x[clustered_data[i+1]], self.y[clustered_data[i+1]])
                axes[0].plot(self.x[clustered_data[i+1]], m*self.x[clustered_data[i+1]]+c, color='blue', linewidth=0.5)
                temp = new_clustering[i] + clustered_data[i+1]
                if len(self.find_gaps(self.x[temp])) == 0:
                    new_clustering.append(new_clustering[i] + clustered_data[i+1])
                    new_clustering[i] = []
                else:
                    new_clustering.append(clustered_data[i+1])

                # Recalculate the parameters for the single cluster. Need to find the new datapoints first.
                xs, ys = self.cluster_indices_to_datapoints(new_clustering[i+1])
                new_linear_params.append(self.LR(xs, ys))

                axes[2].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], s=1, color='blue', label=f'Cluster {i}')
                m, c = self.LR(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]])
                axes[2].plot(self.x[new_clustering[i+1]], m*self.x[new_clustering[i+1]]+c, color='blue', linewidth=0.5)

                for ax in [0,1,2]:
#                    axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
                    axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
                    axes[ax].set_xlabel(r'$x$', fontsize=11)
                    axes[ax].set_xticklabels(axes[0].get_xticklabels(), fontsize=11)
                axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
                for ax in [1,2]:
                    axes[ax].set_ylabel('')
                    axes[ax].set_yticklabels('')

                [x1, y1], [x2, y2], [line2_interp_x1, line1_interp_x2] = sim_ax
                axes[1].set_title(f'Similarity: {similarity:.2f}')
                axes[1].plot(x1, y1, color='red', alpha=0.5)
                axes[1].plot(x2, y2, color='blue', alpha=0.5)
                axes[1].plot(x1, line2_interp_x1, color='blue', alpha=0.5)
                axes[1].plot(x2, line1_interp_x2, color='red', alpha=0.5)

                fig.savefig(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}/{self.super_cluster}/similarity_{i}_{i+1}_{time.time()}.pdf', bbox_inches='tight')
            else:
                new_clustering.append(clustered_data[i+1])
                new_linear_params.append(next_cluster_params)
        clustered_data = [cluster for cluster in new_clustering if cluster != []]

        return clustered_data


    def order_clusters(self, clustered_data):

        '''
        Utility function which orders the clusters based on the minimum x value of each cluster.
        '''
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)

        minimums = [min(cluster_datapoints[i][0]) for i in range(len(cluster_datapoints))]
        index_sorted = sorted(range(len(minimums)), key=lambda k: minimums[k])
        ordered_clusters = [clustered_data[i] for i in index_sorted]

        return ordered_clusters


    '''--------------------------OVERLAP---------------------------'''

    def check_cluster_overlap(self, clustered_data, linear_params, K):
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        cluster_ranges = [[min(cluster_datapoints[i][0]), max(cluster_datapoints[i][0])] for i in range(len(cluster_datapoints))]
        new_clustering = deepcopy(clustered_data)
        for cluster_num, cluster in enumerate(clustered_data):
            if cluster_num < len(clustered_data)-1:
                range1 = cluster_ranges[cluster_num]
                range2 = cluster_ranges[cluster_num+1]
                # If the beginning of the following cluster falls within this cluster, separate them through the midpoint of the intersection.
                method = None
                if (range1[0] <= range2[0] <= range1[1]) and (range1[0] <= range2[1] <= range1[1]):
                    method = 'contain'
                elif (range1[0] <= range2[0] <= range1[1]) or (range1[0] <= range2[1] <= range1[1]):
                    method = 'overlap'

                if method == 'overlap' or method == 'contain':
                    if method == 'overlap':
                        midpoint = range2[0] + (range1[1] - range2[0])/2
                    elif method == 'contain':
                        midpoint = (range1[0] - range1[1])/2

#
                    both_cluster_indices = cluster + clustered_data[cluster_num+1]

                    new_cluster1, new_cluster2 = [], []
#
                    new1xs, new2xs = [], []
                    for indexed_point in both_cluster_indices:
                        x = self.x[indexed_point]
                        y = self.y[indexed_point]
                        if x <= midpoint:
                            new_cluster1.append(indexed_point)
                            new1xs.append(x)
                        else:
                            new_cluster2.append(indexed_point)
                            new2xs.append(x)
                    clustered_data[cluster_num] = new_cluster1
                    clustered_data[cluster_num+1] = new_cluster2
#
#                    fig, axes = plt.subplots(1,2, figsize=(20*0.39,4*0.39))
#                    axes[0].scatter(self.x[cluster], self.y[cluster], color='blue', s=1)
#                    m, c = LR(self.x[cluster], self.y[cluster])
#                    axes[0].plot(self.x[cluster], m*self.x[cluster]+c, color='blue', linewidth=0.5)
#                    axes[0].scatter(self.x[clustered_data[cluster_num+1]], self.y[clustered_data[cluster_num+1]], color='red', s=1)
#                    m, c = LR(self.x[clustered_data[cluster_num+1]], self.y[clustered_data[cluster_num+1]])
#                    axes[0].plot(self.x[clustered_data[cluster_num+1]], m*self.x[clustered_data[cluster_num+1]]+c, color='red', linewidth=0.5)
#                    axes[0].plot([midpoint, midpoint], [0, 1], color='black', linestyle='--')
#                    axes[1].scatter(new1xs, [self.y[i] for i in new_cluster1], color='blue', s=1)
#                    if new_cluster1 != []:
#                        m, c = LR(new1xs, [self.y[i] for i in new_cluster1])
#                        axes[1].plot(new1xs, m*new1xs+c, color='blue', linewidth=0.5)
#                    axes[1].scatter(new2xs, [self.y[i] for i in new_cluster2], color='red', s=1)
#                    if new_cluster2 != []:
#                        m, c = LR(new2xs, [self.y[i] for i in new_cluster2])
#                        axes[1].plot(new2xs, m*new2xs+c, color='red', linewidth=0.5)
#                    axes[1].plot([midpoint, midpoint], [0, 1], color='black', linestyle='--')
#                    for ax in [0,1]:
#                        axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
#                        axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
#                        axes[ax].set_xlabel(r'$x$', fontsize=11)
#                        axes[ax].set_xticklabels(axes[ax].get_xticklabels(), fontsize=11)
#                    axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
#                    axes[1].set_ylabel('')
#                    axes[1].set_yticklabels('')
#                    fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/overlap_{cluster_num}_{cluster_num+1}_{K}.pdf', bbox_inches='tight')
#                    fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
#                    fig.savefig(f'Figures/Clustering/OptimisedClusters/overlap_{cluster_num}_{cluster_num+1}_{K}.pdf')

        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        return clustered_data

    '''--------------------------SPARSITY & COVERAGE---------------------------'''


    def check_cluster_sparsity(self, clustered_data, linear_params):
        data_x_range = abs(max(self.x)-min(self.x))
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        largest_cluster_size = max([len(cluster) for cluster in clustered_data])

        new_clustering = deepcopy(clustered_data)
        for i in range(len(clustered_data)):

            sparsity = len(new_clustering[i])/largest_cluster_size

            if i == len(clustered_data)-1:
                coverage = abs(max(clustered_datapoints[i][0]) - max(clustered_datapoints[i-1][0]))
            else:
                coverage = abs(min(clustered_datapoints[i+1][0]) - min(clustered_datapoints[i][0]))
            coverage = coverage/data_x_range

            if sparsity < self.sparsity_threshold:
                suffix = 'sparsity'
            elif coverage < self.coverage_threshold:
                suffix = 'coverage'
#            if (sparsity < self.sparsity_threshold) or (coverage < self.coverage_threshold):
#                print(self.sparsity_threshold, self.coverage_threshold)
#                print(suffix, sparsity, coverage)
#
#                print([len(cluster) for cluster in new_clustering])
#                fig, axes = plt.subplots(1,3, figsize=(20*0.39,4*0.39))
#                axes[0].scatter(self.x[new_clustering[i]], self.y[new_clustering[i]], color='blue', s=1)
                # If first or last cluster
                if i == 0 or new_clustering[i-1] == []:
                    # if last cluster:
                    if i ==len(new_clustering)-1:
                        # make j the cluster second to last
                        j = i-1
                        # If the second to last is empty then pick the one before
                        while new_clustering[j] == []:
                            j -= 1
                        # If all until j=-1 are empty (newclustering[-1] which is the last element) then skip
                        if j != -1:
                            temp_clustering = new_clustering[j] + new_clustering[i]
                            if len(self.find_gaps(self.x[temp_clustering])) == 0:
                                new_clustering[j] = new_clustering[j] + new_clustering[i]
                                new_clustering[i] = []
#                        axes[0].scatter(self.x[new_clustering[j]], self.y[new_clustering[j]], color='red', s=1)
                    else:
#                        axes[0].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='red', s=1)
                        temp_clustering = new_clustering[i+1] + new_clustering[i]
                        if len(self.find_gaps(self.x[temp_clustering])) == 0:
                            new_clustering[i+1] = new_clustering[i+1] + new_clustering[i]
                            new_clustering[i] = []
                elif i == len(new_clustering)-1 or new_clustering[i+1] == []:

#                    axes[0].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='green', s=1)
#                    axes[1].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='green', s=1)

                    temp_clustering = new_clustering[i-1] + new_clustering[i]
                    if len(self.find_gaps(self.x[temp_clustering])) == 0:
                        new_clustering[i-1] = new_clustering[i-1] + new_clustering[i]
                        new_clustering[i] = []
                else:
                    tempNewClustering1 = deepcopy(new_clustering)

#                    axes[0].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='red', s=1)
#                    axes[2].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='red', s=1)

                    tempNewClustering1[i-1] = new_clustering[i-1] + new_clustering[i]
                    clustering_1_gaps = self.find_gaps(self.x[tempNewClustering1[i-1]])
                    tempNewClustering1[i] = []
                    cost1 = self.calculate_clustering_cost(tempNewClustering1)

#                    axes[1].scatter(self.x[tempNewClustering1[i-1]], self.y[tempNewClustering1[i-1]], color='red', s=1)
#                    axes[1].set_title(f'Clustering Cost = {cost1:.4f}')

                    tempNewClustering2 = deepcopy(new_clustering)

#                    axes[0].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='green', s=1)
#                    axes[1].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='green', s=1)

                    tempNewClustering2[i+1] = new_clustering[i+1] + new_clustering[i]
                    clustering_2_gaps = self.find_gaps(self.x[tempNewClustering2[i+1]])
                    tempNewClustering2[i] = []
                    cost2 = self.calculate_clustering_cost(tempNewClustering2)

#                    axes[2].scatter(self.x[tempNewClustering2[i+1]], self.y[tempNewClustering2[i+1]], color='green', s=1)
#                    axes[2].set_title(f'Clustering Cost = {cost2:.4f}')

                    if cost1 <= cost2  == 0 and len(clustering_1_gaps) == 0:
                        new_clustering = deepcopy(tempNewClustering1)
                    elif len(clustering_2_gaps) == 0:
#                    else:
                        new_clustering = deepcopy(tempNewClustering2)


#                    for ax in [0,1,2]:
#                        axes[ax].set_xlim(1.05*(min(self.x[new_clustering[i-1]])), 1.05*(max(self.x[new_clustering[i+1]])))
#
#                for ax in [0,1,2]:
##                    axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
#                    axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
#                    axes[ax].set_xlabel(r'$x$', fontsize=11)
#                    axes[ax].set_xticklabels(axes[ax].get_xticklabels(), fontsize=11)
#                axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
#                for ax in [1,2]:
#                    axes[ax].set_ylabel('')
#                    axes[ax].set_yticklabels('')
#
#                fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/{suffix}_{i}_{time.time()}.pdf', bbox_inches='tight')
#        print([len(cluster) for cluster in new_clustering])
        clustered_data = [cluster for cluster in new_clustering if cluster != []]

        return clustered_data

    def cluster_indices_to_datapoints(self, clustered_indices):

        '''
        Utility function which converts the clusters containing indices of the x values to the raw x datapoint values.
        '''
        # If it just a single cluster whose datapoints are being fetched, then dont create nested lists.
        if isinstance(clustered_indices[0], (int, np.integer)):
            xs, ys = [], []
            for i in clustered_indices:
                xs.append(self.x[i])
                ys.append(self.y[i])
            return xs, ys
        # If it is a list of clusters, then create nested lists.
        else:
            datapoints = [[[],[]] for i in range(len(clustered_indices))]
            for cluster in range(len(clustered_indices)):
                for i, point_index in enumerate(clustered_indices[cluster]):
#                        print(cluster, point_index)
                    datapoints[cluster][0].append(self.x[point_index])
                    datapoints[cluster][1].append(self.y[point_index])
            return datapoints

    def calculate_clustering_cost(self, clustered_data):

        '''
        Utility function which calculates the cost of a clustering based on the combined error of the linear regression models
        for each cluster. The cost is the sum of the absolute errors of each cluster.
        '''
        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
        total_error = 0
        ys = []
        preds = []
        for cluster_num, cluster in enumerate(clustered_data):

            cluster_x = cluster[0]
            cluster_y = cluster[1]
            w,b = self.LR(cluster_x, cluster_y)
            cluster_pred = [w*x+b for x in cluster_x]
            print(cluster_x, cluster_y, cluster_pred)
            cluster_error = mean_squared_error(cluster_y, cluster_pred, squared=False)
            [ys.append(y) for y in cluster_y]
            [preds.append(pred) for pred in cluster_pred]
        total_error = mean_squared_error(ys, preds,  squared=False)
        return total_error


    def calc_cluster_models(self, X, Y, clustered_data, distFunction=None):

        '''
        Utility function which fits a linear regression model to each cluster and returns the linear parameters.
        '''

        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)

        linear_params = []
        clusterPreds = []
        avg_intersection_cost = 0
        for i in range(len(clustered_datapoints)):
#            For cyclic features uncomment the following:
#            preds, w, b = self.CR.cyclicRegression(clustered_datapoints[i][0], clustered_datapoints[i][1])
            if len(clustered_datapoints[i][0]) > 0 and len(clustered_datapoints[i][1]) > 0:
                w,b = self.LR(clustered_datapoints[i][0], clustered_datapoints[i][1])
                linear_params.append([float(w),b])
        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        return clustered_data, linear_params, self.calculate_clustering_cost(clustered_data)

    def plotMedoids(self, clustered_data, medoids, linear_params, clustering_cost):
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
#    CR = CyclicRegression()
        inch_to_cm = 0.39
        colours = self.colours
        fig, axes = plt.subplots(1,1,figsize=(15*inch_to_cm,8*inch_to_cm))
        axes = fig.axes
        axes[0].set_xlabel(f'$x$', fontsize=11)
        axes[0].set_ylabel(f'$\hat{{y}}$', fontsize=11)

#        colours = []
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
            colour = np.random.rand(1,3)
#            colours.append(colour)
#            colour = colours[i]
            axes[0].scatter(clustered_data[i][0], clustered_data[i][1], s=1, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
            axes[0].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -10, 40, color=colour, label='_nolegend_')
            axes[0].plot(cluster_range, w*cluster_range+b, linewidth=0.5, c=colour)
        if clustering_cost:
            axes[0].set_title(f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {clustering_cost:.2f}', fontsize=11)

        try:
            axes[0].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2, fontsize=11)
        except:
            pass

#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig

    def plot_final_clustering(self, clustered_data, linear_params):
        cost = self.calculate_clustering_cost(clustered_data)
        fig = go.Figure()
        for cluster in range(len(clustered_data)):
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
            xs = self.x[clustered_data[cluster]]
            ys = self.y[clustered_data[cluster]]
            w, b = linear_params[cluster]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=4, color=colour)))
            cluster_range = np.linspace(min(xs), max(xs), 100)
            fig.add_trace(go.Scatter(x=cluster_range, y=w*cluster_range+b, mode='lines', line=dict(color=colour, width=5)))
        fig.update_layout(title=f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {cost:.2f}')
        return fig

