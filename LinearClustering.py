from os import wait
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import compress
from pandas.core.base import NoNewAttributesMixin
from sklearn.metrics import mean_squared_error
import kmedoids
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from cyclicRegression import CyclicRegression
from similaritychecker import calculate_line_similarity
from copy import deepcopy
import os
import glob
import time


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})


def LR(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_

class LinearClustering():

    ''' This class clusters a dataset into regions of linearity. It is provided a distance matrix computed in the LocalLinearRegression class.
    This distance matrix is based on the local linear models of each point as well as the raw distance values. '''

    def __init__(self, x, y, D, xDs, feature, K):
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

        # Clear folder for figures to show evolution of clustering.
        if os.path.isdir(f'Figures/Clustering/OptimisedClusters/{self.feature_name}'):
            pass
        else:
            os.makedirs(f'Figures/Clustering/OptimisedClusters/{self.feature_name}')

        files = glob.glob(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/*')
        for f in files:
            os.remove(f)

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
        print(f'-------------- Clustering for K = {K} --------------')
        num_iterations = 1000
        iter = 0
        self.colours = [np.random.rand(1,3) for i in range(K)]

        new_clustering_cost = None
#        init_clusters = True
        while iter < num_iterations:

            if init_clusters == True:
                if self.medoids == None:
                    medoids = random.choices(np.arange(0,len(self.D),1), k=K)
#                    medoids = np.linspace(min(self.x), max(self.x), K)
#                    medoids = [np.argmin(np.abs(self.x - med)) for med in medoids]
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
                new_clustering_cost, linear_params, clustered_data, medoids = self.recluster(K, clustered_data, medoids, clustering_cost, linear_params)
            if new_clustering_cost:
                if new_clustering_cost == clustering_cost:
                    print('Ending clustering',  iter)
                    iter =num_iterations
                    break
                else:
                    clustering_cost = new_clustering_cost
            if len(clustered_data) == 1:
                break
            iter+=1



        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/before_verifying_{K}.pdf', bbox_inches='tight')
        if len(clustered_data)== 1:
            self.clustered_data = deepcopy(clustered_data)
            self.medoids = deepcopy(medoids)
            self.linear_params = deepcopy(linear_params)
            self.clustering_cost = deepcopy(clustering_cost)
            print(f'Saving figure for final {K} clustering.')
            fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/final_{K}.pdf', bbox_inches='tight')
            clustered_datapoints = self.cluster_indices_to_datapoints(self.clustered_data)
            return clustered_datapoints, self.medoids, self.linear_params, self.clustering_cost, fig

        if self.clustering_cost == None:
            accept_clustering = True
        else:
            if clustering_cost < self.clustering_cost:
                accept_clustering = True
            else:
                accept_clustering = True

        if accept_clustering == True:
            pre_verify_k = len(clustered_data)
            clustered_data, medoids, linear_params, clustering_cost = self.verify_clustering(clustered_data, linear_params, K)
            while len(clustered_data) < pre_verify_k:
                pre_verify_k = len(clustered_data)
                clustered_data, medoids, linear_params, clustering_cost = self.verify_clustering(clustered_data, linear_params, K)
            self.clustered_data = deepcopy(clustered_data)
            self.medoids = deepcopy(medoids)
            self.linear_params = deepcopy(linear_params)
            self.clustering_cost = deepcopy(clustering_cost)


        fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_verifying_{K}_merged_to_{len(self.clustered_data)}.pdf', bbox_inches='tight')

        # If the number of clusters has changed, pick new medoids from the same clusters and optimise further.
        if len(self.clustered_data) < K:
            K = len(self.clustered_data)
            self.K = K
#            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
#            fig.savefig(f'Figures/Clustering/OptimisedClusters/merged_{K}.pdf')
            return self.adapted_clustering(False, self.clustered_data, self.medoids,self.clustering_cost, self.linear_params)
        else:
            print(f'Saving figure for final {K} clustering.')
            fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/final_{K}.pdf', bbox_inches='tight')
            clustered_datapoints = self.cluster_indices_to_datapoints(self.clustered_data)
            return clustered_datapoints, self.medoids, self.linear_params, self.clustering_cost, fig

        #  Check if there is any clusters contained within other clusters. If so, merge the child cluster into the parent cluster and recalculate the           new cluster model parameters.


    def verify_clustering(self, clustered_data, linear_params, K):
        if len(clustered_data) != 1:
            clustered_data = self.check_cluster_containments(clustered_data)
            _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_merging_{time.time()}.pdf', bbox_inches='tight')
        if len(clustered_data) != 1:
            # Check size of clusters. If 10x smaller than neighbouring clusters, merge them.
            clustered_data = self.check_cluster_sparsity(clustered_data, linear_params)
            _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_checking_sparsity_{time.time()}.pdf', bbox_inches='tight')
        if len(clustered_data) != 1:
            # Check if neighbouring clusters overlap.
            clustered_data = self.check_cluster_overlap(clustered_data, linear_params, len(clustered_data))
            _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_checking_overlaps_{time.time()}.pdf', bbox_inches='tight')
        if len(clustered_data) != 1:
            # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
            clustered_data = self.check_cluster_similarity(clustered_data, linear_params)
            _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            fig = self.plotMedoids(clustered_data, None, linear_params, None)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_similarity_merge_{time.time()}.pdf', bbox_inches='tight')
        if len(clustered_data) != 1:
            clustered_data = self.check_cluster_coverage(clustered_data)
            _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            fig = self.plotMedoids(clustered_data, None, linear_params, None)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/after_checking_covering_{time.time()}.pdf', bbox_inches='tight')
            # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
        clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
        medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]

        return clustered_data, medoids, linear_params, clustering_cost

    def recluster(self, K, clustered_data, medoids, clustering_cost, linear_params):

        '''
        This is a utility function used in the adapted clustering algorithm. It will randomly select a cluster
        and then randomly select a new medoid for the cluster and then assigning all points the appropriate cluster
        based on the new medoid. Linear Regression models are fit to all of the new clusters. A cost is calculated
        based on the combined error of all the new LR models. If this cost is lower than the previous cost, the new
        clustering is favoured. This function is called repeatedly for a specified number of iterations.
        '''

        K = len(clustered_data)
        random_cluster = random.choice(np.arange(0,K,1))

        while len(clustered_data[random_cluster]) == 1:
            random_cluster = random.choice(np.arange(0,K,1))

        prev_medoid = medoids[random_cluster]
        clustered_data[random_cluster].remove(prev_medoid)

        new_medoid = random.choice(clustered_data[random_cluster])
        new_medoids = deepcopy(medoids)
        new_medoids[random_cluster] = new_medoid
        clustered_data[random_cluster].append(prev_medoid)



        # Reset clustered data for new clustsering.
        new_clustered_data = [[new_med] for new_med in new_medoids]
        for index,i in enumerate(self.D):
            closest = np.argmin([i[j] for j in new_medoids])
            new_clustered_data[closest].append(index)

        _,new_linear_params, new_clustering_cost = self.calc_cluster_models(self.x, self.y, new_clustered_data)
#        new_clustered_data, new_medoids, new_linear_params, new_clustering_cost = self.verify_clustering(new_clustered_data, new_linear_params, K)
#        if new_clustering_cost < clustering_cost:
        clustering_cost = new_clustering_cost
        linear_params = new_linear_params
        clustered_data = new_clustered_data
        medoids = new_medoids
#        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
#        fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.pdf')
        return clustering_cost, linear_params, clustered_data, medoids


    def check_cluster_similarity(self, clustered_data, linear_params):

        '''
        Utility function which checks the similarity between linear models of neighbouring clusters.
        If the gradient of neighbouring models is similar (based on some defined threshold), the clusters
        are merged.
        '''
        def LR(x, y):
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_

        new_clustering = deepcopy(clustered_data)
        removed = []
        for i in range(len(clustered_data)-1):
            if i not in removed and (i+1) not in removed:
                cluster_datapoints = self.cluster_indices_to_datapoints(new_clustering)
                similarity = calculate_line_similarity(self.x, self.y,i, i+1, cluster_datapoints[i][0], cluster_datapoints[i+1][0], linear_params[i], linear_params[i+1], plotting=False, feature_name=self.feature_name)
                if similarity < 0.1 or abs(linear_params[i][0]-linear_params[i+1][0]) < 0.2*(linear_params[i][0]):
#                if abs(linear_params[i][0]-linear_params[i+1][0]) < 0.2*(linear_params[i][0]):
#                    print(f'Similarity between {i} {linear_params[i]} and {i+1} {linear_params} is {similarity}.')
#                    similarity = calculate_line_similarity(i, i+1, cluster_datapoints[i][0], cluster_datapoints[i+1][0], linear_params[i], linear_params[i+1], plotting=True)
                    new_clustering[i+1] = clustered_data[i+1] + clustered_data[i]
                    new_clustering[i] = []
                    # Recalculate the parameters for the single cluster. Need to find the new datapoints first.
                    xs, ys = self.cluster_indices_to_datapoints(new_clustering[i+1])
                    linear_params[i+1] = LR(xs, ys)
                    linear_params[i] = []

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

    def check_cluster_containments(self, clustered_data):

        '''
        Utility function which checks for overlap between clusters or (child) clusters entirely contained
        within other (parent) clusters. If there is overlap, the parent cluster adopts the child cluster.
        '''

        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        cluster_ranges = [[min(cluster_datapoints[i][0]), max(cluster_datapoints[i][0])] for i in range(len(cluster_datapoints))]
        contained_clusters, overlapping_clusters = [], []
        for cluster1, range1 in enumerate(cluster_ranges):
            for cluster2, range2 in enumerate(cluster_ranges):
                if cluster1 != cluster2:
                    if range1[0] <= range2[0] <= range1[1] or range1[0] <= range2[1] <= range1[1]:
                        if range1[0] <= range2[0] <= range1[1] and range1[0] <= range2[1] <= range1[1]:
                            #                            print(f'Cluster {cluster2} is contained within Cluster {cluster1}.')
                            contained_clusters.append([cluster1, cluster2])


        if len(contained_clusters) > 0:
            merged_clusters = self.adopt_clusters(clustered_data, contained_clusters)
            return self.order_clusters(merged_clusters)
        else:
            return self.order_clusters(clustered_data)


    def check_cluster_coverage(self, clustered_data):

        data_x_range = max(self.x)-min(self.x)
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        for i in range(len(clustered_datapoints)-1):
            if (min(clustered_datapoints[i+1][0])-min(clustered_datapoints[i][0])) < (1/self.init_K)*data_x_range:
                clustered_data[i+1] = clustered_data[i+1] + clustered_data[i]
                clustered_data[i] = []

        print(np.shape(clustered_datapoints))
#        print(max(clustered_datapoints[len(clustered_datapoints)][0])-max(clustered_datapoints[len(clustered_datapoints)-1][0]))
#        if (max(clustered_datapoints[len(clustered_datapoints)-1][0])-max(clustered_datapoints[len(clustered_datapoints)-2][0])) < (1/self.init_K)*data_x_range:
#            clustered_data[-1] = clustered_data[-2] + clustered_data[-1]
#            clustered_data[-1] = []
        clustered_data = [cluster for cluster in clustered_data if cluster != []]

        return clustered_data



    def check_cluster_overlap(self, clustered_data, linear_params, K):
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        cluster_ranges = [[min(cluster_datapoints[i][0]), max(cluster_datapoints[i][0])] for i in range(len(cluster_datapoints))]
        new_clustering = deepcopy(clustered_data)
        for cluster_num, cluster in enumerate(clustered_data):
            if cluster_num < len(clustered_data)-1:
                range1 = cluster_ranges[cluster_num]
                range2 = cluster_ranges[cluster_num+1]
                # If the beginning of the following cluster falls within this cluster, seperate them through the midpoint of the intersection.
                if range1[0] <= range2[0] <= range1[1]:
                    fig, axes = plt.subplots(1,2, figsize=(20*0.39,4*0.39))
                    midpoint = range2[0] + (range1[1] - range2[0])/2

                    axes[0].scatter(self.x[cluster], self.y[cluster], color='blue', s=1)
                    m, c = LR(self.x[cluster], self.y[cluster])
                    axes[0].plot(self.x[cluster], m*self.x[cluster]+c, color='blue', linewidth=0.5)
                    axes[0].scatter(self.x[clustered_data[cluster_num+1]], self.y[clustered_data[cluster_num+1]], color='red', s=1)
                    m, c = LR(self.x[clustered_data[cluster_num+1]], self.y[clustered_data[cluster_num+1]])
                    axes[0].plot(self.x[clustered_data[cluster_num+1]], m*self.x[clustered_data[cluster_num+1]]+c, color='red', linewidth=0.5)
                    axes[0].plot([midpoint, midpoint], [0, 1], color='black', linestyle='--')

                    both_cluster_indices = cluster + clustered_data[cluster_num+1]

                    new_cluster1, new_cluster2 = [], []

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

                    axes[1].scatter(new1xs, [self.y[i] for i in new_cluster1], color='blue', s=1)
                    m, c = LR(new1xs, [self.y[i] for i in new_cluster1])
                    axes[1].plot(new1xs, m*new1xs+c, color='blue', linewidth=0.5)
                    axes[1].scatter(new2xs, [self.y[i] for i in new_cluster2], color='red', s=1)
                    m, c = LR(new2xs, [self.y[i] for i in new_cluster2])
                    axes[1].plot(new2xs, m*new2xs+c, color='red', linewidth=0.5)
                    axes[1].plot([midpoint, midpoint], [0, 1], color='black', linestyle='--')
                    for ax in [0,1]:
                        axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
                        axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
                        axes[ax].set_xlabel(r'$x$', fontsize=11)
                        axes[ax].set_xticklabels(axes[ax].get_xticklabels(), fontsize=11)
                    axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
                    axes[1].set_ylabel('')
                    axes[1].set_yticklabels('')
                    fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/overlap_{cluster_num}_{cluster_num+1}_{K}.pdf', bbox_inches='tight')
#                    fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
#                    fig.savefig(f'Figures/Clustering/OptimisedClusters/overlap_{cluster_num}_{cluster_num+1}_{K}.pdf')
        return clustered_data

    def check_cluster_sparsity(self, clustered_data, linear_params, sparsity_threshold=0.10):


        largest_cluster_size = max([len(cluster) for cluster in clustered_data])
        new_clustering = deepcopy(clustered_data)
        for i in range(len(clustered_data)):
            if len(new_clustering[i]) < sparsity_threshold*largest_cluster_size:
                if i == 0 or new_clustering[i-1] == []:
                    if i ==len(clustered_data)-1:
                        j = i-1
                        while new_clustering[j] == []:
                            j -= 1
                        new_clustering[j] = clustered_data[j] + clustered_data[i]
                    else:
                        new_clustering[i+1] = clustered_data[i+1] + clustered_data[i]
                    new_clustering[i] = []
                elif i == len(clustered_data)-1 or new_clustering[i+1] == []:
                    new_clustering[i-1] = clustered_data[i-1] + clustered_data[i]
                    new_clustering[i] = []
                else:

                    temp_prev_cluster =

#                    cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
#
#                    successor_similarity = calculate_line_similarity(self.x, self.y,i, i+1, cluster_datapoints[i][0], cluster_datapoints[i+1][0], linear_params[i], linear_params[i+1], feature_name=self.feature_name)
#                    predecessor_similarity = calculate_line_similarity(self.x, self.y,i, i-1, cluster_datapoints[i][0], cluster_datapoints[i+1][0], linear_params[i], linear_params[i-1], feature_name=self.feature_name)
#                    if successor_similarity < predecessor_similarity:
#                        new_clustering[i+1] = clustered_data[i+1] + clustered_data[i]
#                        new_clustering[i] = []
#                    else:
#                        new_clustering[i-1] = clustered_data[i-1] + clustered_data[i]
#                        new_clustering[i] = []
        clustered_data = [cluster for cluster in new_clustering if cluster != []]

        return clustered_data

    def create_cluster_heirarchy(self, contained_clusters):

        '''
        Utility function which creates a heirarchy of clusters based on the contained_clusters list.
        Clusters contained within clusters that are also children of other clusters, are all merged into
        the top level parent cluster.
        '''

        all_clusters = np.unique(np.array(contained_clusters).flatten())
        cluster_children = {cluster:[] for cluster in all_clusters}

        all_parents = [j[0] for j in contained_clusters]
        all_children = [j[1] for j in contained_clusters]
        for i in contained_clusters:
            for j in all_parents:
                if i[0] == j:
                    cluster_children[j].append(all_children[all_parents.index(j)])
                    all_children.remove(all_children[all_parents.index(j)])
                    all_parents.remove(j)

        for cluster, children in cluster_children.items():
            all_children = {j:cluster_children[j] for j in all_clusters}
            for parent, child in all_children.items():
                if cluster in child:
                    cluster_children[parent].extend(cluster_children[cluster])
                    cluster_children[cluster] = []
        return cluster_children

    def adopt_clusters(self, clustered_data, contained_clusters):

        '''
        Utility function which merges clusters contained within other clusters into the parent cluster.
        '''

        merges = self.create_cluster_heirarchy(contained_clusters)
        merged_clusters = {i:clustered_data[i] for i in range(len(clustered_data))}
        for parent, children in merges.items():
            for child in children:
#                fig, axes = plt.subplots(1,2, figsize=(20*0.39,4*0.39))
#                axes[0].scatter(self.x[merged_clusters[parent]], self.y[merged_clusters[parent]], color='blue', s=1, label='Parent Cluster')
#                axes[0].scatter(self.x[merged_clusters[child]], self.y[merged_clusters[child]], color='red', s=1, label='Child Cluster')
                merged_clusters[parent].extend(clustered_data[child])
                merged_clusters[child] = []
#                axes[1].scatter(self.x[merged_clusters[parent]], self.y[merged_clusters[parent]], color='blue', s=1)


#                for ax in [0,1]:
#                    axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
#                    axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
#                    axes[ax].set_xlabel(r'$x$', fontsize=11)
#                    axes[ax].set_xticklabels(axes[ax].get_xticklabels(), fontsize=11)
#                axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
#                axes[1].set_ylabel('')
#                axes[1].set_yticklabels('')
#
#
#                fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=11)
#                fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/adopted_{parent}_{child}.pdf', bbox_inches='tight')

        new_clusters = list(merged_clusters.values())
        new_clusters = [i for i in new_clusters if i != []]
        return new_clusters

    def cluster_indices_to_datapoints(self, clustered_indices):

#        try:

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
                        datapoints[cluster][0].append(self.x[point_index])
                        datapoints[cluster][1].append(self.y[point_index])
                return datapoints

    def calculate_clustering_cost(self, clustered_data, linear_params):

        '''
        Utility function which calculates the cost of a clustering based on the combined error of the linear regression models
        for each cluster. The cost is the sum of the absolute errors of each cluster.
        '''

        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
        total_error = 0
        for cluster_num, cluster in enumerate(clustered_data):

            cluster_x = cluster[0]
            cluster_y = cluster[1]
            w,b = linear_params[cluster_num]
            cluster_pred = [w*x+b for x in cluster_x]
#            cluster_error = np.sum([abs(y_hat-y) for y_hat,y in zip(cluster_pred, cluster_y)])
            cluster_error = mean_squared_error(cluster_y, cluster_pred, squared=False)
#            print(f'Cluster {cluster_num} error:', cluster_error)
            total_error += cluster_error
        total_error = total_error/len(clustered_data)
        return total_error


    def calc_cluster_models(self, X, Y, clustered_data, distFunction=None):

        '''
        Utility function which fits a linear regression model to each cluster and returns the linear parameters.
        '''

        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        def LR(x, y):
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_

        linear_params = []
        clusterPreds = []
        avg_intersection_cost = 0
        for i in range(len(clustered_datapoints)):
#            For cyclic features uncomment the following:
#            preds, w, b = self.CR.cyclicRegression(clustered_datapoints[i][0], clustered_datapoints[i][1])
            if len(clustered_datapoints[i][0]) > 0 and len(clustered_datapoints[i][1]) > 0:
                w,b = LR(clustered_datapoints[i][0], clustered_datapoints[i][1])
                linear_params.append([float(w),b])
        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        return clustered_data, linear_params, self.calculate_clustering_cost(clustered_data, linear_params)

    def plotMedoids(self, clustered_data, medoids, linear_params, clustering_cost):
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
#    CR = CyclicRegression()
#    plotly_Fig = None
        inch_to_cm = 0.39
        colours = self.colours
        fig, axes = plt.subplots(1,1,figsize=(10*inch_to_cm,4*inch_to_cm))
        axes = fig.axes
        axes[0].set_xlabel(f'$x$', fontsize=11)
        axes[0].set_ylabel(f'$\hat{{y}}$', fontsize=11)

#        colours = []
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
#            colour = np.random.rand(1,3)
#            colours.append(colour)
            colour = colours[i]
            axes[0].scatter(clustered_data[i][0], clustered_data[i][1], s=1, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
#            axes[0].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -20, 20, color=colour, label='_nolegend_')
            axes[0].plot(cluster_range, w*cluster_range+b, linewidth=0.5, c=colour)
#            if medoids != None:
#                axes[0].scatter(self.x[medoids[i]], self.y[medoids[i]], s=20, marker='X', c=colour, label='_nolegend_')
#        clustering_cost = None
        if clustering_cost:
#            axes[0].text(0,1.05, f'Clustering Cost: {clustering_cost}', fontsize=11)
            axes[0].set_title(f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {clustering_cost:.2f}', fontsize=11)

#        try:
#            axes[0].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(1, 1.05), ncol=len(clustered_data)/2, fontsize=11)
#        except:
#            pass

#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig
#    plotly_Fig.show()

