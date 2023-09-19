
from os import wait
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import compress
from sklearn.metrics import mean_squared_error
import kmedoids
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from cyclicRegression import CyclicRegression
from copy import deepcopy
import os
import glob

class LinearClustering():

    ''' This class clusters a dataset into regions of linearity. It is provided a distance matrix computed in the LocalLinearRegression class.
    This distance matrix is based on the local linear models of each point as well as the raw distance values. '''

    def __init__(self, x, y, D, xDs):
        self.x = x
        self.y = y
        self.D = D
        self.xDs = xDs

        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(self.xDs), np.min(self.xDs)
        self.xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), self.xDs))).reshape(len(self.x), len(self.x))


    def adapted_clustering(self, K, medoids=None, clustered_data=None, linear_params=None, clustering_cost=None):

        '''
        This is the main clustering algorithm. It will start with some arbitrary number of K to generate clusters.
        It takes a K-medoids based approach to generate clusters using the provided distance matrix. It then checks if there
        are any clusters contained within other clusters. If so, it merges the child cluster into the parent cluster.
        Then it checks if there are any neighbouring clusters which have a similar linearity and merges those that do.
        Then based on the new number of clusters after merging, it will recluster the data to find the optimal clustering.
        This is done recursively until there are no new merges therefore an optimal value of K is found.
        '''

        print(f'-------------- Clustering for K = {K} --------------')
        # Clear folder for figures to show evolution of clustering.
        if os.path.isdir(f'Figures/Clustering/{K}'):
            pass
        else:
            os.makedirs(f'Figures/Clustering/{K}')
        files = glob.glob(f'Figures/Clustering/{K}/*')
        for f in files:
            os.remove(f)


        num_iterations = 100
        iter = 0
        self.colours = [np.random.rand(1,3) for i in range(K)]
        while iter < num_iterations:
                if clustered_data == None:
                        if medoids == None:
                            medoids = random.choices(np.arange(0,len(self.D),1), k=K)
                        # Reset clustered data for new clustsering.
                        clustered_data = [[med] for med in medoids]
                        minimum_distances = []
                        for index,i in enumerate(self.D):
                            if index not in medoids:
#                                if index in medoids:
#                                    for med in medoids:
#                                        print(f'Distance between {index} and {med}',i[med])
#                                    print(np.argmin([i[j] for j in medoids]))
                                minimum_distances.append(min([i[j] for j in medoids]))
                                closest = np.argmin([i[j] for j in medoids])
                                clustered_data[closest].append(index)

#                        clustered_data = [cluster for cluster in clustered_data if cluster != []]
                        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
                        clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
                        optimal_counter = 0

                else:
                    # Recluster the data to find the optimal clustering.
                    clustering_cost, linear_params, clustered_data, medoids = self.recluster(K, clustered_data, medoids, clustering_cost, linear_params)

                    iter += 1

        #  Check if there is any clusters contained within other clusters. If so, merge the child cluster into the parent cluster and recalculate the           new cluster model parameters.

        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/before_merging_{K}.pdf')
        clustered_data = self.check_cluster_containments(clustered_data)
        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)


        # Check if neighbouring clusters overlap.
        fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost=100)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/before_checking_overlaps_{K}.pdf')
        clustered_data = self.check_cluster_overlap(clustered_data, linear_params, K)
        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)


        # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
        fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost=100)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/after_checking_overlaps_{K}.pdf')
        clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
        medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]


        # Check size of clusters. If 10x smaller than neighbouring clusters, merge them.
        clustered_data = self.check_cluster_sparsity(clustered_data)
        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
        fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost=100)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/after_checking_sparsity_{K}.pdf')

        # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
        merged_clusters = self.check_cluster_similarity(clustered_data, linear_params)
        if merged_clusters != None:
            clustered_data = merged_clusters
            clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
            medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]
            _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/after_similarity_merge_{K}.pdf')

        # If the number of clusters has changed, pick new medoids from the same clusters and optimise further.
        if len(clustered_data) < K:
            K = len(clustered_data)
#            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
#            fig.savefig(f'Figures/Clustering/OptimisedClusters/merged_{K}.pdf')
            return self.adapted_clustering(K, medoids, clustered_data, linear_params, clustering_cost)
        else:
            print(f'Saving figure for final {K} clustering.')
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/final_{K}.pdf')
            return clustered_data, medoids, linear_params, clustering_cost


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

        _,new_linear_params = self.calc_cluster_models(self.x, self.y, new_clustered_data)
        new_clustering_cost = self.calculate_clustering_cost(new_clustered_data, new_linear_params)
        if new_clustering_cost < clustering_cost:
            clustering_cost = new_clustering_cost
            linear_params = new_linear_params
            clustered_data = new_clustered_data
            medoids = new_medoids
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.pdf')
        return clustering_cost, linear_params, clustered_data, medoids


    def check_cluster_similarity(self, clustered_data, linear_params):

        '''
        Utility function which checks the similarity between linear models of neighbouring clusters.
        If the gradient of neighbouring models is similar (based on some defined threshold), the clusters
        are merged.
        '''

        merges = []
        original_clustering = deepcopy(clustered_data)
        current_clustering = deepcopy(clustered_data)
        for cluster_num, cluster_params in enumerate(linear_params):
            # Check similarity with next cluster.
            if cluster_num < len(current_clustering)-1:
                next_cluster_params = linear_params[cluster_num+1]
#                similarity = np.linalg.norm(np.array(cluster_params) - np.array(next_cluster_params))
                if cluster_params[0] != 0 and next_cluster_params[0] != 0:
                    similarity = abs(cluster_params[0] - next_cluster_params[0])
                    print(f'{cluster_num}: {cluster_params},{cluster_num+1}: {next_cluster_params}, Similarity: {similarity}')
                    # The similarity threshold is defined here. Increasing the thresold will result in more clusters being merged.
                    if similarity < 0:
                        print(f'Similarity between {cluster_num} and {cluster_num+1} is {similarity}.')
                        for i in range(len(current_clustering)):
                            if i == cluster_num:
                                current_clustering[i].extend(current_clustering[i+1])
                                current_clustering.remove(current_clustering[i+1])
                                _, linear_params = self.calc_cluster_models(self.x, self.y, current_clustering)




        return current_clustering





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
                            print(f'Cluster {cluster2} is contained within Cluster {cluster1}.')
                            contained_clusters.append([cluster1, cluster2])


        if len(contained_clusters) > 0:
            merged_clusters = self.adopt_clusters(clustered_data, contained_clusters)
            return self.order_clusters(merged_clusters)
        else:
            return self.order_clusters(clustered_data)


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
                    midpoint = range2[0] + (range1[1] - range2[0])/2
                    print(f'Cluster {cluster_num} and Cluster {cluster_num+1} overlap. Splitting at {midpoint}.')

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
#                    print(midpoint, new1xs, new2xs)
                    clustered_data[cluster_num] = new_cluster1
                    clustered_data[cluster_num+1] = new_cluster2

                    _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
                    fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost=100)
                    fig.savefig(f'Figures/Clustering/OptimisedClusters/merged_{cluster_num}_{cluster_num+1}_{K}.pdf')
        return clustered_data

    def check_cluster_sparsity(self, clustered_data, sparsity_threshold=0.10):


        largest_cluster_size = max([len(cluster) for cluster in clustered_data])
        contained_clusters = []
        for cluster_num, cluster in enumerate(clustered_data):
            if cluster_num < len(clustered_data)-1:
                if len(cluster) < sparsity_threshold*largest_cluster_size:
                    print(f'Cluster {cluster_num} is sparse. Combining with Cluster {cluster_num+1}.')
                    contained_clusters.append([cluster_num+1, cluster_num])

            if cluster_num == len(clustered_data)-1:
                if len(cluster) < sparsity_threshold*largest_cluster_size:
                    print(f'Cluster {cluster_num} is sparse. Combining with Cluster {cluster_num-1}.')
                    contained_clusters.append([cluster_num-1, cluster_num])

        merged_clusters = self.adopt_clusters(clustered_data, contained_clusters)


        return merged_clusters

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
                    print(f'Merging cluster {cluster} into cluster {parent}.')
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
                merged_clusters[parent].extend(clustered_data[child])
                merged_clusters[child] = []

        new_clusters = list(merged_clusters.values())
        new_clusters = [i for i in new_clusters if i != []]
        return new_clusters

    def cluster_indices_to_datapoints(self, clustered_indices):

        '''
        Utility function which converts the clusters containing indices of the x values to the raw x datapoint values.
        '''

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
            cluster_error = np.sum([abs(y_hat-y) for y_hat,y in zip(cluster_pred, cluster_y)])
#            print(f'Cluster {cluster_num} error:', cluster_error)
            total_error += cluster_error

        return total_error


    def calc_cluster_models(self, X, Y, clustered_data, distFunction=None):

        '''
        Utility function which fits a linear regression model to each cluster and returns the linear parameters.
        '''

        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
        def LR(x, y):
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_

        linear_params = []
        clusterPreds = []
        avg_intersection_cost = 0
        for i in range(len(clustered_data)):
#            For cyclic features uncomment the following:
#            preds, w, b = self.CR.cyclicRegression(clustered_data[i][0], clustered_data[i][1])
            w,b = LR(clustered_data[i][0], clustered_data[i][1])
            linear_params.append([float(w),b])
        return clustered_data, linear_params

    def plotMedoids(self, clustered_data, medoids, linear_params, clustering_cost):
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
#    CR = CyclicRegression()
#    plotly_Fig = None
        colours = self.colours
        fig, axes = plt.subplots(1,1,figsize=(10,10))
        axes = fig.axes
#        colours = []
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
#            colour = np.random.rand(1,3)
#            colours.append(colour)
            colour = colours[i]
            axes[0].scatter(clustered_data[i][0], clustered_data[i][1], s=5, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
            axes[0].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -20, 20, color=colour, label='_nolegend_')
            axes[0].plot(cluster_range, w*cluster_range+b, linewidth=5, c=colour)
            if medoids != None:
                axes[0].scatter(self.x[medoids[i]], self.y[i], s=100, marker='X', c=colour, label='_nolegend_')

        axes[0].set_title(f'Clustering Cost: {clustering_cost}')


        axes[0].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2)
#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig
#    plotly_Fig.show()

