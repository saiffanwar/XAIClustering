
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

    def __init__(self, x, y, D, xDs):
        self.x = x
        self.y = y
        self.D = D
        self.xDs = xDs

        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(self.xDs), np.min(self.xDs)
        self.xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), self.xDs))).reshape(len(self.x), len(self.x))


    def recluster(self, K, clustered_data, medoids, clustering_cost, linear_params):
        random_cluster = random.choice(np.arange(0,K,1))
        prev_medoid = medoids[random_cluster]
        clustered_data[random_cluster].remove(prev_medoid)
        new_medoid = random.choice(clustered_data[random_cluster])
        new_medoids = deepcopy(medoids)
        new_medoids[random_cluster] = new_medoid
        clustered_data[random_cluster].append(prev_medoid)

        # Reset clustered data for new clustsering.
        new_clustered_data = [[] for i in range(K)]
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
            fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.png')
        return clustering_cost, linear_params, clustered_data, medoids

    def adapted_clustering(self, K, medoids=None, clustered_data=None, linear_params=None, clustering_cost=None):
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
#            try:
                if clustered_data == None:
                    if medoids == None:
                        medoids = random.choices(np.arange(0,len(self.D),1), k=K)
                    # Reset clustered data for new clustsering.
                    clustered_data = [[] for i in range(K)]
                    minimum_distances = []
                    for index,i in enumerate(self.D):
                        minimum_distances.append(min([i[j] for j in medoids]))
                        closest = np.argmin([i[j] for j in medoids])
                        clustered_data[closest].append(index)

                    _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
                    clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
                    optimal_counter = 0


                else:
                    clustering_cost, linear_params, clustered_data, medoids = self.recluster(K, clustered_data, medoids, clustering_cost, linear_params)

                iter += 1
#            except:
#                None

        #  Check if there is any clusters contained within other clusters. If so, merge the child cluster into the parent cluster and recalculate the           new cluster model parameters.
        # check_cluster_overlap also sorts the clusters into x order so neighbouring clusters are next to each other.

        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
        fig.savefig(f'Figures/Clustering/OptimisedClusters/preMerge_{K}.png')
        clustered_data = self.check_cluster_overlap(clustered_data)
        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
        clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
        medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]

        # Check if parameters of neighbouring clusters models are similar. If they are similar, merge them.
        merged_clusters = self.check_cluster_similarity(clustered_data, linear_params)
        if merged_clusters != None:
            clustered_data = merged_clusters
            clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
            medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]
            _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)

        # If the number of clusters has changed, pick new medoids from the same clusters and optimise further.
        if len(clustered_data) < K:
            K = len(clustered_data)
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/merged_{K}.png')
            return self.adapted_clustering(K, medoids, clustered_data, linear_params, clustering_cost)
        else:
            print(f'Saving figure for final {K} clustering.')
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/OptimisedClusters/final_{K}.png')
            return clustered_data, medoids, linear_params, clustering_cost

    def check_cluster_similarity(self, clustered_data, linear_params):

        merges = []

        for cluster_num, cluster_params in enumerate(linear_params):
            # Check similarity with next cluster.
            if cluster_num != len(linear_params)-1:
                next_cluster_params = linear_params[cluster_num+1]
#                similarity = np.linalg.norm(np.array(cluster_params) - np.array(next_cluster_params))
                similarity = abs(cluster_params[0] - next_cluster_params[0])
                if similarity < 0.06:
                    print(f'Similarity between {cluster_num} and {cluster_num+1} is {similarity}.')
                    merges.append([cluster_num, cluster_num+1])

        if merges != []:
            merged_clusters = deepcopy(clustered_data)
            for merge in merges:
                merged_clusters[merge[1]].extend(merged_clusters[merge[0]])
                merged_clusters[merge[0]] = []

            merged_clusters = [i for i in merged_clusters if i != []]
            return merged_clusters
        else:
            return None


    def order_clusters(self, clustered_data):
        minimums = [min(clustered_data[i]) for i in range(len(clustered_data))]
        index_sorted = sorted(range(len(minimums)), key=lambda k: minimums[k])
        ordered_clusters = [clustered_data[i] for i in index_sorted]

        return ordered_clusters

    def check_cluster_overlap(self, clustered_data):
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        cluster_ranges = [[min(cluster_datapoints[i][0]), max(cluster_datapoints[i][0])] for i in range(len(cluster_datapoints))]
        contained_clusters, overlapping_clusters = [], []
        for cluster1, range1 in enumerate(cluster_ranges):
            for cluster2, range2 in enumerate(cluster_ranges):
                if range1[0] < range2[0] < range1[1] or range1[0] < range2[1] < range1[1]:
                    if range1[0] < range2[0] < range1[1] and range1[0] < range2[1] < range1[1]:
                        print(f'Cluster {cluster2} is contained within Cluster {cluster1}.')
                        contained_clusters.append([cluster1, cluster2])
                    else:

                        print(f'Overlap between cluster {cluster1} and {cluster2}.')
                        print(f'Cluster {cluster1} has range {range1}.')
                        print(f'Cluster {cluster2} has range {range2}.')

        if len(contained_clusters) > 0:
            merged_clusters = self.adopt_clusters(clustered_data, contained_clusters)
            return self.order_clusters(merged_clusters)
        else:
            return self.order_clusters(clustered_data)


    def create_cluster_heirarchy(self, contained_clusters):

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
        datapoints = [[[],[]] for i in range(len(clustered_indices))]
        for cluster in range(len(clustered_indices)):
            for i, point_index in enumerate(clustered_indices[cluster]):
                datapoints[cluster][0].append(self.x[point_index])
                datapoints[cluster][1].append(self.y[point_index])
        return datapoints

    def calculate_clustering_cost(self, clustered_data, linear_params):
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
            try:
                cluster_range = np.arange(min(clustered_data[i][0]), max(clustered_data[i][0]), 1)
            except:
                print(clustered_data[i])
            axes[0].plot(cluster_range, w*cluster_range+b, linewidth=5, c=colour)

        for index, i in enumerate(medoids):
            axes[0].scatter(self.x[i], self.y[i], s=100, marker='X', c=colours[index], label='_nolegend_')
            axes[0].set_title(f'Clustering Cost: {clustering_cost}')


        axes[0].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2)
#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig
#    plotly_Fig.show()

