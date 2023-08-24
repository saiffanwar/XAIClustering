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

class LocalLinearRegression():

    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression()

    def pointwiseDistance(self, distanceFunction, X):
        xDs = []
        for x1 in X:
            x1Ds = []
            for x2 in X:
                x1Ds.append(distanceFunction(x1,x2))
            xDs.append(x1Ds)
        self.xDs = xDs

        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(self.xDs), np.min(self.xDs)
        self.xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), self.xDs))).reshape(self.N, self.N)


    def calculateLocalModels(self, distanceFunction):
        plotModelParameters = False

        w1 = []
        w2 = []
        w = []
        MSE = []

#        Calculate distances between all points so can reuse later.
        self.pointwiseDistance(distanceFunction, self.x)

        for i in tqdm(range(self.N)):
            check = [self.xDs_norm[i][j]<0.05 for j in range(self.N)]
            localxdata = list(compress(self.x, check))
            localydata = list(compress(self.y, check))
            if i ==0:
                self.first_point_neighbours = [localxdata, localydata]
            X = np.array([localxdata, np.ones(len(localxdata))]).T
            wlocal = np.linalg.lstsq(X, localydata, rcond=1)[0]
            w1.append(wlocal[0])
            w2.append(wlocal[1])
            w.append(wlocal)

#           LLR plotting for individual points
#            if i%10 ==0:
#                xrange = np.linspace(min(localxdata), max(localxdata), 100)
#                fig, axes = plt.subplots(1,1,figsize=(10,10))
#                axes.scatter(self.x, self.y, s=3)
#                axes.scatter(localxdata, localydata, s=5, c='red')
#                axes.scatter(self.x[i], self.y[i], s=100, c='green')
#                ys = [wlocal[0]*xrange[j]+wlocal[1] for j in range(100)]
#                axes.legend(['All Data', 'Local Data', 'Selected Point', 'Local Model'])
#                plt.plot(xrange, ys, c='green', linewidth=5)
#                plt.savefig('Figures/LocalModels/housingLinearRegression{}.png'.format(i))
            MSE.append(mean_squared_error(localydata, np.dot(X, wlocal)))

#            for circular regression uncomment the following:

#            preds, m, c = self.CR.cyclicRegression(localxdata, localydata)
#
#            w1.append(m)
#            w2.append(c)
#            w.append([m, c])
#            MSE.append(mean_squared_error(localydata, preds))


#        if plotModelParameters:
#            x = [x[i] for i in range(len(x))]
#            fig, axes = plt.subplots(1,3,figsize=(20,7))
#
#            fontsize=14
#
#            axes[0].scatter(x, MSE, s=5)
#            axes[0].set_xlabel('x', fontsize=fontsize)
#            axes[0].set_ylabel('MSE', fontsize=fontsize)
#
#            axes[1].scatter(x, w1, s=5)
#            axes[1].set_xlabel('x', fontsize=fontsize)
#            axes[1].set_ylabel('w1', fontsize=fontsize)
#
#            axes[2].scatter(x, w2, s=5)
#            axes[2].set_xlabel('x', fontsize=fontsize)
#            axes[2].set_ylabel('w2', fontsize=fontsize)
#
        return w1, w2, w, MSE


    def computeDistanceMatrix(self, w1, w2, w, MSE, distanceFunction, distance_weights= [1,0,0]):

        D = np.zeros((self.N,self.N))

        wDs, mseDs = [], []

        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
        print('Computing Distance Matrix...')
        [(wDs.append(euclidean(w[i], w[j])), mseDs.append(abs(MSE[i]-MSE[j]))) for i in tqdm(range(self.N)) for j in range(self.N)]

#        # Raw value Distances:
#        print('Computing Raw Distances...')
#        xDs = np.array([distanceFunction(self.x[i], self.x[j]) for i in tqdm(range(self.N)) for j in range(self.N)])
##        xDs = np.array(map(map(distanceFunction(xi, xj), self.x), self.x))
#        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
##        wDs = np.array([np.linalg.norm(np.array(w[j]).flatten() - np.array(w[j]).flatten()) for i in range(self.N) for j in range(self.N)])
#        print('Computing Weight Distances...')
#        wDs = np.array([euclidean(w[i], w[j]) for i in tqdm(range(self.N)) for j in range(self.N)])
##        wDs = np.array(map(map(euclidean(wi, wj), w), w))
#
#        print('Computing Error Distances...')
#        mseDs = np.array([MSE[i]-MSE[j] for i in tqdm(range(self.N)) for j in range(self.N)])
##        mseDs = np.array(map(map(lambda x1, x2: x1-x2, MSE), MSE))


        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)

        maxWds, minWds = np.max(wDs), np.min(wDs)
        wDs_norm = np.array(list(map(lambda x: normalise(maxWds, minWds, x), wDs))).reshape(self.N, self.N)

        maxMses, minMses = np.max(mseDs), np.min(mseDs)
        mseDs_norm = np.array(list(map(lambda x: normalise(maxMses, minMses, x), mseDs))).reshape(self.N, self.N)



        print('Distance Weights: ', distance_weights)
        for i in range(self.N):
            for j in range(self.N):
                # print(np.linalg.norm(w[i]-w[j]), MSE[i]-MSE[j], self.x[i]-self.x[j])
#                distance = wDs_norm[i,j] + mseDs_norm[i,j] + 0.25*self.xDs_norm[i,j]
                distance = distance_weights[0]*self.xDs_norm[i,j] + distance_weights[1]*wDs_norm[i,j] +  distance_weights[2]*mseDs_norm[i,j]
                D[i,j] = distance

        return D, self.xDs_norm

    def recluster(self, K, D,  clustered_data, medoids, clustering_cost, linear_params):
        random_cluster = random.choice(np.arange(0,K,1))
        prev_medoid = medoids[random_cluster]
        clustered_data[random_cluster].remove(prev_medoid)
        new_medoid = random.choice(clustered_data[random_cluster])
        new_medoids = deepcopy(medoids)
        new_medoids[random_cluster] = new_medoid
        clustered_data[random_cluster].append(prev_medoid)

        # Reset clustered data for new clustsering.
        new_clustered_data = [[] for i in range(K)]
        for index,i in enumerate(D):
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

    def adapted_clustering(self, K, D, xDs_norm, medoids=None, clustered_data=None, linear_params=None, clustering_cost=None):
        print(f'-------------- Clustering for {K} clusters --------------')
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
                        medoids = random.choices(np.arange(0,len(D),1), k=K)
                    # Reset clustered data for new clustsering.
                    clustered_data = [[] for i in range(K)]
                    minimum_distances = []
                    for index,i in enumerate(D):
                        minimum_distances.append(min([i[j] for j in medoids]))
                        closest = np.argmin([i[j] for j in medoids])
                        clustered_data[closest].append(index)


#                    print(minimum_distances, min(minimum_distances))


                    _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
                    clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
                    optimal_counter = 0


                else:
                    clustering_cost, linear_params, clustered_data, medoids = self.recluster(K, D, clustered_data, medoids, clustering_cost, linear_params)
#                    if optimal_counter == 100:
#                        break



#                if iter == 0:
#                    _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
#                else:
#                    _,new_linear_params = self.calc_cluster_models(self.x, self.y, new_clustered_data)
#
#                if iter == 0:
#                    clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
#                    optimal_counter = 0
#                else:
#                    new_clustering_cost = self.calculate_clustering_cost(new_clustered_data, new_linear_params)
#                    if new_clustering_cost < clustering_cost:
#                        clustering_cost = new_clustering_cost
#                        linear_params = new_linear_params
#                        clustered_data = new_clustered_data
#                        optimal_counter = 0
#                        fig = self.plotMedoids(self.cluster_indices_to_datapoints(clustered_data), medoids, linear_params, clustering_cost)
#                        fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.png')
#                    else:
#                        optimal_counter += 1
##                    if optimal_counter == 100:
##                        break

                iter += 1
#            except:
#                None

        clustered_data = self.check_cluster_overlap(clustered_data)
        _,linear_params = self.calc_cluster_models(self.x, self.y, clustered_data)
        clustering_cost = self.calculate_clustering_cost(clustered_data, linear_params)
        medoids = [random.choice(clustered_data[i]) for i in range(len(clustered_data))]
        print(f'Before {K}, after {len(clustered_data)} clusters')
        if len(clustered_data) < K:
            K = len(clustered_data)
            print(type(medoids))
            print(type(K))
            print(type(clustered_data))
            print(type(linear_params))
            print(type(clustering_cost))
            clustered_data, medoids, linear_params, clustering_cost = self.adapted_clustering(K, D, xDs_norm, medoids, clustered_data, linear_params, clustering_cost)
        else:
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.png')
        return clustered_data, medoids, linear_params, clustering_cost

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
            merged_clusters = self.merge_clusters(clustered_data, contained_clusters)

            return merged_clusters
        else:
            return clustered_data


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
        print(cluster_children)
        return cluster_children

    def merge_clusters(self, clustered_data, contained_clusters):
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


    def KMedoidClustering(self, K, D):
        km = kmedoids.KMedoids(n_clusters=K, init='random', random_state=0, method='pam')
        # c = kmedoids.fasterpam(D,K)
        c=km.fit(D)
        clustered_data = []
        for k in np.unique(c.labels_):
            clustersx = []
            clustersy = []
            for i in range(len(c.labels_)):
                if c.labels_[i] == k:
                    clustersx.append(self.x[i])
                    clustersy.append(self.y[i])

            clustered_data.append([clustersx, clustersy])

        orderedClusters = []
        minimums = [min(clustered_data[j][0]) for j in range(len(clustered_data))]
        minsCopy = minimums.copy()
        for i in range(len(clustered_data)):
            firstCluster = minimums.index(min(minsCopy))
            minsCopy.remove(min(minsCopy))
            orderedClusters.append(clustered_data[firstCluster])
        return orderedClusters

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


        axes[0].legend([str(i) for i in range(len(clustered_data))])
#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig
#    plotly_Fig.show()
