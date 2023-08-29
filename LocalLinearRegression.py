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

    def __init__(self, x,y, distFunction):
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression()
        if distFunction == 'Euclidean':
            self.distFunction = self.euclideanDefine()
        elif distFunction == 'Time':
            self.distFunction = self.timeDiff

    def euclideanDefine(self):
        maxVal = max(self.x)
        euclidean = lambda x1,x2: abs(x1-x2)/maxVal
        return euclidean

    def timeDiff(self, x1,x2):
        return abs(x1-x2).days

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

        return w1, w2, w, MSE


    def computeDistanceMatrix(self, w1, w2, w, MSE, distanceFunction, distance_weights= [1,0,0]):

        D = np.zeros((self.N,self.N))

        wDs, mseDs = [], []

        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
        print('Computing Distance Matrix...')
        [(wDs.append(euclidean(w[i], w[j])), mseDs.append(abs(MSE[i]-MSE[j]))) for i in tqdm(range(self.N)) for j in range(self.N)]

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



#    def KMedoidClustering(self, K, D):
#        km = kmedoids.KMedoids(n_clusters=K, init='random', random_state=0, method='pam')
#        # c = kmedoids.fasterpam(D,K)
#        c=km.fit(D)
#        clustered_data = []
#        for k in np.unique(c.labels_):
#            clustersx = []
#            clustersy = []
#            for i in range(len(c.labels_)):
#                if c.labels_[i] == k:
#                    clustersx.append(self.x[i])
#                    clustersy.append(self.y[i])
#
#            clustered_data.append([clustersx, clustersy])
#
#        orderedClusters = []
#        minimums = [min(clustered_data[j][0]) for j in range(len(clustered_data))]
#        minsCopy = minimums.copy()
#        for i in range(len(clustered_data)):
#            firstCluster = minimums.index(min(minsCopy))
#            minsCopy.remove(min(minsCopy))
#            orderedClusters.append(clustered_data[firstCluster])
#        return orderedClusters
