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
import pickle as pck
from similaritychecker import calculate_line_similarity

class LocalLinearRegression():
    '''
    This class performs local linear regression on a given dataset. It takes in the data and a distance function
    which is used to calculate the neighbourhood of points to include in each points local linear regression.
    The current neighbourhood is set to 5% of the maximum distance.
    '''
    def __init__(self, x,y, dist_function):
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression()
        if dist_function == 'Euclidean':
            self.dist_function = self.euclideanDefine()
        elif dist_function == 'Time':
            self.dist_function = self.timeDiff

    def euclideanDefine(self):
        ''' Defines a eculidea distance function and normalises the distance based on the maximum value in the dataset.'''
        maxVal = max(self.x)
        euclidean = lambda x1,x2: abs(x1-x2)/maxVal
        return euclidean

    def timeDiff(self, x1,x2):
        ''' A time difference function which takes in datetime objects and returns the difference in days.'''
        return abs(x1-x2).days

    def pointwiseDistance(self, X):
        ''' Calculates the distances between every point and every other point based on the defined distance function.
        Returns a matrix of all the distances which are normalised to range between 0 and 1.'''
        xDs = []
        for x1 in X:
            x1Ds = []
            for x2 in X:
                x1Ds.append(self.dist_function(x1,x2))
            xDs.append(x1Ds)
        self.xDs = xDs

        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(self.xDs), np.min(self.xDs)
        self.xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), self.xDs))).reshape(self.N, self.N)


    def calculateLocalModels(self):
        ''' Calculates the local linear model for each point. First calulates the neighbourhood of a point by
        checking which points distance is less than the defined 5% neighbourhood threshold. Then performs linear regression within this neighbourhood.
        Stores the parameters and error of each LR model.'''
        plotModelParameters = False

        w1 = []
        w2 = []
        w = []
        MSE = []

#        Calculate distances between all points so can reuse later.
        self.pointwiseDistance(self.x)
        self.neighbourhods = []

        for i in tqdm(range(self.N)):
            check = [self.xDs_norm[i][j]<0.05 for j in range(self.N)]
            localxdata = list(compress(self.x, check))
            localydata = list(compress(self.y, check))
            self.neighbourhods.append(localxdata)
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


    def compute_distance_matrix(self, w, MSE, distance_weights= [1,0,0], instance=None):
        ''' Computes a distance matrix between all points for a distance function which includes the raw distance values,
        the parameters and error of the Local Linear Regression models for the respective points.
        This will be used as the distance matrix for any clustering algorithms. '''

        wDs, mseDs, model_similarities = [], [], []

        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
        print('Computing Distance Matrix...')
        if instance == None:
            D = np.zeros((self.N,self.N))
            [(wDs.append(euclidean(w[i], w[j])), mseDs.append(abs(MSE[i]-MSE[j]))) for i in tqdm(range(self.N)) for j in range(self.N)]
            normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)

            maxWds, minWds = np.max(wDs), np.min(wDs)
            wDs_norm = np.array(list(map(lambda x: normalise(maxWds, minWds, x), wDs))).reshape(self.N, self.N)

            maxMses, minMses = np.max(mseDs), np.min(mseDs)
            mseDs_norm = np.array(list(map(lambda x: normalise(maxMses, minMses, x), mseDs))).reshape(self.N, self.N)

            for i in range(self.N):
                for j in range(self.N):
                    # print(np.linalg.norm(w[i]-w[j]), MSE[i]-MSE[j], self.x[i]-self.x[j])
#                distance = wDs_norm[i,j] + mseDs_norm[i,j] + 0.25*self.xDs_norm[i,j]
                    distance = distance_weights[0]*self.xDs_norm[i,j] + distance_weights[1]*wDs_norm[i,j] +  distance_weights[2]*mseDs_norm[i,j]
                    D[i,j] = distance
        else:
            D = np.zeros(self.N)
            [(wDs.append(euclidean(w[instance], w[j])), mseDs.append(abs(MSE[instance]-MSE[j]))) for j in range(self.N)]
            normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)

            maxWds, minWds = np.max(wDs), np.min(wDs)
            wDs_norm = np.array(list(map(lambda x: normalise(maxWds, minWds, x), wDs))).reshape(self.N)

            maxMses, minMses = np.max(mseDs), np.min(mseDs)
            mseDs_norm = np.array(list(map(lambda x: normalise(maxMses, minMses, x), mseDs))).reshape(self.N)

            for j in range(self.N):
                # print(np.linalg.norm(w[i]-w[j]), MSE[i]-MSE[j], self.x[i]-self.x[j])
#                distance = wDs_norm[i,j] + mseDs_norm[i,j] + 0.25*self.xDs_norm[i,j]
                distance = distance_weights[0]*self.xDs_norm[instance, j] + distance_weights[1]*wDs_norm[j] +  distance_weights[2]*mseDs_norm[j]
                D[j] = distance

#        with open('saved/distance_matrix.pck', 'wb') as file:
#            pck.dump([D, self.xDs_norm], file)
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
