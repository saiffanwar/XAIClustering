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


class LocalLinearRegression():

    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression()

    def calculateLocalModels(self, distanceFunction):
        plotModelParameters = False
        w1 = []
        w2 = []
        w = []
        MSE = []

        for i in tqdm(range(self.N)):
            xx = self.x[i]
            check = [distanceFunction(self.x[j],xx)<0.25 for j in range(self.N)]
            localxdata = list(compress(self.x, check))
            localydata = list(compress(self.y, check))
#            X = np.array([localxdata, np.ones(len(localxdata))]).T
#            wlocal = np.linalg.lstsq(X, localydata, rcond=1)[0]
            preds, m, c = self.CR.cyclicRegression(localxdata, localydata)

            w1.append(m)
            w2.append(c)
            w.append([m, c])
            MSE.append(mean_squared_error(localydata, preds))

        if plotModelParameters:
            x = [x[i] for i in range(len(x))]
            fig, axes = plt.subplots(1,3,figsize=(20,7))

            fontsize=14

            axes[0].scatter(x, MSE, s=5)
            axes[0].set_xlabel('x', fontsize=fontsize)
            axes[0].set_ylabel('MSE', fontsize=fontsize)

            axes[1].scatter(x, w1, s=5)
            axes[1].set_xlabel('x', fontsize=fontsize)
            axes[1].set_ylabel('w1', fontsize=fontsize)

            axes[2].scatter(x, w2, s=5)
            axes[2].set_xlabel('x', fontsize=fontsize)
            axes[2].set_ylabel('w2', fontsize=fontsize)

        return w1, w2, w, MSE


    def computeDistanceMatrix(self, w1, w2, w, MSE, distanceFunction):

        D = np.zeros((self.N,self.N))

        xDs, wDs, mseDs = [], [], []

        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
        print('Computing Distance Matrix...')
        [(xDs.append(distanceFunction(self.x[i], self.x[j])), wDs.append(euclidean(w[i], w[j])), mseDs.append(MSE[i]-MSE[j])) for i in tqdm(range(self.N)) for j in range(self.N)]

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

        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(xDs), np.min(xDs)
        xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), xDs))).reshape(self.N, self.N)

        maxWds, minWds = np.max(wDs), np.min(wDs)
        wDs_norm = np.array(list(map(lambda x: normalise(maxWds, minWds, x), wDs))).reshape(self.N, self.N)

        maxMses, minMses = np.max(mseDs), np.min(mseDs)
        mseDs_norm = np.array(list(map(lambda x: normalise(maxMses, minMses, x), mseDs))).reshape(self.N, self.N)

        for i in range(self.N):
            for j in range(self.N):
                # print(np.linalg.norm(w[i]-w[j]), MSE[i]-MSE[j], self.x[i]-self.x[j])
                distance = wDs_norm[i,j] + mseDs_norm[i,j] + xDs_norm[i,j]
                D[i,j] = distance

        return D, xDs_norm


    def KMedoidClustering(self, K, D):
        km = kmedoids.KMedoids(n_clusters=K, init='random', random_state=0, method='pam')
        # c = kmedoids.fasterpam(D,K)
        c=km.fit(D)
        clusteredData = []
        for k in np.unique(c.labels_):
            clustersx = []
            clustersy = []
            for i in range(len(c.labels_)):
                if c.labels_[i] == k:
                    clustersx.append(self.x[i])
                    clustersy.append(self.y[i])

            clusteredData.append([clustersx, clustersy])
        return clusteredData

    def LinearModelsToClusters(self, clusteredData):
        plotFinalLinearModels = False

        def LR(x, y):
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_

        if plotFinalLinearModels:
            fig, axes = plt.subplots(1,4,figsize=(20,5))
        linearParams = []
        clusterPreds = []
        for i in range(len(clusteredData)):
            preds, w, b = self.CR.cyclicRegression(clusteredData[i][0], clusteredData[i][1])
            linearParams.append([w,b])
            clusterPreds.append(preds)
            if plotFinalLinearModels:
                axes[i].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data')
                axes[i].plot(np.array(clusteredData[i][0]).flatten(), preds)
        return linearParams, clusterPreds

