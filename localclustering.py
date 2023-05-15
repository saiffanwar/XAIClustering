import cProfile, pstats
from LocalLinearRegression import LocalLinearRegression
import pickle as pck
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from cyclicRegression import CyclicRegression

#   PROFILER BLOCK
#    profiler = cProfile.Profile()
#    profiler.enable()
#    profiler.disable()
#    stats =  pstats.Stats(profiler).sort_stats('cumtime')
#    stats.print_stats()


NumSamples = 2000
dataset = 'webtris'
def fetchData(dataset='webtris'):


    def cyclic(x1, x2, possValues=np.arange(0,96,1)):
        diff = abs(x1-x2)
        return min(len(possValues) - diff, diff)
    def euclideanDefine(xdata):
        maxVal = max(xdata)
        euclidean = lambda x1,x2: abs(x1-x2)/maxVal
        return euclidean

    # WebTRIS DATA:
    if dataset == 'webtris':
        with open('data/WebtrisData.pck', 'rb') as file:
            data = pck.load(file)[0]
        xdata, ydata = data['Time Interval'].values[:NumSamples], data['Total Volume'].values[:NumSamples]
        distFunction = cyclic

    # MIDAS Data
    elif dataset == 'midas':
        with open('dataallMIDASdata.pck', 'rb') as file:
             data = pck.load(file)
        xdata,ydata = data[0], data[1]
        featureNum = 8
        xdata, ydata = xdata[:,featureNum], ydata
        xdata,ydata = zip(*random.sample(list(zip(xdata, ydata)), NumSamples))

        distFunction = euclideanDefine(xdata)

    return xdata, ydata, distFunction

def plotMedoids(clusteredData, linearParams, preds):
    print('Starting Plotting')
    CR = CyclicRegression()
    plotly_Fig = None

    fig, axes = plt.subplots(1,1,figsize=(10,10))
    axes = fig.axes
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'brown', 'grey', 'cyan', 'magenta']
    for i in range(len(clusteredData)):
        w,b = linearParams[i]
        colour = np.random.rand(1,3)
        colour = colours[i]
        axes[0].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data', c=colour)
        axes[0].scatter(np.array(clusteredData[i][0]).flatten(), preds[i], c=colour, linewidth=5)

        plotly_Fig = CR.plotCircularData(clusteredData[i][0], clusteredData[i][1], preds[i], plotly_Fig, colour)
    fig.savefig(dataset+'data'+str(len(clusteredData))+'Medoids.pdf')
    plotly_Fig.show()

def main():
    print('Fethcing Data')
    xdata, ydata, distFunction = fetchData(dataset)

    print('Performing Local Linear Regression')
    # Perform LocalLinear regression on fetched data
    LLR = LocalLinearRegression(xdata,ydata)
    w1, w2, w, MSE = LLR.calculateLocalModels(distFunction)

    print('Calculating Distances')
    # Use local models to compute distance matrix for all points (slow)
    D, xDs= LLR.computeDistanceMatrix(w1, w2, w, MSE, distFunction)
    print('Doing K-medoids-clustering')
    # Define number of medoids and perform K medoid clustering.
    K = 7
    clusteredData = LLR.KMedoidClustering(K, D)

    linearParams, preds = LLR.LinearModelsToClusters(clusteredData)

#   Plot medoids and linear parameters.
    plotMedoids(clusteredData, linearParams, preds)

if __name__ == "__main__":
    main()
