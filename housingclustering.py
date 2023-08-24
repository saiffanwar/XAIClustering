import cProfile, pstats
from LocalLinearRegression import LocalLinearRegression
import pickle as pck
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly
import plotly.graph_objs as go
from cyclicRegression import CyclicRegression
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from mpl_toolkits.mplot3d.axes3d import get_test_data


# Performing linear clustering of housing dataset to segment all linear regions of the data.

train_data = pd.read_csv('Data/DelhiClimate/DailyDelhiClimateTrain.csv').sort_values(by=['date'])
def plotData(train_data, x_col, y_col):
    fig, axes = plt.subplots(1,1,figsize=(10,10))
    axes = fig.axes
#    print(train_data['YearBuilt'].values)
    axes[0].scatter(train_data[x_col], train_data[y_col], s=3)
    axes[0].set_xticks(train_data[x_col][0::100])
    axes[0].set_xticklabels(train_data[x_col][0::100], rotation=45)

#    plt.show()

plotData(train_data, x_col='date', y_col='meantemp')

def euclideanDefine(xdata):
    maxVal = max(xdata)
    euclidean = lambda x1,x2: abs(x1-x2)/maxVal
    return euclidean

def timeDiff(x1,x2):
    return abs(x1-x2).days




def LinearClustering(xdata, ydata, dist_function):

    print('Performing Local Linear Regression')
    # Perform LocalLinear regression on fetched data
    LLR = LocalLinearRegression(xdata,ydata)
    w1, w2, w, MSE = LLR.calculateLocalModels(dist_function)
    xrange = np.linspace(min(xdata), max(xdata), 100)
    print('Calculating Distances')
    # Use local models to compute distance matrix for all points (slow) distance weights indicates how much of each component is contributing to the distance measure.
#    results = []
#    for weight in np.linspace(0,1,5):
#        for component in range(3):
#            distance_weights = [1,1,1]
#            distance_weights[component] = weight
#
    distance_weights = [1,0.75,0]
    D, xDs= LLR.computeDistanceMatrix(w1, w2, w, MSE, dist_function, distance_weights=distance_weights)
    print('Doing K-medoids-clustering')
    # Define number of medoids and perform K medoid clustering.
    K = 10

    costs = []
    k_range = np.arange(17,18,1)
    print(k_range)
    for K in k_range:
        clustered_data, medoids, linear_params, clustering_cost = LLR.adapted_clustering(K,D, xDs)
        print('K: ', K, 'cost: ', clustering_cost)
        costs.append(clustering_cost)

        # Plot clustered data
        fig = LLR.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
        fig.savefig(f'Figures/Clustering/DistanceWeights/{K}medoidsClustering{distance_weights}.pdf')
#    results.append([distance_weights, clustering_cost])
#
#    print(results)
#    Koptimisationfig,axes = plt.subplots(1,1,figsize=(10,10))
#    axes.plot(k_range, costs)
#    axes.set_xlabel('K')
#    Koptimisationfig.savefig('Koptimisation.pdf')

        #    clustered_data = LLR.KMedoidClustering(K, D)
#    print('avg intersection cost: ', avg_intersection_cost)
#   Plot medoids and linear parameters.

    return clustered_data, linear_params

#def create_3d_heatmap(x, y, z, c):
#    # Create a grid of points for the parameters
#    x_grid, y_grid, z_grid, c_grid = np.meshgrid(x, y, z, c)
#
#    # Compute a value for each grid point (you can replace this with your own function)
#    # For demonstration purposes, let's use a simple random value generator
#    values = np.random.rand(*x_grid.shape)
#
#    # Create a 3D plot
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#
#    sizes = np.meshgrid([(1-i)*50 for i in c])
#    # Plot the 3D heatmap
#    ax.plot_surface(x_grid, y_grid, z_grid, c=values, s=100, cmap='viridis')
#
#    # Set axis labels
#    ax.set_xlabel('Parameter 1')
#    ax.set_ylabel('Parameter 2')
#    ax.set_zlabel('Parameter 3')
#
#    # Add color bar
#    cbar = plt.colorbar(ax.scatter(x_grid, y_grid, z_grid, c=values, cmap='viridis'))
#    cbar.set_label('Parameter 4')
#
#    # Show the plot
#    plt.show()

#def plotResults():
#    results = pck.load(open('results.pck', 'rb'))
#    pprint(results)
#    x = [i[0][0] for i in results]
#    y = [i[0][1] for i in results]
#    c = [i[0][2] for i in results]
#    z = [i[1] for i in results]
#
#    create_3d_heatmap(x,y,z,c)

def main():
    print('Fethcing Data')
#    xdata, ydata, dist_function = fetchData(dataset)
    xdata = [datetime.strptime(i, '%Y-%m-%d') for i in train_data['date']]
    xdata = [(j-xdata[0]).days for j in xdata]

    ydata = train_data['meantemp'].values
    dist_function = euclideanDefine(xdata)
#    dist_function = timeDiff
    clustered_data, linear_params = LinearClustering(xdata, ydata, dist_function)

if __name__ == "__main__":
    main()




