from LocalLinearRegression import LocalLinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os, glob
from LinearClustering import LinearClustering

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





def make_linear_ensemble(xdata, ydata):

    print('Performing Local Linear Regression')
    # Perform LocalLinear regression on fetched data
    LLR = LocalLinearRegression(xdata,ydata, dist_function='Euclidean')
    w1, w2, w, MSE = LLR.calculateLocalModels()
    xrange = np.linspace(min(xdata), max(xdata), 100)
    print('Calculating Distances')
#
    distance_weights = [1,0.75,0]

    D, xDs= LLR.compute_distance_matrix(w, MSE, distance_weights=distance_weights)
    print('Doing K-medoids-clustering')
    # Define number of medoids and perform K medoid clustering.

    LC = LinearClustering(xdata, ydata, D, xDs)

    K =20

    files = glob.glob(f'Figures/Clustering/OptimisedClusters/*')
    for f in files:
        os.remove(f)
#    clustered_data, medoids, linear_params, clustering_cost = LLR.adapted_clustering(K,D, xDs)
    clustered_data, medoids, linear_params, clustering_cost = LC.adapted_clustering(K)


    return clustered_data, linear_params

def main():
    print('Fethcing Data')
#    xdata, ydata, dist_function = fetchData(dataset)
    xdata = [datetime.strptime(i, '%Y-%m-%d') for i in train_data['date']]
    xdata = [(j-xdata[0]).days for j in xdata]

    ydata = train_data['meantemp'].values
#    dist_function = timeDiff
    clustered_data, linear_params = make_linear_ensemble(xdata, ydata)

if __name__ == "__main__":
    main()




