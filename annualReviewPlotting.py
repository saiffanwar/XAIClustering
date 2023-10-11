import matplotlib.pyplot as plt
import numpy as np
from LocalLinearRegression import LocalLinearRegression
from  LinearClustering import LinearClustering

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

inch = 0.39

def generate_data():
    #Generate noisy sine wave data
    np.random.seed(1)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)
#    plt.scatter(x,y)
#    plt.show()
    return x, y

def plot_local_models(x,y, w1, w2, neighbourhoods):
    fig, axes = plt.subplots(1,1, figsize=(10*inch, 4*inch))
    for i in range(len(x)):
        axes.plot(neighbourhoods[i], [w1[i]*n_x + w2[i] for n_x in neighbourhoods[i]], color='red', linewidth=0.5)
    axes.scatter(x,y, s=3)
    axes.set_xlabel(r'$x$', fontsize=11)
    axes.set_ylabel(r'$\hat{y}$')
    fig.suptitle(r'Local Linear Regression Models', fontsize=11)
    fig.savefig('Figures/LocalLinearRegression.pdf', bbox_inches='tight')



x,y = generate_data()

LLR = LocalLinearRegression(x,y, 'Euclidean')

w1, w2, w, MSE = LLR.calculateLocalModels()

plot_local_models(x,y, w1, w2, LLR.neighbourhods)

distance_weights = [1,1,0]
D, xDs= LLR.compute_distance_matrix(w, MSE, distance_weights=distance_weights)
print('Doing K-medoids-clustering')

# Define number of medoids and perform K medoid clustering.
K = 5
LC = LinearClustering(x, y, D, xDs, 'x', K)

clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()

#fig.savefig(f'Figures/Clustering/OptimisedClusters/{features[i]}_final_{len(clustered_data)}.pdf')
#
#
#cluster_x_ranges = [[min(clustered_data[i][0]), max(clustered_data[i][0])] for i in range(len(clustered_data))]
#feature_ensembles[features[i]] = [clustered_data, linear_params, cluster_x_ranges]
#
#with open(f'saved/feature_ensembles_K{K}_{distance_weights[0]}_{distance_weights[1]}.pck', 'wb') as file:
#    pck.dump(feature_ensembles, file)
#



