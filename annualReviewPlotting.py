import matplotlib.pyplot as plt
import numpy as np
from LocalLinearRegression import LocalLinearRegression
from  LinearClustering import LinearClustering
from sklearn.linear_model import LinearRegression
import sys
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

inch = 0.39

#sns.set_theme()


def LR(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_

def generate_data(N=100):
    #Generate noisy sine wave data
    np.random.seed(1)
    x = np.linspace(0, 10, N)
    y = np.sin(x) + np.random.normal(0, 0.7, N)
#    plt.scatter(x,y)
#    plt.show(
    return x, y

def plot_local_models(x,y, w1, w2, neighbourhoods):
    fig2, axes2 = plt.subplots(1,1, figsize=(20*inch, 8*inch))
    for i in range(len(x)):
        fig, axes = plt.subplots(1,1, figsize=(20*inch, 8*inch))
        axes.plot(neighbourhoods[i], [w1[i]*n_x + w2[i] for n_x in neighbourhoods[i]], color='red', linewidth=3)
        axes2.plot(neighbourhoods[i], [w1[i]*n_x + w2[i] for n_x in neighbourhoods[i]], color='red', linewidth=1)
        axes.scatter(x,y, s=10)
        axes.set_xlabel(r'$x$', fontsize=11)
        axes.set_ylabel(r'$\hat{y}$')
        fig.suptitle(r'Local Linear Regression Models', fontsize=11)
        fig.savefig(f'Figures/LocalModels/sine/LocalLinearRegression_{i}.png', bbox_inches='tight')
    axes2.scatter(x,y, s=10)
    axes2.set_xlabel(r'$x$', fontsize=11)
    axes2.set_ylabel(r'$\hat{y}$')
    fig2.suptitle(r'Local Linear Regression Models', fontsize=11)
    fig2.savefig(f'Figures/LocalModels/sine/LocalLinearRegression.png', bbox_inches='tight')


#x,y = generate_data()

def generalFeaturePlot(plot_num=0):

    fig, axes = plt.subplots(1,1, figsize=(9*inch, 4*inch))
#x = np.linspace(0, 30, 50)
#y = [float(3*i+4 + np.random.normal(0, 8, 1)) for i in x]
#axes.scatter(x,y, s=10)
#m, c =  LR(x,y)
#axes.plot(x, [m*i+c for i in x], color='red', linewidth=0.5)
#axes.set_xlabel(r'$x_1$', fontsize=11)
#axes.set_ylabel(r'$\hat{y}$')


    colors = [(1, 0, 0), (1, 1, 1)]  # Red to white
    cmap = LinearSegmentedColormap.from_list("my_cmap", colors)



    all_xs, all_ys = [], []
    fig, axes = plt.subplots(1,1, figsize=(9*inch, 4*inch))
    x = np.linspace(0, 10, 20)
    y = [float(2*i + np.random.normal(0, 4, 1)) for i in x]
    all_xs.append(x)
    all_ys.append(y)
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)

    x = np.linspace(11, 14, 8)
    y = [float(6*i+ np.random.normal(0, 4, 1)-46) for i in x]
    all_xs.append(x)
    all_ys.append(y)
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)

    x = np.linspace(14, 30, 20)
    y = [float(2*i + np.random.normal(0, 4, 1)) for i in x]
    all_xs.append(x)
    all_ys.append(y)
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)

    x1 = np.linspace(15, 22, 10)
    y1 = [float(2*i + np.random.normal(0, 20, 1)) for i in x1]
    all_xs.append(x1)
    all_ys.append(y1)
    axes.scatter(x1,y1, s=10, c='grey', alpha=0.5)

    all_xs = [item for sublist in all_xs for item in sublist]
    all_ys = [item for sublist in all_ys for item in sublist]

    m, c =  LR(all_xs,all_ys)
    axes.plot(all_xs, [m*i+c for i in all_xs], color='red', linewidth=2, linestyle='--')

    m, c =  LR(x1,y1)
    axes.plot(x1, [m*i+c for i in x1], color='magenta', linewidth=2, linestyle='--')
    axes.set_xlabel(r'$x_1$', fontsize=11)
    axes.set_ylabel(r'$\hat{y}$')
    axes.scatter(x1[5],y1[5], s=30, c='orange', edgecolors='black', zorder=5)
## Create a filled circle with the custom colormap
#    circle = Circle((x1[5], y1[5]), 5, color=cmap(0.5))  # Center at (0.5, 0.5), radius 0.3
#    axes.add_patch(circle)

# Set aspect ratio to be equal, so the circle appears as a circle
#    axes.set_aspect('equal')

    axes.grid(False)

#fig.gca().set_facecolor('#ffffff')  # Set the background color
    fig.savefig(f'Figures/GeneralFeature{plot_num}.pdf', bbox_inches='tight')


def varying_feature_plot():
    fig, axes = plt.subplots(1,1, figsize=(9*inch, 4*inch))
    x = np.linspace(0, 10, 20)
    y = [float(2*i + np.random.normal(0, 6, 1)) for i in x]
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)
    m, c =  LR(x,y)
    axes.plot(x, [m*i+c for i in x], color='orange', linewidth=2, linestyle='--')
    axes.scatter(x[10],y[10], s=30, c='orange', edgecolors='black', zorder=5)

    x = np.linspace(10, 20, 20)
    y = [float(8*i+ np.random.normal(0, 6, 1)-60) for i in x]
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)
    m, c =  LR(x,y)
    axes.plot(x, [m*i+c for i in x], color='blue', linewidth=2, linestyle='--')
    axes.scatter(x[10],y[10], s=30, c='blue', edgecolors='black', zorder=5)

#axes.scatter(x[10],y[10], s=30, c='orange', edgecolors='black', zorder=5)


    x = np.linspace(20, 30, 20)
    y = [float(-4*i + np.random.normal(0, 6, 1)+180) for i in x]
    axes.scatter(x,y, s=10, c='grey', alpha=0.5)
    m, c =  LR(x,y)
    axes.plot(x, [m*i+c for i in x], color='green', linewidth=2, linestyle='--')
    axes.scatter(x[10],y[10], s=30, c='green', edgecolors='black', zorder=5)
    axes.set_xlabel(r'$x_2$', fontsize=11)
    axes.set_ylabel(r'$\hat{y}$')

    axes.grid(False)


    fig.savefig('Figures/VaryingFeature.pdf', bbox_inches='tight')



generalFeaturePlot(1)
#
#plot_gradient()

#x, y = generate_data()
#LLR = LocalLinearRegression(x,y, 'Euclidean')
#
#w1, w2, w = LLR.calculateLocalModels()
#plot_local_models(x,y, w1, w2, LLR.neighbourhoods)

#distance_weights = [1,1,0]
#D, xDs= LLR.compute_distance_matrix(w, MSE, distance_weights=distance_weights)
#print('Doing K-medoids-clustering')
#
## Define number of medoids and perform K medoid clustering.
#K = 20
#LC = LinearClustering(x, y, D, xDs, 'x', K)
#
#clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()




