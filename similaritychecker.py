import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def calculate_line_similarity(x, y, cluster_num_1, cluster_num_2, cluster1, cluster2, linear_params1, linear_params2, plotting=False, feature_name='X'):
#    print(cluster1)
    x1 = np.linspace(min(cluster1), max(cluster1), 5)
    x2 = np.linspace(min(cluster2), max(cluster2), 5)
    m1, c1 = linear_params1
    m2, c2 = linear_params2

#    if np.sign(m1) != np.sign(m2):
#        m2 = -m2
#        c2 = -c2

    def line1(x):
        return m1*x+c1

    def line2(x):
        return m2*x+c2

    y1 = [line1(i) for i in x1]
    y2 = [line2(i) for i in x2]

    line2_interp_x1 = [line2(i) for i in x1]
    line1_interp_x2 = [line1(i) for i in x2]

#    error = abs(np.sum(np.array(line2_interp_x1) - np.array(y1))) + abs(np.sum(np.array(line1_interp_x2) - np.array(y2)))
    error = (mean_squared_error(y1, line2_interp_x1, squared=False) + mean_squared_error(y2, line1_interp_x2, squared=False))/2

#    cos_sim = cosine_similarity(np.array([x1, y1]), np.array([x2, y2]))
    if plotting:

        y1 = np.array(y1).flatten()
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes.plot(x1, y1, color='blue', alpha=0.5)
        axes.plot(x2, y2, color='green', alpha=0.5)
#        axes.scatter(x1, y1, color='blue', alpha=0.5)
#        axes.scatter(x2, y2, color='green', alpha=0.5)

        axes.plot(x1, line2_interp_x1, color='green', alpha=0.5)
        axes.plot(x2, line1_interp_x2, color='blue', alpha=0.5)
#        axes.scatter(x1, line2_interp_x1, color='green', alpha=0.5)
#        axes.scatter(x2, line1_interp_x2, color='blue', alpha=0.5)
        axes.set_xlim(min(x), max(x))
        axes.set_ylim(min(y), max(y))
        axes.title.set_text('Error: ' + str(error))
        fig.savefig(f'Figures/Clustering/OptimisedClusters/{feature_name}/{cluster_num_1}_{cluster_num_2}_similarity.pdf')

    return error, [[x1, y1], [x2, y2], [line2_interp_x1, line1_interp_x2]]



