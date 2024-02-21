import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from  sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_decision_regions
import random
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle as pck

import warnings
import copy

# Ignore all warnings
warnings.filterwarnings("ignore")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "font.size": 12,
})
plt.style.use('seaborn')


inch = 0.39


all_features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

def LR(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_

def moons_plot():
    x,y = make_moons(n_samples=1000, noise=0.25, random_state=42)

    df = pd.DataFrame(dict(x=x[:,0], y=x[:,1], label=y))
    df.sort_values(by=['x'], inplace=True)
    colors = {0:'red', 1:'blue'}
    y = df['label']
    df.drop(['label'], axis=1, inplace=True)
    x = df

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    kernel = 'rbf'
    model = SVC(kernel=kernel).fit(X_train, y_train)

    print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)
    print(model.score(X_test, y_test))


    def get_neighbourhood(instance, X_test, y_test,  distance_threshold):

        def distance(point_one, point_two):
            return ((point_one[0] - point_two[0]) ** 2 +
                    (point_one[1] - point_two[1]) ** 2) ** 0.5

        distances = [distance(x1, instance) for x1 in X_test.values]
        neighbourhood = [i for i in range(len(distances)) if distances[i] <= distance_threshold]
        non_local = [i for i in range(len(distances)) if distances[i] > distance_threshold]
        return neighbourhood, non_local
    instance_index = int(len(X_test)/2)
    instance_index = 50
    instances = random.choices(np.arange(len(X_test)), k=10)
#instances = [310]
    instances = range(len(X_test))
    instances = [33,26,27]
    thresholds = [0.5, 1, 1.5]

    colors = ['yellow', 'green', 'orange']
    for instance_index, distance_threshold, color in zip(instances, thresholds, colors):
#    try:

            instance = X_test.iloc[instance_index]
            instance_y = y_test.iloc[instance_index]
            neighbourhood, non_local = get_neighbourhood(instance, X_test, y_test, distance_threshold)
            neighbourhood_ys = [y_pred[i] for i in neighbourhood]
            non_local_ys = [y_pred[i] for i in non_local]

            xs = np.array(X_test.iloc[neighbourhood].values)
            xs = list(zip(X_test.iloc[neighbourhood]['x'], X_test.iloc[neighbourhood]['y']))

# Settings for plotting
#        fig, ax = plt.subplots(figsize=(6, 5))
#        ax.set(xlim=(min(X_test['x']), max(X_test['x'])), ylim=(min(X_test['y']), max(X_test['y'])))
#        ax.scatter(X_test.iloc[neighbourhood]['x'], y=X_test.iloc[neighbourhood]['y'], c=neighbourhood_ys, s=200, edgecolors="k", cmap="bwr")
#        ax.scatter(X_test.iloc[non_local]['x'], y=X_test.iloc[non_local]['y'], c=non_local_ys, s=50, edgecolors="k", cmap="bwr", alpha=0.2)
            if instance_index ==33:
                fig, ax = plt.subplots(figsize=(6, 5))
                common_params = {"estimator": model, "X": X_test, "ax": ax}
                ax.set(xlim=(min(X_test['x']), max(X_test['x'])), ylim=(min(X_test['y']), max(X_test['y'])))
                ax.scatter(X_test['x'], y=X_test['y'], c=y_pred, s=200, edgecolors="k", cmap="bwr", alpha=0.4)
                DecisionBoundaryDisplay.from_estimator(
                            **common_params,

                            response_method="predict",
                            plot_method="pcolormesh",
                            cmap="bwr",
                            alpha=0.3,
                            )
# Plot decision boundary and margins
                DecisionBoundaryDisplay.from_estimator(
                        **common_params,
                        response_method="decision_function",
                        plot_method="contour",
                        #    levels=[-1, 0, 1],
                        levels = [0],
                        colors=["black"],
                        linewidths=[4],
                        #    linestyles=["--", "-", "--"],
                        )
            locality = plt.Circle((instance['x'], instance['y']), radius=distance_threshold, color=color, fill=False, linewidth=6)
            ax.add_patch(locality)
            ax.scatter(instance['x'], instance['y'],s=500, color=color)

## Fit the data to a logistic regression model.
            clf = LogisticRegression()
            X, Y = X_test.iloc[neighbourhood], neighbourhood_ys
            print(Counter(Y))
            sm = SMOTE(k_neighbors=2)
            X, Y = sm.fit_resample(X, Y)
            print(Counter(Y))
            print(min(X['x']), max(X['x']))
            clf.fit(X, Y)

## Retrieve the model parameters.
#b = clf.intercept_[0]
#w1, w2 = clf.coef_.T
## Calculate the intercept and gradient of the decision boundary.
#c = -b/w2
#m = -w1/w2
#xmin, xmax = -1, 2
#ymin, ymax = -1, 2.5
#xd = np.array([xmin, xmax])
#yd = m*xd + c
#plt.plot(xd, yd, 'k', lw=1, ls='--')
#plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
#plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
#
#plt.scatter(X_test.iloc[neighbourhood]['x'], y=X_test.iloc[neighbourhood]['y'], c=neighbourhood_ys, alpha=0.25)
#plt.xlim(min(X['x']), max(X['x']))
#plt.ylim(min(X['y']), max(X['y']))
#plt.ylabel(r'$x_2$')
#plt.xlabel(r'$x_1$')
#plt.show()

###exp = SVC(kernel='linear').fit(X_test.iloc[neighbourhood], y_test.iloc[neighbourhood])
            common_params = {"estimator": clf, "X": X_test.iloc[neighbourhood], "ax": ax}
#DecisionBoundaryDisplay.from_estimator(
#        **common_params,
#
#        response_method="predict",
#        plot_method="pcolormesh",
#        cmap="bwr",
#        alpha=0.3,
#        )
            DecisionBoundaryDisplay.from_estimator(
                    **common_params,
                    response_method="decision_function",
                    plot_method="contour",
                    #    levels=[-1, 0, 1],
                    levels = [0],
                    colors=["white"],
                    linestyles=["--"],
                    linewidths=[5],
                    )





#ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

#ax.set_aspect('equal', 'box')
#plt.axis('scaled')
    fig.savefig(f'Figures/SVC/{instance_index}_explanation.pdf')
#    except:
#        pass

def plot_sample_clustering():

    # ------------ Feature 1 ----------------
    fig, axes = plt.subplots(1,1, figsize=(9*inch, 6*inch))
    x1 = np.linspace(0, 5, 30)
    y1 = [float(2*i + np.random.normal(0, 4, 1)) for i in x1]

    x2 = np.linspace(5, 15, 30)
    y2 = [float(15*i+ np.random.normal(0, 10, 1)-65) for i in x2]

#axes.scatter(x[10],y[10], s=30, c='orange', edgecolors='black', zorder=5)

    x3 = np.linspace(15, 23, 20)
    y3 = [float(-4*i + np.random.normal(0, 6, 1)+220) for i in x3]

    x4 = np.linspace(23, 28, 20)
    y4 = [float(-30*i + np.random.normal(0, 16, 1)+818) for i in x4]

    x5 = np.linspace(28, 35, 30)
    y5 = [float(-8*i + np.random.normal(0, 16, 1)+202) for i in x5]

    x = np.concatenate((x1, x2, x3, x4, x5))
    y = np.concatenate((y1, y2, y3, y4, y5))
    y = MinMaxScaler().fit(y.reshape(-1,1)).transform(y.reshape(-1,1)).reshape(-1)


    axes.scatter(x[:30],y[:30], s=10, c='grey', alpha=0.25, label='Model Predictions')
    m, c =  LR(x[:30],y[:30])
    axes.plot(x[:30], [m*i+c for i in x[:30]], color='orange', linewidth=2, linestyle='--', label='k=1')
    axes.scatter(x[30:60],y[30:60], s=10, c='blue', alpha=0.5, label='Local Points')
    m, c =  LR(x[30:60],y[30:60])
    axes.plot(x[30:60], [m*i+c for i in x[30:60]], color='blue', linewidth=2, linestyle='--', label='k=2')
    axes.scatter(x[40],0.55, s=80, c='black', edgecolors='black', zorder=5, marker='o', label='Instance')
    axes.scatter(x[60:80],y[60:80], s=10, c='grey', alpha=0.25,label='_nolegend_')
    m, c =  LR(x[60:80],y[60:80])
    axes.plot(x[60:80], [m*i+c for i in x[60:80]], color='green', linewidth=2, linestyle='--', label='k=3')
    axes.set_xlabel(r'$x_2$', fontsize=11)
    axes.scatter(x[80:100],y[80:100], s=10, c='grey', alpha=0.25,label='_nolegend_')
    m, c =  LR(x[80:100],y[80:100])
    axes.plot(x[80:100], [m*i+c for i in x[80:100]], color='pink', linewidth=2, linestyle='--', label='k=4')
    axes.set_xlabel(r'$x_2$', fontsize=11)
    axes.scatter(x[100:130],y[100:130], s=10, c='grey', alpha=0.25,label='_nolegend_')
    m, c =  LR(x[100:130],y[100:130])
    axes.plot(x[100:130], [m*i+c for i in x[100:130]], color='purple', linewidth=2, linestyle='--', label='k=5')
    axes.set_xlabel(r'Feature 1', fontsize=11)

#    axes.set_yticks([0,50,100])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,3,5,6,7,0,4,2]
    fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.35))
    fig.savefig('Figures/feature1_sampleClustering.pdf', bbox_inches='tight')

    # ----------- Feature 2 ----------------
    all_xs = []
    all_ys = []

    fig, axes = plt.subplots(1,1, figsize=(9*inch, 6*inch))
    x = np.linspace(0, 5, 30)
    y = [float(-6*i + np.random.normal(0, 4, 1)) for i in x]
    finaly = y[-1]
    [all_xs.append(i) for i in x]
    [all_ys.append(i) for i in y]

    x = np.linspace(5, 15, 30)
    y = [float(8*i+ np.random.normal(0, 10, 1)) for i in x]
    firsty = y[0]
    y = [i + finaly - firsty for i in y]
    finaly = y[-1]
    [all_xs.append(i) for i in x]
    [all_ys.append(i) for i in y]

#axes.scatter(x[10],y[10], s=30, c='orange', edgecolors='black', zorder=5)
    x = np.linspace(15, 23, 20)
    y = [float(-4*i + np.random.normal(0, 6, 1)) for i in x]
    firsty = y[0]
    y = [i + finaly - firsty for i in y]
    finaly = y[-1]
    [all_xs.append(i) for i in x]
    [all_ys.append(i) for i in y]

    x = np.linspace(23, 28, 20)
    y = [float(22*i + np.random.normal(0, 16, 1)) for i in x]
    firsty = y[0]
    y = [i + finaly - firsty for i in y]
    finaly = y[-1]
    [all_xs.append(i) for i in x]
    [all_ys.append(i) for i in y]

    x = all_xs
    y = np.array(all_ys)
    y = MinMaxScaler().fit(y.reshape(-1,1)).transform(y.reshape(-1,1)).reshape(-1)


    axes.scatter(x[:30],y[:30], s=10, c='grey', alpha=0.25, label='Model Predictions')
    m, c =  LR(x[:30],y[:30])
    axes.plot(x[:30], [m*i+c for i in x[:30]], color='orange', linewidth=2, linestyle='--', label = 'k=1')
    axes.scatter(x[30:60],y[30:60], s=10, c='grey', alpha=0.25, label='_nolegend_')
    m, c =  LR(x[30:60],y[30:60])
    axes.plot(x[30:60], [m*i+c for i in x[30:60]], color='blue', linewidth=2, linestyle='--', label = 'k=2')
    axes.scatter(x[90],0.55, s=80, c='black', edgecolors='black', zorder=5, marker='o', label='Instance')
    axes.scatter(x[60:80],y[60:80], s=10, c='grey', alpha=0.25, label='_nolegend_')
    m, c =  LR(x[60:80],y[60:80])
    axes.plot(x[60:80], [m*i+c for i in x[60:80]], color='pink', linewidth=2, linestyle='--', label = 'k=3')
    axes.set_xlabel(r'$x_2$', fontsize=11)
    axes.scatter(x[80:100],y[80:100], s=10, c='green', alpha=0.5, label='Local Points')
    m, c =  LR(x[80:100],y[80:100])
    axes.plot(x[80:100], [m*i+c for i in x[80:100]], color='green', linewidth=2, linestyle='--', label = 'k=4')
    axes.set_xlabel(r'Feature 2', fontsize=11)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,4,6,0,3,5]
    fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.35))
    fig.savefig('Figures/feature2_sampleClustering.pdf', bbox_inches='tight')
#plot_sample_clustering()



def plot_exp_variance(varyingkw=True):
    fig, axes = plt.subplots(1,1, figsize=(10*inch, 20*inch))
    if varyingkw:
        with open('saved/results/PHM08_results_kernel_widths.pck', 'rb') as file:
            results = pck.load(file)
    else:
        with open('saved/results/PHM08_results_samek_kw.pck', 'rb') as file:
            results = pck.load(file)

    model_predictions, llc_results, chilli_results = results
    print(chilli_results.keys())

    colours = ['blue', 'orange', 'green', 'red', 'purple','olive']
    print(len(chilli_results.keys()))
    knum = 1
    width=0.15

    for k,v in chilli_results.items():
        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, exp = v[0]

        exp_list = exp.as_list()
        fontsize = 10

        exp_size = 12
        # Plot the explanation
        explained_features = [i[0] for i in exp_list]

        explained_feature_indices = [all_features.index(i) for i in explained_features]
        explained_feature_perturbations = np.array(perturbations)[:,explained_feature_indices]
        sorted_exp = []
        for f in all_features:
            for e in exp_list:
                if e[0] == f:
                    sorted_exp.append(e)
                    break
        feature_contributions = [i[1] for i in exp_list]

        contributions = [i[1] for i in sorted_exp]
        scaler = StandardScaler()
        contributions = scaler.fit_transform(np.array(contributions).reshape(-1,1)).reshape(-1)

#        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        ys = [1.2*i+width*knum for i in range(len(sorted_exp))]

        print(ys)
        for i in range(len(sorted_exp)):
            if i == 0:
                if varyingkw:
                    axes.barh(ys[i], contributions[i], width, color=colours[knum-1], align='center', label=f'Iteration {k}')
                else:
                    axes.barh(ys[i], contributions[i], width, color=colours[knum-1], align='center', label=f'Iteration {k}')
            else:
                axes.barh(ys[i], contributions[i], width, color=colours[knum-1], align='center', label='_nolegend_')
#        axes.barh(0, sorted_exp[0][1], width, color=colours[knum], align='center', label=f'k={k}')
#        axes.barh([i*width*knum for i in range(len(e))], [e[1] for e in sorted_exp], width, color=colours[knum], align='center', label='_nolegend_')
#        plt.show()
        knum +=1
    axes.tick_params(axis='both', labelsize=14)
    fig.legend(loc='upper center', ncol=3, fontsize=14)
    axes.set_yticks(ys)
    axes.set_yticklabels(all_features, rotation=0, fontsize=fontsize)
#    plt.show()
    if varyingkw:
        fig.savefig('Figures/kw_variance.pdf', bbox_inches='tight')
    else:
        fig.savefig('Figures/same_kw.pdf', bbox_inches='tight')

def plot_exp_fidelity():

#    kws = ['0.01', '0.1', '0.5', '1', '5', '10']
#    kws = ['0.01']
#    results = {kw:[] for kw in kws}
#    for kw in kws:
    with open(f'saved/results/PHM08_results_25instances.pck', 'rb') as file:
        data = pck.load(file)
        model_predictions, llc_predictions, chilli_predictions = data
    model_instance_predictions = [i[0] for i in model_predictions]
    model_neighbours_predictions = [i[1] for i in model_predictions]

    llc_instance_predictions = [i[0] for i in llc_predictions]
    llc_neighbours_predictions = [i[1] for i in llc_predictions]

    chilli_instance_predictions = [i[0] for i in chilli_predictions]
    chilli_neighbours_predictions = [i[1] for i in chilli_predictions]

    for i in range(len(model_instance_predictions)):
        print(i, 'chilli')
        print(mean_squared_error(model_neighbours_predictions[i], llc_neighbours_predictions[i], squared=False))
        print(i, 'llc')
        print(mean_squared_error(model_neighbours_predictions[i], chilli_neighbours_predictions[i], squared=False))


    errors = []
    for kw in kws:
        mse = mean_squared_error(chilli_predictions, model_predictions, squared=False)
        print(mse)
    print(mean_squared_error(model_predictions, llc_predictions, squared=False))

#    print(results)

def plot_deviations():
    with open('saved/results/deviations.pck', 'rb') as file:
        deviations = pck.load(file)

    chilli_deviations, llc_deviations = deviations

    fig, axes = plt.subplots(2,2, figsize=(15,5))
    ax = fig.get_axes()
    for p in range(len(llc_deviations)):
        ax[p].plot([x for x in range(len(llc_deviations[p]))], llc_deviations[p], label=f'LLC {p}')
#    axes.set_yscale('log')
    fig.savefig('Figures/llc_robustness.pdf')

    fig, axes = plt.subplots(2,2, figsize=(15,5))
    ax = fig.get_axes()
    for p in range(len(chilli_deviations)):
        ax[p].plot([x for x in range(len(chilli_deviations[p]))], chilli_deviations[p], label=f'LLC {p}')

#    ax[p].set_yscale('log')
    fig.savefig('Figures/chilli_robustness.pdf')

def bad_exp_plotter():
    xs = [i for i in range(30)]
    ys = [3*i+np.random.normal(0,4,1) for i in xs]
    exp1 = [-2*i+70 for i in xs[10:20]]
    exp2 = [15*i-(15*14-42) for i in xs[12:16]]
    exp3 = [3*i for i in xs[10:20]]

    fig, axes = plt.subplots(1, 1, figsize=(4, 2.5))
    axes.scatter(xs, ys, color='black', s=10)
    axes.scatter(14, 42, color='red', s=40, edgecolors='black', zorder=10)
    axes.plot(xs[10:20], exp1, color='green', linewidth=3, linestyle='dashed')
    axes.plot(xs[12:16], exp2, color='blue', linewidth=3, linestyle='dashed')
    axes.plot(xs[10:20], exp3, color='orange', linewidth=3, linestyle='dashed')
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlabel(r'$x$', fontsize=14)
    axes.set_ylabel(r'$\hat{y}$', fontsize=14)

    fig.savefig('Figures/bad_exp.pdf', bbox_inches='tight')

def remove_outliers(data):
    # Convert the list to a numpy array for convenience
    data_array = np.array(data)

    # Calculate the first and third quartiles (Q1 and Q3)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify and remove outliers
    outliers_removed = [value for value in data_array if lower_bound <= value <= upper_bound]

    return outliers_removed

def exp_sorter(exp_list, features):

    explained_features = [i[0] for i in exp_list]
    for e in explained_features:
        if len(e.split('=')) >1:
            explained_features[explained_features.index(e)] = e.split('=')[0]

    feature_contributions = {f:[] for f in features}
    contributions = [e[1] for e in exp_list]

    explained_feature_indices = [features.index(i) for i in explained_features]
    for f in features:
        for num, e in enumerate(explained_features):
            if e == f:
#                    sorted_exp.append(e[1])
                feature_contributions[f].append(contributions[explained_features.index(e)])
    sorted_exp = [feature_contributions[f][0] for f in features]

    return sorted_exp


def similar_instances(num_matches, colours, extra_one=False, similar=True):
    if similar:
        with open(f'saved/results/similar_instances.pck', 'rb') as file:
            results = pck.load(file)
            results = results
    else:
        with open(f'saved/results/same_instances.pck', 'rb') as file:
            results = pck.load(file)
            results = results

    lime_contributions = {f: [] for f in all_features}
    chilli_contributions = {f: [] for f in all_features}
    llc_contributions = {f: [] for f in all_features}
    max_lime = 0
    max_chilli = 0
    max_llc = 0

    if extra_one == False:
        del results[8]

    for i in range(len(results)):
        lime_exp_list = results[i][0]
        chilli_exp_list = results[i][1]
        llc_feature_contributions = results[i][2]
#        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, exp = chilli_plotting_data



        for exp_list in [lime_exp_list, chilli_exp_list]:

            explained_features = [i[0] for i in exp_list]
            for e in explained_features:
                if len(e.split('=')) >1:
                    explained_features[explained_features.index(e)] = e.split('=')[0]

            if exp_list == lime_exp_list:
                feature_contributions = lime_contributions
                contributions = [e[1] for e in exp_list]
                if max([abs(c) for c in contributions]) > max_lime:
                    max_lime = max([abs(c) for c in contributions])
            else:
                feature_contributions = chilli_contributions
                contributions = [e[1] for e in exp_list]
                if max([abs(c) for c in contributions]) > max_chilli:
                    max_chilli = max([abs(c) for c in contributions])

            explained_feature_indices = [all_features.index(i) for i in explained_features]
            sorted_exp = []
            scaler = StandardScaler()
#        contributions = scaler.fit_transform(np.array(contributions).reshape(-1,1)).reshape(-1)
            for f in all_features:
                for num, e in enumerate(explained_features):
                    if e == f:
#                    sorted_exp.append(e[1])
                        feature_contributions[f].append(contributions[explained_features.index(e)])
                        break
#        feature_contributions = [i[1] for i in sorted_exp]
#        for f in all_features:
#            chilli_contributions[f].append(feature_contributions[all_features.index(f)])

#        data_instance, instance_index, local_x, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data




        feature_contributions = llc_feature_contributions
        if max([abs(c) for c in feature_contributions]) > max_llc:
            max_llc = max([abs(c) for c in feature_contributions])

        scaler = StandardScaler()
#        feature_contributions = scaler.fit_transform(np.array(feature_contributions).reshape(-1,1)).reshape(-1)
        for f in all_features:
            llc_contributions[f].append(feature_contributions[all_features.index(f)])

    lime_variances = []
    chilli_variances = []
    llc_variances = []
    lime_means = []
    chilli_means = []
    llc_means = []


    for f in range(len(all_features)):
        normalised_llc = [i/max_llc for i in llc_contributions[all_features[f]]]
        llc_contributions[all_features[f]] = normalised_llc
        normalised_lime = [i/max_lime for i in lime_contributions[all_features[f]]]
        lime_contributions[all_features[f]] = normalised_lime
        normalised_chilli = [i/max_chilli for i in chilli_contributions[all_features[f]]]
        chilli_contributions[all_features[f]] = normalised_chilli

#        cv = lambda x: np.std(x) / np.mean(x)
        cv = lambda x: np.std(x)
        llc_variances.append(cv(llc_contributions[all_features[f]]))
        lime_variances.append(cv(lime_contributions[all_features[f]]))
        chilli_variances.append(cv(chilli_contributions[all_features[f]]))
        llc_means.append(np.mean(llc_contributions[all_features[f]]))
        lime_means.append(np.mean(lime_contributions[all_features[f]]))
        chilli_means.append(np.mean(chilli_contributions[all_features[f]]))

#    print(lime_variances)
#    print(chilli_variances)
#    print(llc_variances)
#    print(np.average(lime_variances, weights=lime_means), np.average(chilli_variances, weights=chilli_means), np.average(llc_variances, weights=llc_means))
    print(np.mean(lime_variances), np.mean(chilli_variances), np.mean(llc_variances))

    if similar:
        sp = 6
    else:
        sp = 5
    fig, axes = plt.subplots(1, sp, figsize=(14, 5))
    for p, method in enumerate([lime_contributions, chilli_contributions, llc_contributions]):

        knum =0
        width=0.1
        tick_pos = []
        max_contribution = 0
        plot = p*2
        for n, f in enumerate(all_features):
            contributions = method[f]

#            scaler = StandardScaler()
#            contributions = scaler.fit_transform(np.array(contributions).reshape(-1,1)).reshape(-1)

#        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

            for i in range(len(contributions)):

#                if i != 1:
                    y = 1.5*n+i*width
                    if i ==5:
                        tick_pos.append(y)
                    contribution = contributions[i]
                    if abs(contribution) > max_contribution:
                        max_contribution = abs(contribution)
                    if i == 0:
                        axes[plot].barh(y, contribution, width, color=[colours[n]], align='center', label=f'Iteration {i}')
                    else:
                        axes[plot].barh(y, contribution, width, color=[colours[n]], align='center', label=f'Iteration {i}')

        knum +=1
    for p, vars in enumerate([lime_variances, chilli_variances, llc_variances]):
        plot = p*2+1
        if plot<=sp-1:
            axes[plot].barh([i for i in range(len(vars))], vars, width, color=colours, align='center', label=f'Iteration {i}')
            axes[plot].set_title('LLC')
#        axes.barh(0, sorted_exp[0][1], width, color=colours[knum], align='center', label=f'k={k}')
#        axes.barh([i*width*knum for i in range(len(e))], [e[1] for e in sorted_exp], width, color=colours[knum], align='center', label='_nolegend_')
#        plt.show()
#        for y in [x for x in range(len(all_features))]:
#            axes[plot].hlines(y+1, -20, 20, color='#ede9e8', linestyle='dashed', label='_nolegend_')

#        BPdata = [remove_outliers(method[f]) for f in all_features]
#        BPdata = [method[f] for f in all_features]

#        axes[plot].boxplot(BPdata, vert=False, showfliers=False, labels=all_features, patch_artist=True)
#        axes[plot].set_xlim(-max_contribution-0.1*max_contribution, max_contribution+0.1*max_contribution)

    for a, ax in enumerate(fig.get_axes()):
        if a in [0,2,4]:
            ax.set_xlim(-1.1,1.1)
            ax.set_xlabel('Contribution')
        else:
            max_cov = max([abs(i) for i in np.array(lime_variances+chilli_variances+llc_variances)])
            ax.set_xlim(-1.1*max_cov, 1.1*max_cov)
            ax.set_xlabel('Standard Deviation')
        if a != 0:
            ax.set_yticklabels([])




    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels(all_features)
    axes[0].set_title('LIME')
    axes[1].set_title('LIME')
    axes[2].set_title('CHILLI')
    axes[3].set_title('CHILLI')
    axes[4].set_title('LLC')
    if similar:
        fig.savefig(f'Figures/similar_instances_extra_{extra_one}.pdf', bbox_inches='tight')
    else:
        fig.savefig('Figures/same_instances.pdf', bbox_inches='tight')




#bad_exp_plotter()
#plot_deviations()
#colours = [np.random.rand(3) for c in range(len(all_features))]
#
#similar_instances(24,colours,True, similar=False)
#for ex in [True, False]:
#    similar_instances(24,colours,ex, similar=True)

def normalise(exp):
    max_val = max(abs(max(exp)), abs(min(exp)))
    norm = lambda x: x/max_val
    normalised = list(map(norm, exp))
    if len(np.unique(normalised)) ==  1:
        print(exp)
    return normalised

def exp_progression(dataset='MIDAS', kernel_width=0.1, mode='same'):

    if dataset == 'MIDAS':
        features = ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date']
    elif dataset == 'PHM08':
        features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    with open(f'saved/results/{dataset}_{mode}_instances_kw={kernel_width}.pck', 'rb') as f:
        results = pck.load(f)


    fig = plt.figure(figsize=(3, 14))
    spec = gridspec.GridSpec(ncols=1, nrows=6, figure=fig, height_ratios=[1, 10, 1, 10, 1, 10])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    ax4 = fig.add_subplot(spec[4])
    ax5 = fig.add_subplot(spec[5])


#    maxb = max(predictions)
    minb = min([min(results[i][0]) for i in range(3)])
    minb=0
    maxb = max([max(results[i][0]) for i in range(3)])
#    minb = min(results[4])
#    maxb = max(results[4])
    norm = matplotlib.colors.Normalize(minb,maxb)
    colors = [[norm(minb), "white"],
              [norm(maxb), "black"]]

    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    axes = [[ax0, ax1], [ax2, ax3], [ax4, ax5]]
    methods = ['LIME', 'CHILLI', 'LLC']

    for i in range(1,3):
        method_preds = results[i][0]
        predictions = copy.deepcopy(results[4])
        explanation_models = results[i][1]

        if i == 0 or i == 1:
            explanation_models = [exp_sorter(e, features = features) for e in explanation_models]
        print(methods[i])
        explanation_models = list(map(normalise, explanation_models))
        print(mean_squared_error(predictions, method_preds, squared=False))


        outliers = []
        instances = list(results[3])
        [predictions.pop(instances.index(o)) for o in outliers]
        [explanation_models.pop(instances.index(o)) for o in outliers]
        [method_preds.pop(instances.index(o)) for o in outliers]
        [instances.pop(instances.index(o)) for o in outliers]

        max_val = np.array(explanation_models).max()
        min_val = np.array(explanation_models).min()
        boundary = max(abs(max_val), abs(min_val))
#        boundary = 1

        norm = matplotlib.colors.Normalize(-boundary,boundary)
        colors = [[norm(-boundary), "red"],
                    [norm(-0.2*boundary), 'pink'],
                    [norm(0), 'lightgrey'],
                    [norm(0.2*boundary), 'lightgreen'],
                    [norm(boundary), "green"]]

        cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)





#    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

        sns.heatmap(ax=axes[i][1], data=np.transpose(explanation_models), cmap=cmap1, vmin = -boundary, vmax=boundary)
        predictions = [[p] for p in predictions]
        sns.heatmap(ax=axes[i][0], data=np.transpose(np.column_stack([predictions, method_preds])), cmap=cmap2, vmin=minb, vmax=maxb, annot=False, fmt='.2f')
        axes[i][0].set_xticklabels([])
        axes[i][0].set_title(methods[i])
        axes[i][1].set_yticks(np.arange(len(features))+0.5)
        axes[i][1].set_yticklabels(features, rotation=0)
#        axes[i][0].set_yticklabels(['Model Predictions', 'Explanation Predictions'], rotation=0)
        if i !=2:
            axes[i][1].set_xticklabels([])
        else:
            if mode == 'same':
                axes[i][1].set_xticklabels(range(10), rotation=0)
            elif mode == 'similar':
                axes[i][1].set_xticklabels(instances, rotation=0)



#    ax.set_xticklabels([np.round(x,2) for x in predictions])
    plt.savefig(f'Figures/{dataset}_{mode}_instances_progression_kw={kernel_width}.pdf', bbox_inches='tight')


#for kw in [0.01, 0.1, 0.25, 0.5]:
#    exp_progression('PHM08', kw)
exp_progression('PHM08', 0.1, 'similar')
