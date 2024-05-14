import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import plotly.express as px
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
from pprint import pprint

from texttable import Texttable
import latextable

import warnings
import copy
import sys

# Ignore all warnings
warnings.filterwarnings("ignore")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "font.size": 12,
})
#plt.style.use('seaborn-v0_8-bright')
#sns.set_theme()

inch = 0.39


all_features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

cleaned_midas = ['Heathrow wind speed', 'Heathrow wind direction', 'Heathrow total cloud cover', 'Heathrow cloud base height', 'Heathrow visibility', 'Heathrow MSL pressure', 'Heathrow relative humidity', 'Heathrow rainfall', 'Date']

def LR(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_

def good_locality_plot(fig, class_colors, colors):
    x,y = make_moons(n_samples=1000, noise=0.25, random_state=42)

    df = pd.DataFrame(dict(x=x[:,0], y=x[:,1], label=y))
    df.sort_values(by=['x'], inplace=True)
    y = df['label']
    df.drop(['label'], axis=1, inplace=True)
    x = df

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    kernel = 'rbf'
    model = SVC(kernel=kernel).fit(X_train, y_train)


    y_pred = model.predict(X_test)
    ids = np.argsort(X_test['x'].values)
    vals = np.sort(X_test['x'].values)


    def get_neighbourhood(instance, X_test, y_test,  distance_threshold):

        def distance(point_one, point_two):
            return ((point_one[0] - point_two[0]) ** 2 +
                    (point_one[1] - point_two[1]) ** 2) ** 0.5

        distances = [distance(x1, instance) for x1 in X_test.values]
        neighbourhood = [i for i in range(len(distances)) if distances[i] <= distance_threshold]
        non_local = [i for i in range(len(distances)) if distances[i] > distance_threshold]
        return neighbourhood, non_local

    instances = [381, 183, 472]
#    thresholds = [0.7 for i in instances]
    thresholds = [0.5,0.6, 0.9]
#    xminmax = [[0,1.2],[-0.7,-0], [1.2,1.4], [-2,2],[-2,2]]
    xminmax = [[-0.62,-0.5],[0.1,1.3],[1.5,2.8]]
#    xminmax = [[-2,2] for i in instances]


#    fig, ax = plt.subplots(1,1, figsize=(14, 2.8), sharex=True, sharey=True)
    ax = fig.get_axes()[3]
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [class_colors[0], class_colors[1]])

    common_params = {"estimator": model, "X": X_test, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
            **common_params,

            response_method="predict",
            plot_method="pcolormesh",
            cmap=cmap1,
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
            label = 'SVC Boundary'
            #    linestyles=["--", "-", "--"],
            )
    for instance_index, distance_threshold, color, xmm in zip(instances, thresholds, colors, xminmax):
        #    try:
            instance = X_test.iloc[instance_index]
            instance_y = y_test.iloc[instance_index]
            neighbourhood, non_local = get_neighbourhood(instance, X_test, y_test, distance_threshold)
            neighbourhood_preds = [y_pred[i] for i in neighbourhood]
            non_local_ys = [y_pred[i] for i in non_local]

            xs = np.array(X_test.iloc[neighbourhood].values)
            xs = list(zip(X_test.iloc[neighbourhood]['x'], X_test.iloc[neighbourhood]['y']))

# Settings for plotting
#            fig, ax = plt.subplots(figsize=(3, 2))
#            ax.set(xlim=(min(X_test['x']), max(X_test['x'])), ylim=(min(X_test['y']), max(X_test['y'])))
#            ax.scatter(X_test.iloc[neighbourhood]['x'], y=X_test.iloc[neighbourhood]['y'], c=neighbourhood_preds, s=200, edgecolors="k", cmap="bwr", label='Local points')
#            ax.scatter(X_test.iloc[non_local]['x'], y=X_test.iloc[non_local]['y'], c=non_local_ys, s=50, edgecolors="k", cmap="bwr", alpha=0.2)
#            ax.set_xlim(-2, 3)
#            ax.set_ylim(-1.2,1.7)


            all_xs = copy.deepcopy(X_test['x']).values
            all_ys = copy.deepcopy(X_test['y']).values
            class_0 = np.argwhere(y_pred == 0).flatten()
            class_1 = np.argwhere(y_pred == 1).flatten()

            neighbourhood_xs = X_test.iloc[neighbourhood]['x'].values
            neighbourhood_ys = X_test.iloc[neighbourhood]['y'].values
            neighbourhood_class_0 = np.argwhere(np.array(neighbourhood_preds) == 0).flatten()
            neighbourhood_class_1 = np.argwhere(np.array(neighbourhood_preds) == 1).flatten()


            nl_0 = ax.scatter(all_xs[class_0], y=all_ys[class_0], c=class_colors[0], s=200,  alpha=0.1, label='Non-Local points')
            nl_1 = ax.scatter(all_xs[class_1], y=all_ys[class_1], c=class_colors[1], s=200, alpha=0.1, label='Non-Local points')
#            fig1 = px.scatter(x=all_xs, y=all_ys, color=y_pred)
#            fig1.show()
            l_0 = ax.scatter(neighbourhood_xs[neighbourhood_class_0], y=neighbourhood_ys[neighbourhood_class_0], c=class_colors[0], s=200, edgecolors="k", label='Local points', zorder=5)
            l_1 = ax.scatter(neighbourhood_xs[neighbourhood_class_1], y=neighbourhood_ys[neighbourhood_class_1], c=class_colors[1], s=200, edgecolors="k", label='Local points', zorder=5)
            locality = plt.Circle((instance['x'], instance['y']), radius=distance_threshold, color=color, fill=False, linewidth=3, zorder = 7)
            ax.add_patch(locality)
            t_x =ax.scatter(instance['x'], instance['y'],s=200, color=color, zorder=7, marker='o', label='Target Instance')

## Fit the data to a logistic regression model.
            clf = LogisticRegression()
            X, Y = X_test.iloc[neighbourhood], neighbourhood_preds
            print(Counter(Y))
            sm = SMOTE(k_neighbors=2)
            X, Y = sm.fit_resample(X, Y)
            print(Counter(Y))
            print(min(X['x']), max(X['x']))
            clf.fit(X, Y)
# Retrieve the model parameters.
            b = clf.intercept_[0]
            w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
            c = -b/w2
            m = -w1/w2
            xd = np.array(xmm)
            yd = m*xd + c
            exp,= ax.plot(xd, yd, color=color, lw=3, ls='dashed', label='Explanation Surrogate Model', zorder=7)
            exp_preds = clf.predict(X_test.iloc[neighbourhood])

            exp_class_0 = np.argwhere(np.array(exp_preds) == 0).flatten()
            exp_class_1 = np.argwhere(np.array(exp_preds) == 1).flatten()
#            plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
#            plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

            instance_pred= clf.predict(instance.values.reshape(1,-1))
            ax.scatter(instance['x'], y=instance['y'], c=instance_pred, cmap='Greys', edgecolors=class_colors[int(instance_pred)],  s=30, label='Explanation Prediction', zorder=8)
            exp_0 = ax.scatter(neighbourhood_xs[exp_class_0], y=neighbourhood_ys[exp_class_0], c='k', edgecolors=class_colors[0], s=30, label='Explanation Prediction',zorder=6)
            exp_1 = ax.scatter(neighbourhood_xs[exp_class_1], y=neighbourhood_ys[exp_class_1], c='w', edgecolors=class_colors[1], s=30, label='Explanation Prediction', zorder=6)
            ax.set_ylabel(r'$x_2$')
            ax.set_xlabel(r'$x_1$')



###exp = SVC(kernel='linear').fit(X_test.iloc[neighbourhood], y_test.iloc[neighbourhood])
            common_params = {"estimator": clf, "X": X_test.iloc[neighbourhood], "ax": ax}
#            DecisionBoundaryDisplay.from_estimator(
        #                    **common_params,
        #
        #                    response_method="predict",
        #                    plot_method="pcolormesh",
        #                    cmap="bwr",
        #                    alpha=0.3,
        #                    )
#            DecisionBoundaryDisplay.from_estimator(
        #                    **common_params,
        #                    response_method="decision_function",
        #                    plot_method="contour",
        #                    #    levels=[-1, 0, 1],
        #                    levels = [0],
        #                    colors=["white"],
        #                    linestyles=["--"],
        #                    linewidths=[5],
        #                    )
#ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
#            plt.axis('scaled')
            ax.set_xlim(-2,3)
            ax.set_ylim(-1.2, 1.7)
            ax.set_aspect('equal', 'box')
            ax.grid(False)
#            handles, labels = plt.gca().get_legend_handles_labels()
##    plt.axis('scaled')
#    svc = mlines.Line2D([],[],color='black')
#    locality_patch = mlines.Line2D([],[],color=colors[0])
#
##            order = [1,2,4,6,0,3,5]
#    legend = fig.legend(handles=[(nl_0, nl_1), (l_0, l_1), (svc), t_x, locality_patch, ( exp ), (exp_0, exp_1)], labels = ['Non-Local Points', 'Local Points', 'SVC Decsision Boundary','Target Instance', 'Locality Around Instance', 'Explanation Surrogate Model', 'Explanation Predictions'],
                         #       handler_map={tuple: HandlerTuple(ndivide=None, pad=1)}, loc='upper center', ncol=4, fontsize=11, bbox_to_anchor=(0.5,1.2), labelspacing=1)
#    print(legend.legendHandles[5])
#    legend.legendHandles[3]._sizes = [100]
#
#    fig.savefig(f'Figures/SVC/good_locality_explanation.pdf', bbox_inches='tight')
    return fig

#good_locality_plot()


def varying_locality_plot():
    x,y = make_moons(n_samples=1000, noise=0.25, random_state=42)

    df = pd.DataFrame(dict(x=x[:,0], y=x[:,1], label=y))
    df.sort_values(by=['x'], inplace=True)

    teal = (0/255, 128/255, 128/255)  # Teal: RGB(0, 128, 128)
    goldenrod = (218/255, 165/255, 32/255)  # Goldenrod: RGB(218, 165, 32)
    slate_blue = (106/255, 90/255, 205/255)  # Slate Blue: RGB(106, 90, 205)

    class_colors = {0:teal, 1:slate_blue}
    y = df['label']
    df.drop(['label'], axis=1, inplace=True)
    x = df

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    kernel = 'rbf'
    model = SVC(kernel=kernel).fit(X_train, y_train)


    y_pred = model.predict(X_test)


    def get_neighbourhood(instance, X_test, y_test,  distance_threshold):

        def distance(point_one, point_two):
            return ((point_one[0] - point_two[0]) ** 2 +
                    (point_one[1] - point_two[1]) ** 2) ** 0.5

        distances = [distance(x1, instance) for x1 in X_test.values]
        neighbourhood = [i for i in range(len(distances)) if distances[i] <= distance_threshold]
        non_local = [i for i in range(len(distances)) if distances[i] > distance_threshold]
        return neighbourhood, non_local

#    instances = [33,26,27]
#    thresholds = [0.5, 1, 1.5]
    instances = [33, 33, 33]
    thresholds = [1.5, 0.3, 0.6]
    xminmax = [[-1,2],[0,1], [0,1]]

    colors = [goldenrod for i in range(3)]
    fig, axes = plt.subplots(2,2, figsize=(10, 6), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
    ax_num = 0
    for instance_index, distance_threshold, color, xmm in zip(instances, thresholds, colors, xminmax):
#    try:
            ax = fig.get_axes()[ax_num]
            instance = X_test.iloc[instance_index]
            instance_y = y_test.iloc[instance_index]
            neighbourhood, non_local = get_neighbourhood(instance, X_test, y_test, distance_threshold)
            neighbourhood_preds = [y_pred[i] for i in neighbourhood]
            non_local_ys = [y_pred[i] for i in non_local]

            xs = np.array(X_test.iloc[neighbourhood].values)
            xs = list(zip(X_test.iloc[neighbourhood]['x'], X_test.iloc[neighbourhood]['y']))

# Settings for plotting
#            fig, ax = plt.subplots(figsize=(3, 2))
#            ax.set(xlim=(min(X_test['x']), max(X_test['x'])), ylim=(min(X_test['y']), max(X_test['y'])))
#            ax.scatter(X_test.iloc[neighbourhood]['x'], y=X_test.iloc[neighbourhood]['y'], c=neighbourhood_preds, s=200, edgecolors="k", cmap="bwr", label='Local points')
#            ax.scatter(X_test.iloc[non_local]['x'], y=X_test.iloc[non_local]['y'], c=non_local_ys, s=50, edgecolors="k", cmap="bwr", alpha=0.2)
#            ax.set_xlim(-2, 3)
#            ax.set_ylim(-1.2,1.7)

            cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [class_colors[0], class_colors[1]])

            common_params = {"estimator": model, "X": X_test, "ax": ax}
            DecisionBoundaryDisplay.from_estimator(
                    **common_params,

                    response_method="predict",
                    plot_method="pcolormesh",
                    cmap=cmap1,
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
                    label = 'SVC Boundary'
                    #    linestyles=["--", "-", "--"],
                    )
            if instance_index ==33:

                all_xs = copy.deepcopy(X_test['x']).values
                all_ys = copy.deepcopy(X_test['y']).values
                class_0 = np.argwhere(y_pred == 0).flatten()
                class_1 = np.argwhere(y_pred == 1).flatten()

                neighbourhood_xs = X_test.iloc[neighbourhood]['x'].values
                neighbourhood_ys = X_test.iloc[neighbourhood]['y'].values
                neighbourhood_class_0 = np.argwhere(np.array(neighbourhood_preds) == 0).flatten()
                neighbourhood_class_1 = np.argwhere(np.array(neighbourhood_preds) == 1).flatten()


                nl_0 = ax.scatter(all_xs[class_0], y=all_ys[class_0], c=class_colors[0], s=200, alpha=0.2, label='Non-Local points')
                nl_1 = ax.scatter(all_xs[class_1], y=all_ys[class_1], c=class_colors[1], s=200, alpha=0.2, label='Non-Local points')
                l_0 = ax.scatter(neighbourhood_xs[neighbourhood_class_0], y=neighbourhood_ys[neighbourhood_class_0], c=class_colors[0], s=200, edgecolors="k", label='Local points')
                l_1 = ax.scatter(neighbourhood_xs[neighbourhood_class_1], y=neighbourhood_ys[neighbourhood_class_1], c=class_colors[1], s=200, edgecolors="k", label='Local points')
            locality = plt.Circle((instance['x'], instance['y']), radius=distance_threshold, color=color, fill=False, linewidth=3, zorder = 7)
            ax.add_patch(locality)
            t_x =ax.scatter(instance['x'], instance['y'],s=200, color=color, zorder=7, marker='o', label='Target Instance')

## Fit the data to a logistic regression model.
            clf = LogisticRegression()
            X, Y = X_test.iloc[neighbourhood], neighbourhood_preds
            sm = SMOTE(k_neighbors=2)
            X, Y = sm.fit_resample(X, Y)
            clf.fit(X, Y)
# Retrieve the model parameters.
            b = clf.intercept_[0]
            w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
            c = -b/w2
            m = -w1/w2
            xd = np.array(xmm)
            yd = m*xd + c
            exp,= ax.plot(xd, yd, color=color, lw=3, ls='dashed', label='Explanation Surrogate Model', zorder=7)
            exp_preds = clf.predict(X_test.iloc[neighbourhood])

            exp_class_0 = np.argwhere(np.array(exp_preds) == 0).flatten()
            exp_class_1 = np.argwhere(np.array(exp_preds) == 1).flatten()
#            plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
#            plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
            instance_pred= clf.predict(instance.values.reshape(1,-1))
            ax.scatter(instance['x'], y=instance['y'], c=instance_pred, cmap='Greys', edgecolors=class_colors[int(instance_pred)],  s=30, label='Explanation Prediction', zorder=8)
            exp_0 = ax.scatter(neighbourhood_xs[exp_class_0], y=neighbourhood_ys[exp_class_0], c='k', edgecolors=class_colors[0], s=30, label='Explanation Prediction', zorder=6)
            exp_1 = ax.scatter(neighbourhood_xs[exp_class_1], y=neighbourhood_ys[exp_class_1], c='w', edgecolors=class_colors[1], s=30, label='Explanation Prediction', zorder=6)
            ax.set_ylabel(r'$x_2$')
            ax.set_xlabel(r'$x_1$')



###exp = SVC(kernel='linear').fit(X_test.iloc[neighbourhood], y_test.iloc[neighbourhood])
            common_params = {"estimator": clf, "X": X_test.iloc[neighbourhood], "ax": ax}
#            DecisionBoundaryDisplay.from_estimator(
#                    **common_params,
#
#                    response_method="predict",
#                    plot_method="pcolormesh",
#                    cmap="bwr",
#                    alpha=0.3,
#                    )
#            DecisionBoundaryDisplay.from_estimator(
#                    **common_params,
#                    response_method="decision_function",
#                    plot_method="contour",
#                    #    levels=[-1, 0, 1],
#                    levels = [0],
#                    colors=["white"],
#                    linestyles=["--"],
#                    linewidths=[5],
#                    )
#ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
#            plt.axis('scaled')
            ax.set_xlim(-2,3)
            ax.set_ylim(-1.2, 1.7)
            ax.set_aspect('equal', 'box')
            ax_num +=1

            ax.grid(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
#            handles, labels = plt.gca().get_legend_handles_labels()

    fig.get_axes()[0].text(x=0.5, y=-1.8, s='(a)', fontsize=11, ha='center', va='center')
    fig.get_axes()[1].text(x=0.5, y=-1.8, s='(b)', fontsize=11, ha='center', va='center')
    fig.get_axes()[2].text(x=0.5, y=-2, s='(c)', fontsize=11, ha='center', va='center')
    fig.get_axes()[3].text(x=0.5, y=-2, s='(d)', fontsize=11, ha='center', va='center')
    fig = good_locality_plot(fig, class_colors, colors)
#    plt.axis('scaled')
    svc = mlines.Line2D([],[],color='black')
    locality_patch = mlines.Line2D([],[],color=colors[0])

#            order = [1,2,4,6,0,3,5]
    legend = fig.legend(handles=[(nl_0, nl_1), (l_0, l_1), (svc), t_x, locality_patch, ( exp ), (exp_0, exp_1)], labels = ['Non-Local Points', 'Local Points', 'SVC Decsision Boundary','Target Instance', 'Locality Around Instance', 'Explanation Surrogate Model', 'Explanation Predictions'],
       handler_map={tuple: HandlerTuple(ndivide=None, pad=1)}, loc='upper center', ncol=4, fontsize=11, bbox_to_anchor=(0.5,1.05), labelspacing=1, borderpad=1)
    print(legend.legendHandles[5])
    legend.legendHandles[3]._sizes = [100]

    fig.savefig(f'Figures/SVC/varying_locality_explanation.pdf', bbox_inches='tight')

#ax.set_aspect('equal', 'box')

#    except:
#        pass


varying_locality_plot()


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
    axes.plot(x[60:80], [m*i+c for i in x[60:80]], color='green', linewidth=2, linestyle='--', label = 'k=3')
    axes.set_xlabel(r'$x_2$', fontsize=11)
    axes.scatter(x[80:100],y[80:100], s=10, c='pink', alpha=0.5, label='Local Points')
    m, c =  LR(x[80:100],y[80:100])
    axes.plot(x[80:100], [m*i+c for i in x[80:100]], color='pink', linewidth=2, linestyle='--', label = 'k=4')
    axes.set_xlabel(r'Feature 2', fontsize=11)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,4,6,0,3,5]
    fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.35))
    fig.savefig('Figures/feature2_sampleClustering.pdf', bbox_inches='tight')
plot_sample_clustering()



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

def bad_exp_plotter():
    xs = [i for i in range(50)]
    ys = [3*i+np.random.normal(0,8,1) for i in xs]

    small_xs1 = xs[18:23]
    small_xs2 = xs[19:22]
    good_xs = xs[10:30]
    selected_x = 30
    exp1 = lambda xs: [3*i for i in xs]
    exp2 = lambda xs: [20*i-(20*20-60) for i in xs]
    exp3 = lambda xs: [-6*i-(-6*20-60) for i in xs]

    fig, axes = plt.subplots(1, 1, figsize=(4, 2.5))
    data = axes.scatter(xs, ys, color='black', s=10, label='Training Data')
    t_x = axes.scatter(20,60, color='red', s=40, edgecolors='black', zorder=10, label='Target Instance')
    e1, = axes.plot(good_xs, exp1(good_xs), color='green', linewidth=3, linestyle='dashed')
    e2, = axes.plot(small_xs2, exp2(small_xs2), color='blue', linewidth=3, linestyle='dashed')
    e3, = axes.plot(small_xs1, exp3(small_xs1), color='orange', linewidth=3, linestyle='dashed')
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlabel(r'$x$', fontsize=14)
    axes.set_ylabel(r'$\hat{y}$', fontsize=14)

    e1 = mlines.Line2D([],[],color='green')
    e2 = mlines.Line2D([],[],color='orange')
    e3 = mlines.Line2D([],[],color='blue')
#    axes.add_artist(e1)
#    legend = fig.legend(handles=[data, t_x, (e1,e2,e3)], labels = ['Training Data', 'Target Instance', 'Explanation Surrogate Models'],
#       handler_map={tuple: HandlerTuple(ndivide=None, pad=1)}, loc='upper center', ncol=4, fontsize=11, bbox_to_anchor=(0.5,1), labelspacing=1)

    fig.savefig('Figures/good_pred_bad_exp.pdf', bbox_inches='tight')

#bad_exp_plotter()



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


def normalise(exp):
    if len(np.unique(exp)) == 1:
        return(exp)
    else:
        max_val = max(abs(max(exp)), abs(min(exp)))
        norm = lambda x: x/max_val
        normalised = list(map(norm, exp))
        return normalised


def solo_exp(dataset='MIDAS', kernel_width=0.1, mode='same', primary_instance='', num_instances=10):

    if dataset == 'MIDAS':
        features = ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date']
    elif dataset == 'PHM08':
        features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    if primary_instance != '':
        primary_instance = '_'+str(primary_instance)
    try:
        with open(f'saved/results/{dataset}/{dataset}{primary_instance}_{num_instances}_{mode}_kw={kernel_width}.pck', 'rb') as f:
            results = pck.load(f)
    except:
        return None

#    fig, axes = plt.subplots(2, 1, figsize=(1.5, 6), gridspec_kw={'height_ratios': [1, 8], 'width_ratios': [0.7, 1]})
    fig = plt.figure(figsize=(1, 4))
    grid = plt.GridSpec(8, 20, wspace=3, hspace=0.5)
    ax1 = fig.add_subplot(grid[:2, :13])
    ax2 = fig.add_subplot(grid[2:,:])

    axes = [ax1, ax2]


    maxb = max([max([r for r in results[i]['predictions']]) for i in range(3)])
    minb = min([min([r for r in results[i]['predictions']]) for i in range(3)])

    errors = [[abs(p - m) for p, m in zip(results[i]['predictions'], results[-1])] for i in range(3)]
#    maxb = max([max(e) for e in errors])
#    minb = min([min(e) for e in errors])

    norm = matplotlib.colors.Normalize(minb,maxb)
    colors = [[norm(minb), "white"],
              [norm(maxb), "black"]]

    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


    axes = fig.get_axes()

    methods = ['LLC']

    method_preds = results[2]['predictions']
    predictions = copy.deepcopy(results[4])
    explanation_models = results[2]['explanations']

#        if i == 0 or i == 1:
#            explanation_models = [exp_sorter(e, features = features) for e in explanation_models]
    explanation_models = list(map(normalise, explanation_models))


    outliers = []
    instances = list(results[3])
    [predictions.pop(instances.index(o)) for o in outliers]
    [explanation_models.pop(instances.index(o)) for o in outliers]
    [method_preds.pop(instances.index(o)) for o in outliers]
    [instances.pop(instances.index(o)) for o in outliers]

    max_val = np.array(explanation_models).max()
    min_val = np.array(explanation_models).min()
    boundary = max(abs(max_val), abs(min_val))
    if boundary == 0:
        boundary = 1
#        boundary = 1

    norm = matplotlib.colors.Normalize(-boundary,boundary)
    colors = [[norm(-boundary), "red"],
                [norm(-0.2*boundary), 'pink'],
                [norm(0), 'lightgrey'],
                [norm(0.2*boundary), 'lightgreen'],
                [norm(boundary), "green"]]

    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    explanation_models = explanation_models[6:7]
    predictions = predictions[6:7]
    method_preds = method_preds[6:7]

    print(explanation_models)
    print(predictions)
    print(method_preds)
    instances = instances[:1]



#    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

    im_exp = sns.heatmap(ax=axes[1], data=np.transpose(explanation_models), cmap=cmap1, vmin = -boundary, vmax=boundary, cbar=False)
    predictions = [[p] for p in predictions]
#        im_pred = sns.heatmap(ax=axes[i][0], data=np.transpose([abs(p-m) for p,m in zip(predictions, method_preds)]), cmap=cmap2, vmin=0.75*min_error, vmax=1.25*max_error, annot=False, fmt='.2f', cbar=False)
    im_pred = sns.heatmap(ax=axes[0], data=[np.array(predictions).flatten(), method_preds], cmap=cmap2, vmin=0.75*minb, vmax=1.25*maxb, cbar=False)

    axes[0].set_xticklabels([])
#    axes[0].set_title(f'{methods[0]} - Average Error: {mean_squared_error(predictions, method_preds, squared=False):.2f}')
    axes[1].set_yticks(np.arange(len(cleaned_midas))+0.5)
    axes[1].set_xticks(np.arange(len(instances))+0.5)
    axes[1].set_yticklabels(cleaned_midas, rotation=0)
    axes[0].set_yticks(np.arange(2)+0.5)
    axes[0].set_yticklabels(['Model Prediction', 'Surrogate Prediction'], rotation=0)
    axes[1].set_xticklabels([], rotation=90)
#    if mode == 'same':
#        print(instances)
#        axes[1].set_xticklabels(instances, rotation=90)
#    elif mode == 'similar':
#        axes[1].set_xticklabels(instances, rotation=90)

#    axins = inset_axes(axes[3], # here using axis of the lowest plot
#               width="20%",  # width = 5% of parent_bbox width
#               height="100%",  # height : 340% good for a (4x4) Grid
#               loc='lower left',
#               bbox_to_anchor=(1.05, 0.3, 1, 1),
#               bbox_transform=axes[1].transAxes,
#               borderpad=0,
#               )
#    axins2 = inset_axes(axes[1],
#                   width="5%",  # width = 5% of parent_bbox width
#                   height="50%",  # height : 50%
#                   loc='lower left',
#                   bbox_to_anchor=(1.05, 0., 1, 1),
#                   bbox_transform=axes[1].transAxes,
#                   borderpad=0,
#                   )

    mappable = im_exp.get_children()[0]
    cbar1 = fig.colorbar(mappable, ax=axes[1], orientation='vertical')

    mappable = im_pred.get_children()[0]
    cbar2 = fig.colorbar(mappable, ax=axes[1], orientation='vertical')

    cbar1_ax = cbar1.ax
    cbar2_ax = cbar2.ax
    cbar1_ax.set_position([0.9, 0.1, 0.75, 0.6])
    cbar2_ax.set_position([1.6, 0.1, 0.75, 0.6])


#    ax.set_xticklabels([np.round(x,2) for x in predictions])


    print('saving')
    plt.savefig(f'Figures/new_exp_style.pdf', bbox_inches='tight')

    # ------ ORIGINAL LIME EXP -----

    fig, axes = plt.subplots(1,1, figsize=(4,4))
    axes.set_yticklabels(features, rotation=0, fontsize=12)

    colours = ['green' if x>= 0 else 'red' for x in explanation_models[0][::-1]]
    axes.barh(cleaned_midas[::-1], explanation_models[0][::-1], color=colours, align='center', label='_nolegend_')
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.set_xlabel('Feature Contribution', fontsize=14)
    axes.set_yticklabels(cleaned_midas[::-1])
    # axes.text(-3000,2.5, 'a)', fontsize=14)
    axes.set_xlim(-1.2, 1.2)
    fig.savefig(f'Figures/old_exp_style.pdf', bbox_inches='tight')




def exp_progression(dataset='MIDAS', kernel_width=0.1, mode='same', primary_instance='', num_instances=10):

    if dataset == 'MIDAS':
        features = ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date']
    elif dataset == 'PHM08':
        features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    if primary_instance != '':
        primary_instance = '_'+str(primary_instance)
#    try:
        with open(f'saved/results/{dataset}/{dataset}{primary_instance}_{num_instances}_{mode}_kw={kernel_width}.pck', 'rb') as f:
            results = pck.load(f)
#    except:
#        return ValueError('No file found')

#    fig = plt.figure(figsize=(14, 3))
#    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, height_ratios=[1, 6])
#    ax0 = fig.add_subplot(spec[0])
#    ax1 = fig.add_subplot(spec[1])
#    ax2 = fig.add_subplot(spec[2])
#    ax3 = fig.add_subplot(spec[3])
#    ax4 = fig.add_subplot(spec[4])
#    ax5 = fig.add_subplot(spec[5])

    fig = plt.figure(figsize=(10, 3))
    grid = plt.GridSpec(8, 7, wspace=0.5, hspace=0.5)
    ax0 = fig.add_subplot(grid[:2, :2])
    ax1 = fig.add_subplot(grid[:2, 2:4])
    ax2 = fig.add_subplot(grid[:2,4:6])

    ax3 = fig.add_subplot(grid[2:, :2])
    ax4 = fig.add_subplot(grid[2:, 2:4])
    ax5 = fig.add_subplot(grid[2:,4:7])

    maxb = max([max([r for r in results[i]['predictions']]) for i in range(3)])
    minb = min([min([r for r in results[i]['predictions']]) for i in range(3)])

    errors = [[abs(p - m) for p, m in zip(results[i]['predictions'], results[-1])] for i in range(3)]
#    maxb = max([max(e) for e in errors])
#    minb = min([min(e) for e in errors])

    norm = matplotlib.colors.Normalize(minb,maxb)
    colors = [[norm(minb), "white"],
              [norm(maxb), "black"]]

    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    axes = [[ax0, ax3], [ax1, ax4], [ax2, ax5]]
    methods = ['LIME', 'CHILLI', 'LLC']
    plot_order = ['CHILLI', 'LIME', 'LLC']

    titles = ['LIME', 'CHILLI', 'ALLEX']
    for i in range(0,3):
#        r = methods.index(plot_order[i])
        r = i
        print(i)
        method_preds = results[r]['predictions']
        predictions = copy.deepcopy(results[4])
        explanation_models = results[r]['explanations']

        if i == 0 or i == 1:
            explanation_models = [exp_sorter(e, features = features) for e in explanation_models]
        if i ==0:
            print(explanation_models)
        explanation_models = list(map(normalise, explanation_models))


        outliers = []
        instances = list(results[3])
        [predictions.pop(instances.index(o)) for o in outliers]
        [explanation_models.pop(instances.index(o)) for o in outliers]
        [method_preds.pop(instances.index(o)) for o in outliers]
        [instances.pop(instances.index(o)) for o in outliers]

        max_val = np.array(explanation_models).max()
        min_val = np.array(explanation_models).min()
        boundary = max(abs(max_val), abs(min_val))
        if boundary == 0:
            boundary = 1
#        boundary = 1

        norm = matplotlib.colors.Normalize(-boundary,boundary)
        colors = [[norm(-boundary), "red"],
                    [norm(-0.2*boundary), 'pink'],
                    [norm(0), 'lightgrey'],
                    [norm(0.2*boundary), 'lightgreen'],
                    [norm(boundary), "green"]]

        cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)





#    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

        im_exp = sns.heatmap(ax=axes[i][1], data=np.transpose(explanation_models), cmap=cmap1, vmin = -boundary, vmax=boundary, cbar=False)
        predictions = [[p] for p in predictions]
#        im_pred = sns.heatmap(ax=axes[i][0], data=np.transpose([abs(p-m) for p,m in zip(predictions, method_preds)]), cmap=cmap2, vmin=0.75*min_error, vmax=1.25*max_error, annot=False, fmt='.2f', cbar=False)
        im_pred = sns.heatmap(ax=axes[i][0], data=[np.array(predictions).flatten(), method_preds], cmap=cmap2, vmin=0.75*minb, vmax=1.25*maxb, annot=False, fmt='.2f', cbar=False)

        axes[i][0].set_xticklabels([])
        axes[i][0].set_title(f'{titles[i]} \n Average Error: {mean_squared_error(predictions, method_preds, squared=False):.2f}')
        axes[i][1].set_yticks(np.arange(len(cleaned_midas))+0.5)
        axes[i][1].set_xticks(np.arange(len(instances))+0.5)
        axes[i][1].set_yticklabels(cleaned_midas, rotation=0)
        axes[i][0].set_yticks(np.arange(2)+0.5)
        axes[i][0].set_yticklabels(['Model Prediction', 'Surrogate Prediction'], rotation=0)
        if i !=0:
            axes[i][1].set_yticklabels([])
            axes[i][0].set_yticklabels([])
        else:
            if mode == 'same':
                print(instances)
                axes[i][1].set_xticklabels(instances, rotation=90)
            elif mode == 'similar':
                axes[i][1].set_xticklabels(instances, rotation=90)
        axes[i][1].set_xticklabels([])

    mappable = im_exp.get_children()[0]
    cbar1 = plt.colorbar(mappable, ax=axes[2][1], orientation='vertical', pad=-0.11)

    mappable = im_pred.get_children()[0]
    cbar2 =plt.colorbar(mappable, ax=axes[2][1], orientation='vertical', pad=0.2)

    cbar1_ax = cbar1.ax
    cbar2_ax = cbar2.ax
    cbar1_ax.set_position([0.8, 0.1, 0.75, 0.6])
    cbar2_ax.set_position([0.88, 0.1, 0.75, 0.6])

#    ax.set_xticklabels([np.round(x,2) for x in predictions])


    plt.savefig(f'Figures/{dataset}/{dataset}_{primary_instance}_{mode}_progression_kw={kernel_width}.pdf', bbox_inches='tight')


#for kw in [0.01, 0.1, 0.25, 0.5]:
#    exp_progression('PHM08', kw)
#exp_progression('PHM08', 0.01, 'similar')

def single_method_exp_progression(dataset='MIDAS', method='LIME', mode='similar', primary_instance=None, num_instances=10):

    if dataset == 'MIDAS':
        features = ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date']
    elif dataset == 'PHM08':
        features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    new_results = []

    if dataset == 'MIDAS':
        kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
    elif dataset == 'PHM08':
        kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
    for kernel_width in kernel_widths:
        with open(f'saved/results/{dataset}/{dataset}_{primary_instance}_{num_instances}_{mode}_kw={kernel_width}.pck', 'rb') as f:
            results = pck.load(f)
        new_results.append(results[['LIME', 'CHILLI','LLC'].index(method)])

    new_results.append(results[3])
    new_results.append(results[4])
    results = new_results

    fig = plt.figure(figsize=(2,20 ))
    height_ratios = []
    ncols = 1
#    nrows = int( 2*np.ceil(len(kernel_widths)/2) )
    nrows = len(kernel_widths)*2

    for k in range(int(nrows/2)-1):
        height_ratios.append(1)
        height_ratios.append(6)
    height_ratios.append(1)
    height_ratios.append(12)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, hspace=0.5, figure=fig, height_ratios=height_ratios)
    axes = [[fig.add_subplot(spec[2*i]), fig.add_subplot(spec[2*i+1])] for i in range(len(kernel_widths))]
#    axes = []
#    for i in range(len(kernel_widths)):
#        if i%2 == 0:
#            axes.append([fig.add_subplot(spec[2*i]), fig.add_subplot(spec[2*i+2])])
#        else:
#            axes.append([fig.add_subplot(spec[2*i-1]), fig.add_subplot(spec[2*i+1])])
#        axes.append([fig.add_subplot(spec[2*i]), fig.add_subplot(spec[2*i+1])])
#        axes.a

    errors = [[abs(p - m) for p, m in zip(results[i]['predictions'], results[-1])] for i in range(len(kernel_widths))]
    max_error = max([max(e) for e in errors])
    min_error = min([min(e) for e in errors])

    norm = matplotlib.colors.Normalize(min_error,max_error)
    colors = [[norm(min_error), "white"],
              [norm(max_error), "black"]]

    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

#    axes = [[ax0, ax1], [ax2, ax3], [ax4, ax5], [ax6, ax7]]
#    methods = ['LIME', 'CHILLI', 'LLC']

    for i in range(len(kernel_widths)):
        method_preds = results[i]['predictions']
        predictions = copy.deepcopy(results[-1])
        explanation_models = results[i]['explanations']

#        if i == 0 or i == 1:
        explanation_models = [exp_sorter(e, features = features) for e in explanation_models]
        explanation_models = list(map(normalise, explanation_models))


        outliers = []
        instances = list(results[-2])
        [predictions.pop(instances.index(o)) for o in outliers]
        [explanation_models.pop(instances.index(o)) for o in outliers]
        [method_preds.pop(instances.index(o)) for o in outliers]
        [instances.pop(instances.index(o)) for o in outliers]

        max_val = np.array(explanation_models).max()
        min_val = np.array(explanation_models).min()
        boundary = max(abs(max_val), abs(min_val))
        if boundary == 0:
            boundary = 1
#        boundary = 1

        norm = matplotlib.colors.Normalize(-boundary,boundary)
        colors = [[norm(-boundary), "red"],
                    [norm(-0.2*boundary), 'pink'],
                    [norm(0), 'lightgrey'],
                    [norm(0.2*boundary), 'lightgreen'],
                    [norm(boundary), "green"]]

        cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


#    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

        im_exp = sns.heatmap(ax=axes[i][1], data=np.transpose(explanation_models), cmap=cmap1, vmin = -boundary, vmax=boundary, cbar=False)
        predictions = [[p] for p in predictions]
        im_pred = sns.heatmap(ax=axes[i][0], data=np.transpose([abs(p-m) for p,m in zip(predictions, method_preds)]), cmap=cmap2, vmin=min_error*0.75, vmax=max_error*1.25, annot=False, fmt='.2f', cbar=False)

        axes[i][0].set_xticklabels([])
        axes[i][0].set_title(f'{method} ({kernel_widths[i]}) - Average Error: {mean_squared_error(predictions, method_preds, squared=False):.2f}')
        axes[i][1].set_yticks(np.arange(len(features))+0.5)
        axes[i][1].set_xticks(np.arange(len(instances))+0.5)
        axes[i][1].set_yticklabels(features, rotation=0)
        axes[i][0].set_yticks(np.arange(1)+0.5)
        axes[i][0].set_yticklabels(['Explanation Absolute Error'], rotation=0)
        if i <=len(kernel_widths)-2:
            axes[i][1].set_xticklabels([])
        else:
            if mode == 'same':
                axes[i][1].set_xticklabels(instances, rotation=90)
            elif mode == 'similar':
                axes[i][1].set_xticklabels(instances, rotation=90)

    mappable = im_exp.get_children()[0]
    plt.colorbar(mappable, ax=axes[-1][1], orientation='horizontal', pad=0.01)

    mappable = im_pred.get_children()[0]
    plt.colorbar(mappable, ax=axes[-1][1], orientation='horizontal', pad=0.2)


#    ax.set_xticklabels([np.round(x,2) for x in predictions])
    plt.savefig(f'Figures/{dataset}/{method}_{mode}_instances_{primary_instance}_progression.pdf', bbox_inches='tight')

def fidelity_comparison(dataset='MIDAS', mode='random', num_instances=10):

    all_lime_predictions = []
    all_chilli_predictions = []
    all_llc_predictions = []

    kernel_widths = [0.025, 0.05, 0.1, 0.25, 0.5, 1, 5]
    for kernel_width in kernel_widths:
        with open(f'saved/results/{dataset}/{dataset}_{num_instances}_{mode}_kw={kernel_width}.pck', 'rb') as f:
            results = pck.load(f)
        lime_results, chilli_results, llc_results, instances, model_predictions = results
        all_lime_predictions.append(lime_results[0])
        all_chilli_predictions.append(chilli_results[0])
        all_llc_predictions.append(llc_results[0])

#    model_predictions = model_predictions[:10]
    print(instances[0])

    lime_results = []
    chilli_results = []
    for k in range(len(kernel_widths)):
        print(f'\n Kernel Width: {kernel_widths[k]}')
        lime_results.append(mean_squared_error(model_predictions, all_lime_predictions[k], squared=False))
        print(f'LIME: {lime_results[k]}')
        lime_errors = {inst: abs(model_predictions[i] - all_lime_predictions[k][i]) for inst,i in zip(instances,range(len(model_predictions)))}
#        print(lime_errors)
        chilli_results.append(mean_squared_error(model_predictions, all_chilli_predictions[k], squared=False))
        print(f'CHILLI: {chilli_results[k]}')
        chilli_errors = {inst: abs(model_predictions[i] - all_chilli_predictions[k][i]) for inst,i in zip(instances,range(len(model_predictions)))}
#        print(chilli_errors)
    llc_result = mean_squared_error(model_predictions, all_llc_predictions[k], squared=False)
    print(f'LLC: {llc_result}')

def standard_deviation_with_mean(data):
    mean = data[0]  # First value is the mean
    values = data[1:]  # Remaining values
    squared_diffs = [(x - mean) ** 2 for x in values]
    mean_squared_diffs = sum(squared_diffs) / len(squared_diffs)
    std_dev = math.sqrt(mean_squared_diffs)
    return std_dev

def average_error(dataset, mode, num_instances=10):

    new_results = []
    primary_instances = None

    if dataset == 'MIDAS':
        kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5][:]
        features = ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date']
        if mode == 'similar':
            primary_instances = [2140, 1645, 1669, 1435, 2213, 130, 3548, 1635, 3508, 1632] # 2037, 2236, 1870
            primary_instances = [465, 773, 3533, 2751, 404, 2271, 895, 516, 3264, 2768][:]
        elif mode == 'same':
            primary_instances = [122, 1274, 529, 1791, 823, 2087, 3349, 2167, 2778, 1126, 2346, 3674, 1094, 2317, 3035]
        elif mode == 'random':
#            primary_instances = [3257, 630, 1821, 695, 1933, 1855, 938, 416, 182, 1904]
            primary_instances = [226]

    elif dataset == 'PHM08':
        kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
        features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        if mode == 'similar':
            primary_instances = [3121, 4893, 4625, 4165, 2236, 3691, 2270, 825, 4555, 2650]
        elif mode == 'same':
            primary_instances = [2800, 1396, 2390, 1277, 2327, 1967, 2147]
        elif mode == 'random':
            primary_instances = [3763]

    if primary_instances == None:
        for kernel_width in kernel_widths:
            with open(f'saved/results/{dataset}/{dataset}_{num_instances}_{mode}_kw={kernel_width}.pck', 'rb') as f:
                results = pck.load(f)
                new_results.append([results[0]['predictions'], results[1]['predictions'],results[2]['predictions']])


        model_predictions = results[4]
        results = new_results

        methods = ['LIME', 'CHILLI', 'LLC']
        errors = {l:[] for l in methods}
        for kw in kernel_widths:
            print(f'Kernel Width: {kw}')
            for i,m in enumerate(methods):
                print(np.round(mean_squared_error(model_predictions, results[kernel_widths.index(kw)][i], squared=False), 2))
    else:

#        for p in primary_instances:
#            single_method_exp_progression(dataset=dataset, mode=mode, method='CHILLI', primary_instance=p, num_instances=num_instances)
        methods = ['LIME', 'CHILLI', 'LLC']
        all_errors= {l:[] for l in methods}
        all_local_errors = {l:[] for l in methods}
        all_stds = {l:[] for l in methods}
        for kw in kernel_widths:
            errors = {l:[] for l in methods}
            results = {l:[] for l in methods}
            local_errors = {l:[] for l in methods}
            method_stds = {l:[] for l in methods}
            for p in primary_instances:
#                exp_progression(dataset=dataset, mode=mode, kernel_width=kw, num_instances=num_instances, primary_instance=p)


                with open(f'saved/results/{dataset}/{dataset}_{p}_{num_instances}_{mode}_kw={kw}.pck', 'rb') as f:
                    results = pck.load(f)
                    new_results.append([results[0]['predictions'], results[1]['predictions'],results[2]['predictions']][:5])
                    for i in range(3):
#                        pprint(list(zip(results[4], results[i]['predictions'])))
                        errors[methods[i]].append(mean_squared_error(results[4], results[i]['predictions'], squared=False))

                        local_errors[methods[i]].append(np.mean(results[i]['local errors']))
                        explanation_models = results[i]['explanations']
                        if i == 0 or i == 1:
                            explanation_models = [exp_sorter(e, features = features) for e in explanation_models]
                        explanation_models = np.array(list(map(normalise, explanation_models)))
                        stds = []
                        for f in range(len(features)):
                            stds.append(np.std(explanation_models[:,f]))
#                            stds.append(standard_deviation_with_mean(explanation_models[:,f]))
                        method_stds[methods[i]].append(np.mean(stds))




            print(f'\n############ Kernel Width: {kw} ############')
            for m in methods:
                print(f'------ {m} -----')
                all_errors[m].append(np.mean(errors[m]))
                all_local_errors[m].append(np.mean(local_errors[m]))
                all_stds[m].append(np.mean(method_stds[m]))
                if mode == 'random':
                    print(f'Average RMSE: {np.round(np.mean(errors[m]), 4)}')
#                print(f'Local Error: {np.round(np.mean(local_errors[m]), 4)}')
                if mode == 'same' or 'similar':
                    print(f'StD: {np.round(np.mean(method_stds[m]), 3)}')


        fig, ax = plt.subplots()
        twin1 = ax.twinx()
        colours = ['red', 'blue', 'green']
        for i, m in enumerate( methods ):
#            if i == 0:
#                p1, = ax.plot(kernel_widths[], all_errors[m][1:] , label=f'{m} RMSE', color=colours[i], linestyle='--')
#                p2, = twin1.plot(kernel_widths[1:], all_stds[m][1:], label=f'{m} StDs', color=colours[i], linestyle='-')
#            else:
            p1, = ax.plot(kernel_widths, all_errors[m] , label=f'{m} RMSE', color=colours[i], linestyle='--')
            p2, = twin1.plot(kernel_widths, all_stds[m], label=f'{m} Inconsistency', color=colours[i], linestyle='-')
        ax.set_xlabel('Kernel Width')
        ax.set_ylabel('Average RMSE')
        ax.set_xscale('log')
#        ax.set_ylim(0,150)
        twin1.set_ylabel('Average StD')
        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1.3))
        twin1.legend(loc='upper right', bbox_to_anchor=(0.95, 1.3))
        plt.savefig(f'Figures/{dataset}/{mode}_fidelity_comparison.pdf', bbox_inches='tight')

    return all_errors, all_stds


def error_per_kw(dataset, mode, num_instances, method='CHILLI',):

    kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
    if dataset == 'MIDAS':
        kernel_widths = [0.1, 0.25, 0.5, 1, 5]
        primary_instances = 226
    elif dataset == 'PHM08':
        primary_instances = 3763

    fig, ax  = plt.subplots(figsize=(6,3))
    colors = np.random.rand(7,3)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink']
    all_errors = {kw:[] for kw in kernel_widths}
    for kw in kernel_widths:
        with open(f'saved/results/{dataset}/{dataset}_{primary_instances}_{num_instances}_{mode}_kw={kw}.pck', 'rb') as f:
            results = pck.load(f)
            all_errors[kw] = np.array([abs(i-j) for i,j in zip(results[4], results[['LIME', 'CHILLI','LLC'].index(method)]['predictions'])])

    all_errors['llc'] = np.array([abs(i-j) for i,j in zip(results[4], results[2]['predictions'])])

    for i in range(int(num_instances)):
        instance_errors = [all_errors[kw][i] for kw in kernel_widths]
        llc_error = all_errors['llc'][i]
        alphas = [1 if i == min(instance_errors) else 0.2 for i in instance_errors]
        if i == 0:
            for kw in range(len(kernel_widths)):
                ax.scatter(i, instance_errors[kw], label=f'Kernel Width: {kernel_widths[kw]}', color=colors[kw], alpha=alphas[kw], marker='x')
#            ax.scatter(i, [llc_error], label=f'LLC', color='cyan', alpha=1, marker='x')
#            ax.scatter([i for j in range(len(kernel_widths))], instance_errors, label=f'Instance {i}', color=colors, alpha=alphas, marker='x')
        else:
            ax.scatter([i for j in range(len(kernel_widths))], instance_errors, color=colors, alpha=alphas, marker='x', label='_nolegend_')
#            ax.scatter(i, [llc_error], label='_nolegend_', color='cyan', alpha=1, marker='x')

    print(mean_squared_error(results[4], results[2]['predictions'], squared=False))

    ax.set_xlabel('Instance')
    ax.set_ylabel('Absolute Error')
    ax.set_xticks(range(int(num_instances)))
    ax.set_xticklabels(range(int(num_instances)))
    leg = ax.legend([str(kw) for kw in kernel_widths], title='Kernel Width', loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.savefig(f'Figures/{dataset}/{mode}_error_per_kw.pdf', bbox_inches='tight')




#error_per_kw(sys.argv[1], sys.argv[2], sys.argv[3])
#average_error(sys.argv[1], sys.argv[2], sys.argv[3])
#single_method_exp_progression(dataset=sys.argv[1 mode=sys.argv[2], method=sys.argv[3], num_instances=num_instances)
#if sys.argv[1] == 'MIDAS':
#    kernel_widths = [0.01, 0.1, 0.25, 0.5, 1, 5]
#elif sys.argv[1] == 'PHM08':
#    kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
#for kw in kernel_widths:
#    exp_progression(dataset=sys.argv[1], mode=sys.argv[2], kernel_width=kw, num_instances=num_instances, primary_instance=5113)

exp_progression(dataset=sys.argv[1], mode=sys.argv[2], kernel_width=0.1, num_instances=sys.argv[3],  primary_instance=3349)
#solo_exp(dataset=sys.argv[1], mode=sys.argv[2], kernel_width=0.01, num_instances=num_instances, primary_instance=2037)
#fidelity_comparison(dataset=sys.argv[1], mode=sys.argv[2], num_instances=num_instances)
