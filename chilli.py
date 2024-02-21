import numpy as np
import matplotlib.pyplot as plt
from limeLocal import lime_tabular
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils.extmath import safe_sparse_dot
from pprint import pprint

class CHILLI():

    def __init__(self, dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, newMethod=True, automated_locality=False):
        self.dataset = dataset
        self.model = model
        # These should be scaled numpy arrays
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.features = features
        self.newMethod = newMethod
        self.automated_locality = automated_locality


    def build_explainer(self, categorical_features=None, kernel_width=None, mode='regression'):
#        The explainer is built herem on the training data with the features and type of model specified.
#        y_hat_test = self.model.predict(self.x_test)
        explainer = lime_tabular.LimeTabularExplainer(self.x_train, test_data=self.x_test, test_labels=self.y_test, test_predictions=self.y_test_pred, automated_locality=self.automated_locality, feature_names=self.features, categorical_features=categorical_features, mode=mode, verbose=True, kernel_width=kernel_width)
        return explainer

    def make_explanation(self, predictor, explainer, instance, num_features=25, num_samples=1000):
        ground_truth = self.y_test[instance]
        if self.dataset == 'PHM08':
            predictor = self.model.predict
        instance_prediction = predictor(self.x_test[instance].reshape(1,-1))[0]
        exp, local_model, perturbations, model_perturbation_predictions, exp_perturbation_predictions = explainer.explain_instance(self.x_test[instance], instance_num=instance, predict_fn=predictor, num_features=num_features, num_samples=num_samples, newMethod=self.newMethod)
        self.local_model = local_model
        exp_instance_prediction = local_model.predict(self.x_test[instance].reshape(1,-1))[0]

        explanation_error = mean_squared_error(model_perturbation_predictions, exp_perturbation_predictions, squared=False)
        exp.intercept = self.local_model.intercept_

#        print(f'Ground Truth: {ground_truth}')
#        print(f'Model Prediction: { instance_prediction }')

        if self.newMethod:
            print(f'CHILLI prediction: { exp_instance_prediction }')
        else:
            print(f'LIME prediction: { exp_instance_prediction }')
        return exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction

    def exp_sorter(self, exp_list, features):

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


    def plot_explanation(self, instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature):
        exp_list = exp.as_list()
        fontsize = 10
        fig = plt.figure(figsize=(14,12))
        plt.tight_layout()
        grid = plt.GridSpec(6, 3, hspace=0.35, wspace=0.1)
        expPlot = fig.add_subplot(grid[:2, 1:])

        exp_size = 12
        # Plot the explanation
        all_features = self.features
        explained_features = [i[0] for i in exp_list][:12]
        explained_feature_indices = [all_features.index(i) for i in explained_features]
        explained_feature_perturbations = np.array(perturbations)[:,explained_feature_indices]
        perturbations = explained_feature_perturbations
        feature_contributions = [i[1] for i in exp_list][:12]


        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        expPlot.set_yticklabels(explained_features, rotation=0, fontsize=fontsize)
        expPlot.barh(explained_features, feature_contributions, color=colours, align='center', label='_nolegend_')
        expPlot.tick_params(axis='both', which='major', labelsize=14)
        # expPlot.text(-3000,2.5, 'a)', fontsize=14)
#        expPlot.set_xlim(-2000, 2000)


        perturb1 = fig.add_subplot(grid[1+1,0])
        perturb2 = fig.add_subplot(grid[1+1, 1], sharey=perturb1)
        perturb3 = fig.add_subplot(grid[1+1, 2])
        perturb4 = fig.add_subplot(grid[2+1, 0], sharey=perturb3)
        perturb5 = fig.add_subplot(grid[2+1, 1])
        perturb6 = fig.add_subplot(grid[2+1, 2], sharey=perturb5)
        perturb7 = fig.add_subplot(grid[3+1, 0], sharey=perturb3)
        perturb8 = fig.add_subplot(grid[3+1, 1])
        perturb9 = fig.add_subplot(grid[3+1, 2], sharey=perturb5)
        perturb10 = fig.add_subplot(grid[4+1, 0], sharey=perturb3)
        perturb11 = fig.add_subplot(grid[4+1, 1])
        perturb12 = fig.add_subplot(grid[4+1, 2], sharey=perturb5)
        perturbation_plots = [perturb1, perturb2, perturb3, perturb4, perturb5, perturb6, perturb7, perturb8, perturb9, perturb10, perturb11, perturb12]

        for plot in [perturb2, perturb3, perturb5, perturb6, perturb8, perturb9, perturb11, perturb12]:
            plt.setp(plot.get_yticklabels(), visible=False)
            plt.setp(plot.set_ylabel(''), visible=False)


        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), np.array(model_perturbation_predictions[0]), np.array(exp_perturbation_predictions[0])
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
#        y_perturb_full = [x[0] for x in y_perturb_full]

        perturbation_weights = exp.weights[1:]

        for i in range(len(perturbation_plots)):
            perturbation_plots[i].scatter(instance_x[i], instance_model_y ,c="red",marker="o", s=100)

#            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_exp_y, c=perturbation_weights, cmap='Oranges', s=3)
#            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_model_y, c=perturbation_weights, cmap='Greens', s=3)
            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_exp_y, color='Orange',  s=3, alpha=0.9)
            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_model_y, color='green', s=3, alpha=0.9)

            perturbation_plots[i].scatter(instance_x[i], instance_model_y, c="red",marker="o", s=100, label='_nolegend_')
            perturbation_plots[i].set_title(explained_features[i], fontsize=fontsize)

#            minX = min(x_perturb[:,i])
#            minX = min(minX,-0.05)
#            maxX = max(x_perturb[:,i])
#            maxX = max(maxX,1)
#            minY = min(y_perturb_full)
#            minY = min(minY,0)
#            maxY = max(y_perturb_full)
#            maxY = max(maxY,1)

#            perturbation_plots[i].set_ylim(-750,1000)
#            perturbation_plots[i].set_xlim(-0.7, 1.4)
            perturbation_plots[i].set_ylabel(targetFeature, fontsize=fontsize)
            perturbation_plots[i].tick_params(axis='both', which='major', labelsize=14)
            perturbation_plots[i].grid('x')

        # for plot in [perturb1, perturb2, perturb3]:
        #     plt.setp(plot.get_xticklabels(), visible=False)
        #     plt.setp(plot.set_xlabel(''), visible=False)

        for plot in [perturb1, perturb4]:
            # plt.setp(plot.get_xticklabels(), visible=False)
            plt.setp(plot.set_ylabel(targetFeature), fontsize=fontsize)
        fig.savefig(f'Figures/CHILLI/{instance}_explanation_{suffix}.pdf', bbox_inches='tight')


    def interactive_perturbation_plot(self, instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature, neighbours=None):

        exp_list = exp.as_list()
#        explained_features = [i[0] for i in exp_list]
##        all_features = ['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
#        all_features = self.features
#        unsorted_instance = perturbations[0]
#        explained_feature_indices = []
#        for i in range(len(explained_features)):
#            for j in range(len(explained_features)):
#                if all_features[j] in explained_features[i]:
#                    explained_feature_indices.append(j)
#                    break
#
##        explained_feature_indices = [all_features.index(i) for i in explained_features]
#        explained_feature_perturbations = np.array(perturbations)[:,explained_feature_indices]
#        explained_features_x_test = np.array(self.x_test)[:,explained_feature_indices]
#        perturbations = explained_feature_perturbations
#        feature_contributions = [i[1] for i in exp_list]

        feature_contributions = self.exp_sorter(exp_list, self.features)
        explained_features = self.features
        explained_features_x_test = self.x_test
#
        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), int(np.array(model_perturbation_predictions[0])), int(np.array(exp_perturbation_predictions[0]))
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
        perturbations_model_y = [int(i) for i in perturbations_model_y]
        explanation_error = mean_squared_error(perturbations_model_y, perturbations_exp_y)


        perturbation_weights = exp.weights[1:]

        num_rows=int(np.ceil(len(explained_features)/4))+1
        num_cols=4

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[0.33]+[0.16]*(num_rows-1),
                            specs = [
                                [{'colspan':2}, None, {'colspan':2}, None],
                                ]+[[{}, {}, {}, {}] for i in range(num_rows-1)], subplot_titles=['Explanation Prediction Convergence', 'Feature Significance']+explained_features,
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        # Plot convergence of error as features are added
#        exp_convergence = []
#        for included_features in range(len(explained_features)):
#            intercept = exp.intercept
#            for i in range(included_features):
#                intercept+= feature_contributions[i]*instance_x[i]
#
#            exp_convergence.append(intercept)
#
#        fig.add_trace(go.Scatter(x=[num for num in range(len(exp_convergence))], y=exp_convergence, mode='lines', marker = dict(color='orange', size=3), showlegend=False), row=1, col=1)


        # Plot explanation bar chart
        fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=3)

        axes = [[row, col] for row in range(2,num_rows+1) for col in range(1,num_cols+1)]

#        for n in range(len(neighbours)):
#            neighbours[n] = [neighbours[n][i] for i in explained_feature_indices]
#
        for i in range(len(self.features)):
#        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='orange', size=3)), row=ax[0], col=ax[1])
            if i==0:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(x=explained_features_x_test[:,i],y=self.y_test_pred,
                                     mode='markers', marker = dict(color='lightgrey', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Test data'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_model_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Oranges', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Model (f) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Greens', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[instance_model_y],
                                     mode='markers', marker = dict(color='red', size=20),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[exp.local_model.predict(instance_x.reshape(1,-1))],
                                     mode='markers', marker = dict(color='blue', size=10, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])


        fig.update_layout(title=dict(text = f' Explanation for instance {instance} <br> Explanation Error = {explanation_error:.2f} <br> Model Instance Prediction {instance_model_y} <br> Explanation Instance Prediction {instance_exp_y}', y=0.99, x=0),
                          font=dict(size=14),
                          legend=dict(yanchor="top", y=1.1, xanchor="right"),
                          height=300*num_rows, )
        if self.automated_locality == True and self.newMethod == True:
            suffix = 'LLC'
        elif self.automated_locality == False and self.newMethod == True:
            suffix = '_CHILLI'
        elif self.automated_locality == False and self.newMethod == False:
            suffix = '_LIME'
        else:
            suffix = ''
        fig.write_html(f'Figures/{self.dataset}/Explanations/instance_{instance}{suffix}_explanation.html', auto_open=False)
