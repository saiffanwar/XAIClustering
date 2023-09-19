import numpy as np
import matplotlib.pyplot as plt
from limeLocal import lime_tabular
from sklearn.metrics import mean_squared_error

class CHILLI():

    def __init__(self, model, x_train, y_train, x_test, y_test, features):
        self.model = model
        # These should be scaled numpy arrays
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_pred = model.predict(x_test)
        self.y_test = y_test
        self.features = features


    def build_explainer(self, categorical_features=None, kernel_width=None, mode='classification'):
#        The explainer is built herem on the training data with the features and type of model specified.
#        nptrain = np.array(self.x_train)
        nptrain = self.x_train
        explainer = lime_tabular.LimeTabularExplainer(nptrain, feature_names=self.features, categorical_features=categorical_features, mode=mode, verbose=True, kernel_width=kernel_width)
        return explainer
#    def preidctor_function(self, testData):
#        predict_proba = self.model.predict_proba(testData)
#        return


    def make_explanation(self, explainer, instance, newMethod=True, num_features=9, num_samples=5000):
#        print(f'Explaining Data instance {instance} {self.x_train.iloc[instance]}')
        predictor = self.model.predict
#        nptest = np.array(self.x_test)
        nptest = self.x_test
        print(f'Generating {num_samples} samples')
        exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions = explainer.explain_instance(nptest[instance], predictor, num_features=num_features, num_samples=5000, newMethod=newMethod)
#        prediction = predictor(nptest[instance].reshape(1,-1))

#        model_perturbation_predictions = [p[1] for p in model_perturbation_predictions]
        explanation_error = mean_squared_error(model_perturbation_predictions, exp_perturbation_predictions)

        return exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error

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

            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_exp_y, c=perturbation_weights, cmap='Oranges', s=3, alpha=0.9)
            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_model_y, c=perturbation_weights, cmap='Greens', s=3, alpha=0.9)
#            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_exp_y, color='Orange',  s=3, alpha=0.9)
#            perturbation_plots[i].scatter(perturbations_x[:,i],perturbations_model_y, color='green', s=3, alpha=0.9)

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

        fig.savefig(f'Figures/CHILLI/{instance}_explanation.pdf', bbox_inches='tight')
