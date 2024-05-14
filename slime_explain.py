import numpy as np
from slime import lime_tabular
from sklearn.metrics import mean_squared_error


class SLIME():

    def __init__(self, dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features):
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


    def build_explainer(self, categorical_features=None, mode='regression'):
        explainer = lime_tabular.LimeTabularExplainer(self.x_train,
                                                      mode = mode,
                                                      feature_names = self.features,
                                                      discretize_continuous = False,
                                                      feature_selection = "lasso_path",
                                                      sample_around_instance = True)
        return explainer

    def make_explanation(self, predictor, explainer, instance, num_features=25, num_samples=1000):
        ground_truth = self.y_test[instance]
        if self.dataset == 'PHM08':
            predictor = self.model.predict
        instance_prediction = predictor(self.x_test[instance].reshape(1,-1))[0]
        exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, instance_prediction, exp_instance_prediction = explainer.explain_instance(self.x_test[instance], predict_fn=predictor, num_features = num_features, num_samples = 1000)

        print(len(model_perturbation_predictions), len(exp_perturbation_predictions))
        explanation_error = mean_squared_error(model_perturbation_predictions, exp_perturbation_predictions, squared=False)
        return exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction
