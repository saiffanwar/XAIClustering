import midas_model_data as midas
import torch.optim as optim
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import pickle as pck
import random
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


from MidasDataProcessing import MidasDataProcessing
from midasDistances import combinedFeatureDistances, calcAllDistances, calcSingleDistance, pointwiseDistance
from chilli import CHILLI
from llc_explainer import LLCExplanation
from llc_ensemble_generator import LLCGenerator

is_cuda = torch.cuda.is_available()
is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

parser = argparse.ArgumentParser(description='Run the MIDAS model and generate explanations')
parser.add_argument('-d', '--dataset', type=str, default='MIDAS', help='Which dataset to work on')
parser.add_argument('-l', '--load_model', type=bool, default=True, help='Load the model from file')
parser.add_argument('-m', '--mode', type=str, default='explain', help='Whether to generate ensembles or explanations.')
parser.add_argument('-e', '--exp_mode', type=str, default='random', help='Which instances to generate explanations for.')
parser.add_argument('-k', '--kernel_width', type=float, default=None, help='The kernel width to use for the CHILLI explainer.')
parser.add_argument('--lime_exp', type=bool, default=True, help='Whether to use LIME explanations.')
parser.add_argument('--chilli_exp', type=bool, default=True, help='Whether to use CHILLI explanations.')
parser.add_argument('--sparsity', type=float, default=0.05, help='The sparsity threshold to use for the LLC explainer.')
parser.add_argument('--coverage', type=float, default=0.05, help='The coverage threshold to use for the LLC explainer.')
parser.add_argument('--starting_k', type=int, default=10, help='The number of neighbours to use for the LLC explainer.')
parser.add_argument('--neighbourhood', type=float, default=0.05, help='The neighbourhood threshold to use for the LLC explainer.')


args = parser.parse_args()

class MIDAS():

    def __init__(self, load_model=True):
        self.load_model = load_model
        self.data = MidasDataProcessing(linearFeaturesIncluded=True)
        cleanedFeatureNames = ['Heathrow wind speed', 'Heathrow wind direction', 'Heathrow total cloud cover', 'Heathrow cloud base height', 'Heathrow visibility', 'Heathrow MSL pressure', 'Date']
        self.df = self.data.create_temporal_df(mainLocation='heathrow')
        self.train_loader, self.val_loader, self.test_loader, self.train_loader_one, self.test_loader_one = self.data.datasplit(self.df, 'heathrow air_temperature')

    def train_midas_rnn(self,):

        self.input_dim = self.data.inputDim
        self.output_dim = 1
        self.hidden_dim = 12
        self.layer_dim = 3
        self.batch_size = 64
        self.dropout = 0.2
        self.n_epochs = 1000
        self.learning_rate = 1e-2
        self.weight_decay = 1e-6

        model_params = {'input_dim': self.input_dim,
                        'hidden_dim' : self.hidden_dim,
                        'layer_dim' : self.layer_dim,
                        'output_dim' : self.output_dim,
                        'dropout_prob' : self.dropout}

        model_path = f'saved/models/MIDAS_model.pck'
        self.model = midas.RNNModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim, self.dropout)

        if self.load_model:
            self.model.load_state_dict(torch.load(model_path))
        loss_fn = nn.L1Loss(reduction="mean")
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.opt = midas.Optimization(model=self.model.to(device), loss_fn=loss_fn, optimizer=optimizer)

        if not self.load_model:
            self.opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
#            opt.plot_losses()
            torch.save(self.model.state_dict(), model_path)

        return self.opt.predictor_from_numpy

    def make_midas_predictions(self,):
        train_pred, values = self.opt.evaluate(self.train_loader_one, batch_size=1, n_features=self.input_dim)
        test_pred, values = self.opt.evaluate(self.test_loader_one, batch_size=1, n_features=self.input_dim)

        self.train_preds = np.array(train_pred).flatten()
        self.test_preds = np.array(test_pred).flatten()

#        fig = midas.plotPredictions(features, data.X_test, vals, preds)
#        fig = midas.plotPredictions(self.data.trainingFeatures, self.data.X_train, self.vals, self.preds)
#        plt.tight_layout()
#        fig.savefig('Figures/MidasPredictions.pdf', bbox_inches='tight')
#        plt.show()
        return self.train_preds, self.test_preds


class RUL():

    def __init__(self, load_model=True):
        self.load_model = load_model

    def data_preprocessing(self,):

        self.data = pd.read_csv('Data/PHM08/PHM08.csv')

        self.features = self.data.drop(['RUL','cycle', 'id'], axis=1).columns.tolist()

        # ----------------------------------------------------------
        # RERUN ALL EXPERIMENTS USING THE FOLLOWING DATA SPLIT METHOD

        train_df = self.data[self.data['id'] <= 150]
        test_df = self.data[self.data['id'] > 150]

        scaler = StandardScaler()
#        x_train = scaler.fit_transform(x_train)
#        x_test = scaler.fit_transform(x_test)
        self.x_train = scaler.fit_transform(train_df.drop(['RUL','cycle', 'id'], axis=1).values)
        self.x_test = scaler.fit_transform(test_df.drop(['RUL', 'cycle','id'], axis=1).values)
        self.y_train = train_df['RUL'].values
        self.y_test = test_df['RUL'].values

        # ----------------------------------------------------------

        return self.x_train, self.x_test, self.y_train, self.y_test, self.features

#    def data_visualisation(self,):
#        self.data = pd.read_csv('Data/PHM08/PHM08.csv')
#        for col in self.data.columns:
##        for i in data['id'].unique():
#            fig, axes = plt.subplots(1, 1, figsize=(10, 4))
#            axes.scatter(col, 'RUL', data=self.data, alpha=0.5,s=1)
#            fig.savefig('Figures/PHM08/' + col + '.png')
#data_visualisation()

    def train(self):
        model = GradientBoostingRegressor(max_depth=5, n_estimators=500, random_state=42)
        model.fit(self.x_train, self.y_train)
        with open('saved/models/PHM08_model.pck', 'wb') as file:
            pck.dump(model, file)
        return model

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        y_hat_train = model.predict(X_train)
        print('training RMSE: ',mean_squared_error(y_train, y_hat_train),)
        y_hat_test = model.predict(X_test)
        print('test RMSE: ',mean_squared_error(y_test, y_hat_test))

        return y_hat_train, y_hat_test


def run_clustering(model, x_test,y_pred, dataset, features, discrete_features, search_num, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold):

    print('Starting thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)
    GLE = LLCGenerator(model=model, x_test=x_test, y_pred=y_pred, features=features, discrete_features=discrete_features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=preload_explainer)
    GLE.multi_layer_clustering(search_num)
    print('finishing thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)


def chilli_explain(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=25, automated_locality=False, newMethod=True, kernel_width=None, noisey_instance=None, categorical_features=None, neighbours=None):
    chilliExplainer = CHILLI(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction = chilliExplainer.make_explanation(model, explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'target', neighbours=neighbours)
    with open(f'saved/explanation_{instance}.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)
    plotting_data = [x_test[instance], instance, perturbations, model_perturbation_predictions, y_test[instance], y_train[instance], exp_instance_prediction, exp_perturbation_predictions, exp]

    return explanation_error, instance_prediction, exp_instance_prediction, plotting_data


def main(primary_instance=None):
    dataset = args.dataset

    if dataset == 'MIDAS':
        midas_runner = MIDAS(load_model=args.load_model)
        x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures

        target_feature = 'heathrow air_temperature'
        discrete_features = ['heathrow cld_ttl_amt_id']
        model = midas_runner.train_midas_rnn()
        y_train_pred, y_test_pred = midas_runner.make_midas_predictions()

        sampling=False
        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.05, 0.05, 10, 0.05
        x_test = x_train
        y_test = y_train
        y_test_pred = y_train_pred

    elif dataset == 'PHM08':
        phm08_runner = RUL()
        x_train, x_test, y_train, y_test, features = phm08_runner.data_preprocessing()
        target_feature = 'RUL'
        discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        with open(f'saved/models/PHM08_model.pck', 'rb') as file:
            model = pck.load(file)
#        model = phm08_runner.train()
        y_train_pred, y_test_pred = phm08_runner.evaluate(model, x_train, x_test, y_train, y_test)

        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.5, 0.05, 5, 0.5
        sampling = True

    categorical_features = [features.index(feature) for feature in discrete_features]
    if sampling:
        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), 5000)

        x_train = x_train[random_samples]
        y_train = y_train[random_samples]
#        model = train(x_train, y_train, features)
#        y_train_pred, y_test_pred = evaluate(model, x_train, x_test, y_train, y_test)

        x_test = x_test[random_samples]
        y_pred = y_test_pred[random_samples]
        y_test = y_test[random_samples]
        print(f'Training samples: {len(x_train)}')
        print(f'Test samples: {len(x_test)}')

#    sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = args.sparsity, args.coverage, args.starting_k, args.neighbourhood

    if args.mode == 'ensembles':
        run_clustering(model, x_test, y_test, dataset, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)

    elif args.mode=='explain':
#        x_test = x_train
#        y_test = y_train
#        y_test_pred = y_train_pred

        LLCGen = LLCGenerator(model, x_test, y_test_pred, features, discrete_features, dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)
        GLE = LLCExplanation(model=model, x_test=x_test, y_pred=y_test_pred, features=features, discrete_features=discrete_features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)


        primary_instances = [random.randint(0, len(x_test)) for i in range(9)]
        for primary_instance in primary_instances:
            if primary_instance == None:
                instances = [random.randint(0, len(x_test)) for i in range(1)]
            else:
                instances = [primary_instance]
            instance = instances[0]

            # ---- Similar Instances ----
            if args.exp_mode == 'similar':
                new_eval = False

                if new_eval:

                    llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_test_pred[instance], y_test[instance])
                    data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data
                    relevant_instances = GLE.new_evaluation(instance, instance_explanation_model, x_test[instance], importance_threshold = 10)
                    instances = relevant_instances
#                combined_data = np.concatenate((x_test, y_test_pred.reshape(-1,1)), axis=1)
                    combined_data = x_test
                    if dataset == 'MIDAS':
                        relevant_instance_distances = [pointwiseDistance(x_test[instance], x_test[r], features) for r in relevant_instances]
                    elif dataset == 'PHM08':
                        relevant_instance_distances = np.linalg.norm(combined_data[relevant_instances]-combined_data[instance], axis=1)
                    instances = instances[np.argsort(relevant_instance_distances)][:10]
                else:
                    if dataset == 'MIDAS':
                        distances = [pointwiseDistance(x_test[instance], x_test[r], features) for r in range(len(x_test))]
                    elif dataset == 'PHM08':
                        distances = np.linalg.norm(x_test-x_test[instance], axis=1)
                    closest_instances = np.argsort(distances)[:10]
                    instances = closest_instances
#
                LLCGen.plot_all_clustering(instance = instances[0], instances_to_show=instances[1:])

            # ---- Same Instances ----
            elif args.exp_mode == 'same':
                instances = [instance for i in range(10)]

            #  ---- Random Instances ----
            elif args.exp_mode == 'random':
                instances = [random.randint(0, len(x_test)) for i in range(10)]
                LLCGen.plot_all_clustering(instance = instances[0], instances_to_show=[])
#            instances = [4302, 794, 3928, 4358, 1240, 3880, 1934, 2545, 1096, 2807]

            print('Generating explanations for instances: ', instances)

            # ---- Explanation Generation ----

            if args.kernel_width == None:
                if args.dataset == 'MIDAS':
                    kernel_widths = [0.1, 0.25, 0.5, 1, 5]
                elif args.dataset == 'PHM08':
                    kernel_widths = [0.01, 0.025, 0.1, 0.25, 0.5, 1, 5]
            else:
                kernel_widths = [args.kernel_width]

            for kw in kernel_widths:
                model_predictions, lime_predictions, chilli_predictions, llc_predictions = [], [], [], []
                lime_explanations, chilli_explanations, llc_explanations = [], [], []

                for instance in tqdm(instances):
#                    GLE.plot_all_clustering(instance=instance)
                    print(f'################# Instance  = {instance} ###################')

                    # ------ BASE MODEL ------
#                print(f'Ground Truth: {y_test[instance]}')
#                print(f'Model Prediction: {y_test_pred[instance]}')

                    model_predictions.append(y_test_pred[instance])

                    # ---- LIME EXPLANATION -------
                    if args.lime_exp:

                        print('\n ----- LIME EXPLANATION -----')
                        _,_, lime_prediction, lime_plotting_data = chilli_explain(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred,features, instance=instance, newMethod=False, kernel_width=kw, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, lime_exp = lime_plotting_data
                        print(f'LIME Error: {abs(ground_truth-lime_prediction)}')
                        lime_predictions.append(lime_prediction)
                        lime_explanations.append(lime_exp.as_list())

                    # ---- CHILLI EXPLANATION -------
                    if args.chilli_exp:
                        print('\n ----- CHILLI EXPLANATION -----')
                        _,_, chilli_prediction, chilli_plotting_data = chilli_explain(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=instance, newMethod=True, kernel_width=kw, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, chilli_exp = chilli_plotting_data

                        print(f'CHILLI Error: {abs(ground_truth-chilli_prediction)}')
                        chilli_predictions.append(chilli_prediction)
                        chilli_explanations.append(chilli_exp.as_list())

                    # ---- LLC EXPLANATION -------
                    print('\n ----- LLC EXPLANATION -----')
                    llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_test_pred[instance], y_test[instance])

                    data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data

                    print(f'LLC Error: {abs(ground_truth-llc_prediction)}')
                    llc_predictions.append(llc_prediction)
                    llc_explanations.append(instance_explanation_model.coef_)

                    GLE.interactive_exp_plot(data_instance, instance_index, instance_prediction, y_test_pred, exp_instance_prediction, instance_explanation_model.coef_, local_x, local_x_weights, local_y_pred, exp_local_y_pred, target_feature)

                with open(f'saved/results/{dataset}/{dataset}_{primary_instance}_{len(instances)}_{args.exp_mode}_kw={kw}.pck', 'wb') as f:
                    pck.dump([[lime_predictions, lime_explanations], [chilli_predictions, chilli_explanations], [llc_predictions, llc_explanations], instances, model_predictions ], f)


if __name__ == '__main__':
    main()
