import midas_model_data as midas
import torch.optim as optim
import torch
from MidasDataProcessing import MidasDataProcessing
from midasDistances import combinedFeatureDistances, calcAllDistances
from chilli import CHILLI
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import pickle as pck
import random
from tqdm import tqdm

from run_linear_clustering import GlobalLinearExplainer

is_cuda = torch.cuda.is_available()
is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class MIDAS():

    def __init__(self, load_model=False):
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

        return self.model

    def make_midas_predictions(self,):
#        for x_test, y_test in self.train_loader:
#            # print(len(x_test))
#            x_test = (x_test.to(torch.float32).reshape([64, -1, self.input_dim])).to(device)
#            # x_test = torch.from_numpy(x_test)
#            pred = self.model(x_test)
#            break

#predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
        train_pred, values = self.opt.evaluate(self.train_loader_one, batch_size=1, n_features=self.input_dim)
        test_pred, values = self.opt.evaluate(self.test_loader_one, batch_size=1, n_features=self.input_dim)

        self.train_preds = np.array(train_pred).flatten()
        self.test_preds = np.array(test_pred).flatten()

#fig = midas.plotPredictions(features, data.X_test, vals, preds)
#        fig = midas.plotPredictions(self.data.trainingFeatures, self.data.X_train, self.vals, self.preds)
#        plt.tight_layout()
#        fig.savefig('Figures/MidasPredictions.pdf', bbox_inches='tight')
#plt.show()
        return self.train_preds, self.test_preds


def run_clustering(model, x_test,y_pred, dataset, features, discrete_features, search_num, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold, preload_explainer=False):

    print('Starting thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)
    GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_pred, features=features, discrete_features=discrete_features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=preload_explainer)
    if preload_explainer:
        GLE.plot_all_clustering()
    else:
        GLE.multi_layer_clustering(search_num)
    print('finishing thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)




def chilli_explain(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=25, automated_locality=False, newMethod=True, kernel_width=None, noisey_instance=None, categorical_features=None, neighbours=None):
    chilliExplainer = CHILLI('MIDAS', model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, automated_locality=automated_locality, newMethod=newMethod)
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction = chilliExplainer.make_explanation(model, explainer, instance=instance, num_samples=1000)
#    chilliExplainer.plot_explanation(35, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL')
    chilliExplainer.interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'RUL', neighbours=neighbours)
    with open(f'saved/explanation_{instance}.pck', 'wb') as file:
        pck.dump([exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error], file)
    plotting_data = [x_test[instance], instance, perturbations, model_perturbation_predictions, y_test[instance], y_train[instance], exp_instance_prediction, exp_perturbation_predictions, exp]

    return explanation_error, instance_prediction, exp_instance_prediction, plotting_data

if __name__ == '__main__':
    mode = sys.argv[1]

    dataset = 'MIDAS'
    midas_runner = MIDAS(load_model=True)
    x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures

    discrete_features = ['heathrow cld_ttl_amt_id']
    categorical_features = [features.index(feature) for feature in discrete_features]

    model = midas_runner.train_midas_rnn()
    y_train_pred, y_test_pred = midas_runner.make_midas_predictions()

    sampling=False
    if sampling:
        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), 1000)
        x_train = x_train[random_samples]
        y_train = y_train[random_samples]
        y_train_pred = y_train_pred[random_samples]

        x_test = x_test[random_samples]
        y_test = y_test[random_samples]
        y_test_pred = y_test_pred[random_samples]

    # -------- PARAMETER SEARCH -------- #
#
#        parameter_search_list = []
#        for sparsity_threshold in parameter_search['sparsity']:
#            for coverage_threshold in parameter_search['coverage']:
#                for starting_k in parameter_search['starting_k']:
#                    for neighbourhood_threshold in parameter_search['neighbourhood']:
#                        parameter_search_list.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
#        num_nodes = 4
#        tasks_per_node = len(parameter_search_list)/num_nodes
##        parameter_search_list = parameter_search_list[sys.argv[1]-1*tasks_per_node:sys.argv[1]*tasks_per_node]

    # ---------------------------------- #

    # Write list of LLC parameters here
    # [sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold]
    parameter_search_list = [[0.05, 0.05, 10, 0.05]]
#        parameter_search_list = [[0.5, 0.05, 5, 0.5]]

    for params in parameter_search_list:
        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = params

        if mode == 'ensembles':

            # If doing parameter search then use multiprocessing

#                process = multiprocessing.Process(target=run_clustering, args=(model, x_test, y_pred, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold))
#                process.start()

            run_clustering(model, x_train, y_train_pred, dataset, features, discrete_features,  parameter_search_list.index(params), sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)

        elif mode=='explain':
            x_test = x_train
            y_test = y_train
            y_test_pred = y_train_pred

            GLE = GlobalLinearExplainer(model=model, x_test=x_test, y_pred=y_test_pred, features=features, discrete_features=discrete_features, dataset='MIDAS', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)


            # ---- Similar Instances ----
#            instances24matches = [3325, 68, 81, 92, 206, 253, 254, 302, 350, 425, 499, 621, 660, 669, 786, 846, 962, 975, 1011, 1012, 1076, 1339, 1584, 1604, 1614, 1636, 1656, 1677, 1930, 2131, 2132, 2246, 2295, 2335, 2343, 2723, 2997, 3116, 3146, 3156, 3176, 3286, 3297, 3325, 3413, 3419, 3421, 3495, 3509, 3523, 3566, 3581, 3582, 3589, 3642, 3817, 3958, 3992, 4141, 4165, 4247, 4282, 4285, 4314, 4334, 4354, 4397, 4409, 4417, 4668, 4693, 4735, 4851, 4953, 498][:10]
#            instances23matches = [3325, 1, 3, 29, 51, 55, 67, 68, 81, 83, 92, 94, 111, 117, 153, 178, 179, 192, 198]

            # ---- Same Instances ----
#            instances = [100 for i in range(10)]


            #  ---- Random Instances ----
#            instances = [random.randint(0, len(x_test)) for i in range(1)]
            instances = [3870]
            instance = instances[0]

            llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_test_pred[instance], y_test[instance])

            data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data

            relevant_instances = GLE.new_evaluation(instance, instance_explanation_model, x_test[instance])
            instances = relevant_instances[:10]

#            distances = combinedFeatureDistances(calcAllDistances(x_test[instance], x_test, features))
##                    distances = [math.dist(x_test[instance], x) for x in x_test]
#
#            print('Min Distance: ', min(distances))
#            closest_instances = np.argsort(distances)[:100]
#            instances = closest_instances

#            print(list(closest_instances))
#            instances = [1757]

#                instances = closest_instances_indices
#                if os.path.exists(f'saved/feature_ensembles/PHM08_feature_ensembles_full_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck'):

#                    chilli_predictions = []
            kernel_widths = [0.01, 0.1, 0.5, 1, 5, 10]
#                    chilli_predictions = {kernel_width: [] for kernel_width in kernel_widths}
#                for k in kernel_widths:
            model_predictions = []
            lime_predictions = []
            lime_explanations = []
            chilli_predictions = []
            chilli_explanations = []
            llc_predictions = []
            llc_explanations = []
            similar_explanation_data = []
            chilli_exp = True
            lime_exp = True
            chilli_deviations = []
            llc_deviations = []



            for instance in tqdm(instances):
                    GLE.plot_all_clustering(instance=instance)
                    print(f'################# Instance  = {instance} ###################')

                    # ------ BASE MODEL ------
                    print(f'Ground Truth: {y_test[instance]}')
                    print(f'Model Prediction: {y_test_pred[instance]}')
                    # Add noise to instance
#                        print(f'Noisey model prediction: {noisey_model_prediction}')
#                        model_predictions.append([y_pred[instance], noisey_model_prediction])
                    model_predictions.append(y_test_pred[instance])

                    # ---- LIME EXPLANATION -------
                    if chilli_exp:

                        print('\n ----- LIME EXPLANATION -----')
                        _,_, lime_prediction, lime_plotting_data = chilli_explain(midas_runner.opt.predictor_from_numpy, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred,features, instance=instance, newMethod=False, kernel_width=2, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, lime_exp = lime_plotting_data
                        lime_predictions.append(lime_prediction)
                        lime_explanations.append(lime_exp.as_list())

                    # ---- CHILLI EXPLANATION -------
                    if chilli_exp:
                        print('\n ----- CHILLI EXPLANATION -----')
                        _,_, chilli_prediction, chilli_plotting_data = chilli_explain(midas_runner.opt.predictor_from_numpy, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=instance, newMethod=True, kernel_width=0.1, categorical_features=categorical_features)

                        instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, chilli_exp = chilli_plotting_data

                        chilli_predictions.append(chilli_prediction)
                        chilli_explanations.append(chilli_exp.as_list())

                    # ---- LLC EXPLANATION -------
                    print('\n ----- LLC EXPLANATION -----')
                    llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_test_pred[instance], y_test[instance])

                    data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data
                    llc_predictions.append(llc_prediction)
                    llc_explanations.append(instance_explanation_model.coef_)

                    GLE.interactive_exp_plot(data_instance, instance_index, instance_prediction, y_test_pred, exp_instance_prediction, instance_explanation_model.coef_, local_x, local_x_weights, local_y_pred, exp_local_y_pred, 'heathrow air_temperature')

                    if chilli_exp and lime_exp:
                        similar_explanation_data.append([lime_exp.as_list(), chilli_exp.as_list(), instance_explanation_model.coef_])

            with open(f'saved/results/{dataset}_similar_instances.pck', 'wb') as f:
                pck.dump([[lime_predictions, lime_explanations], [chilli_predictions, chilli_explanations], [llc_predictions, llc_explanations], instances, model_predictions ], f)

#                with open('saved/results/similar_instances24matches.pck', 'wb') as f:
            with open(f'saved/results/{dataset}_same_instances.pck', 'wb') as f:
                pck.dump(similar_explanation_data, f)





#


