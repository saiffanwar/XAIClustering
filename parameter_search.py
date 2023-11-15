import numpy as np
import pickle as pck
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from run_linear_clustering import GlobalLinearExplainer
from rul_phm08 import data_preprocessing, train, evaluate

#parameter_search = {
#                    'sparsity': [0, 0.05, 0.1, 0.25, 0.5, 1],
#                    'coverage': [0, 0.05, 0.1, 0.25, 0.5, 1],
#                    'starting_k': [1,5,10,20],
#                    'neighbourhood': [0.01, 0.05, 0.1, 0.25, 0.5, 1],
#                    }
parameter_search = {
                    'sparsity': [0, 0.05, 0.1, 0.25, 0.5, 1],
                    'coverage': [0, 0.05, 0.1, 0.25, 0.5, 1],
                    'starting_k': [1,5,10,20],
                    'neighbourhood': [0.01, 0.05, 0.1, 0.25, 0.5, 1],
                    }

x_train, x_test, y_train, y_test, features = data_preprocessing()

discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
with open(f'saved/PHM08_model.pck', 'rb') as file:
    model = pck.load(file)
y_pred = evaluate(model, x_test, y_test)

R = np.random.RandomState(42)
random_samples = R.randint(2, len(x_test), 2500)
x_test = x_test[random_samples]
y_pred = y_pred[random_samples]
y_test = y_test[random_samples]

results = []
for sparsity_threshold in parameter_search['sparsity']:
    for coverage_threshold in parameter_search['coverage']:
        for starting_k in parameter_search['starting_k']:
            for neighbourhood_threshold in parameter_search['neighbourhood']:
#                try:
                    with open(f'saved/PHM08_results_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck', 'rb') as file:
                        model_predictions, chilli_predictions, llc_predictions = pck.load(file)
                        mse = mean_squared_error(model_predictions, llc_predictions)
                        results.append([starting_k, neighbourhood_threshold, mse])
                    GLE = GlobalLinearExplainer(model, x_test, y_pred, features, 'PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True)
                    GLE.plot_all_clustering()
#                except:
#                    pass
results = np.array(results)
