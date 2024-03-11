import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, callback_context
#from run_linear_clustering import LLCExplanation
from run_llc_explanation import LLCExplanation
from generate_ensembles import LLCGenerator
from rul_phm08 import data_preprocessing, train, evaluate, chilli_explain
import midas_model_data as midas
from results_midas import MIDAS
import pickle as pck
import numpy as np


dataset = 'MIDAS'
if dataset == 'PHM08':
    x_train, x_test, y_train, y_test, features = data_preprocessing()
    discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    with open(f'saved/models/{dataset}_model.pck', 'rb') as file:
        model = pck.load(file)
elif dataset == 'MIDAS':
    midas_runner = MIDAS(load_model=True)
    x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures
    discrete_features = ['heathrow cld_ttl_amt_id']
    categorical_features = [features.index(feature) for feature in discrete_features]
    model = midas_runner.train_midas_rnn()

sampling = False
if sampling:
    R = np.random.RandomState(42)
    random_samples = R.randint(2, len(x_test), 5000)

    x_train = x_train[random_samples]
    y_train = y_train[random_samples]

    x_test = x_test[random_samples]
    y_test = y_test[random_samples]
if dataset == 'PHM08':
    y_train_pred, y_test_pred = evaluate(model, x_train, x_test, y_train, y_test)
    y_pred = y_test_pred
elif dataset == 'MIDAS':
    y_train_pred, y_test_pred = midas_runner.make_midas_predictions()
    if sampling:
        y_pred = y_test_pred[random_samples]
        y_pred = y_pred[random_samples]
    y_pred = y_train_pred
    x_test = x_train
    y_test = y_train

chilli_prediction = None

sparsity_threshold = 0.05
coverage_threshold = 0.05
starting_k = 10
neighbourhood_threshold = 0.05

LLCExp = LLCExplanation(model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)
LLCGen = LLCGenerator(model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)
#instance=100
#llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
#data_instance, instance_index, local_x, local_y_pred, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data



app = Dash(__name__)


app.layout = html.Div(
    children=[
        html.H1(children=f"Local Linear Explanations for the {dataset} dataset"),
        html.P(
            children=(
            ),
        ),
            html.Div(id='top-bar', children=[
            html.Div(id='feature-bar', children=[
            html.H2(children="Features to display: "),
            dcc.Dropdown(id='feature-selection',
                options=LLCGen.features,
                value=LLCGen.features,
                multi=True,
                         clearable=False, style={'width':'500px'}
            ),
            html.Div(id='feature-buttons', children=[
            html.Button('Deselect All Features', id='no-features', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '0px'}),
            html.Button('Select All Features', id='all-features', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px'}),
            html.Button('Show All Clusters', id='show-clustering', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px'}),
            html.Button('Reset Plot', id='reset-button', n_clicks=0, style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px'})
            ], style=dict(margin='10px 0')),
            html.Div(id='instances', children=[dcc.Input(id='instance-selection', type='number', value=None, style={'width': '100px', 'margin-left': '0px'}),
                                            html.Button('Show Instance', id='instance-plot', n_clicks=0, style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px'}),
                                            dcc.Input(id='other-instances', type='text', value='None', style={'width': '100px', 'margin-left': '10px'})
], style={'margin': '10px 0'}),
            html.Div(id='thresolds-selction', children=[
            html.Div(children=[html.Label('Sparsity Threshold: '), dcc.Dropdown(id='sparsity-threshold', options=[0,0.01,0.05,0.1,0.25,0.5,1], value=sparsity_threshold, multi=False, clearable=False,style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '0px', 'width': '60px'})], style={'margin': '10px 0'}),
            html.Div(children=[html.Label('Coverage Threshold: '), dcc.Dropdown(id='coverage-threshold', options=[0,0.01,0.05,0.1,0.25,0.5,1], value=coverage_threshold, multi=False, clearable=False,  style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px', 'width': '60px'})], style={'margin': '10px 0'}),
            html.Div(children=[html.Label('Starting K:  '), dcc.Dropdown(id='starting-k', options=[1,5,10,20,50], value=starting_k, multi=False, clearable=False, style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px', 'width': '60px'})], style={'margin': '10px 0'}),
            html.Div(children=[html.Label('Neighbourhood Threshold:  '), dcc.Dropdown(id='neighbourhood-threshold', options=[0,0.01,0.05,0.1,0.25,0.5,1], value=neighbourhood_threshold, multi=False, clearable=False, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px', 'width': '60px'})], style={'margin': '10px 0'}),
            html.Div(children=[html.Label('CHILLI Prediction: '), html.Label(children=[chilli_prediction], id='chilli-prediction')])
            ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '0px'}),
            ]),
            dcc.Graph(id='explanation-plot', figure=LLCExp.plot_explanation(), style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '10px'})
            ]),
        dcc.Graph(id='data-plot',
            figure=LLCGen.plot_data(features_to_plot = LLCGen.features),
        ),
    ]
)
@app.callback(Output('data-plot','figure'),
            Output('feature-selection', 'value'),
            Output('data-plot', 'clickData'),
            Input('sparsity-threshold', 'value'),
            Input('coverage-threshold', 'value'),
            Input('starting-k', 'value'),
            Input('neighbourhood-threshold', 'value'),
            Input('data-plot', 'clickData'),
            Input('feature-selection', 'value'),
            Input('instance-plot', 'n_clicks'),
            Input('instance-selection', 'value'),
            Input('other-instances', 'value'),
            Input('reset-button', 'n_clicks'),
            Input('all-features', 'n_clicks'),
            Input('no-features', 'n_clicks'),
            Input('show-clustering', 'n_clicks'))

def update_data_plot(sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold, click_data, features_to_plot, instance_plot, instance, other_instances, reset_button, all_features_button, no_features_button, show_clustering_button):

#    GLE = LLCExplanation(model, x_test, y_pred, features, 'PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)
    LLCExp = LLCExplanation(model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)
#    if callback_context.triggered_id in ['sparsity_threshold', 'coverage_threshold', 'starting_k', 'neighbourhood_threshold']:
#        GLE = LLCExplanation(model, x_test, y_pred, features, 'PHM08', sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True)

    if callback_context.triggered_id == 'all-features':
        features_to_plot = LLCGen.features
        if click_data != None:
            instance = click_data['points'][0]['pointIndex']
            print(x_test[instance])
            llc_prediction, plotting_data, matched_instances = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
            fig =  LLCGen.plot_data(plotting_data, features_to_plot = features_to_plot)
        else:
            fig = LLCGen.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data
    elif callback_context.triggered_id == 'no-features':
        features_to_plot = ['cycle']
        if click_data != None:
            instance = click_data['points'][0]['pointIndex']
            llc_prediction, plotting_data = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
            fig =  LLCGen.plot_data(plotting_data, features_to_plot = features_to_plot)
        else:
            fig = LLCGen.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data
    elif callback_context.triggered_id == 'show-clustering':
        if other_instances == 'None':
            instances_to_show = None
        else:
            instances_to_show = [int(num) for num in other_instances.split(',')]
        fig = LLCGen.plot_all_clustering(features_to_plot=features_to_plot, instances_to_show = instances_to_show)
        return fig, features_to_plot, None
    elif callback_context.triggered_id == 'instance-plot':
        if other_instances == 'None':
            instances_to_show = []
        else:
            instances_to_show = [int(num) for num in other_instances.split(',')]
        print(x_test[instance])
        llc_prediction, plotting_data, matched_instances = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
        fig =  LLCGen.plot_data(plotting_data, features_to_plot = features_to_plot, instances_to_show = instances_to_show)
        return fig, features_to_plot, click_data
    elif click_data == None or callback_context.triggered_id == 'reset-button':
        fig = LLCGen.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, None
    else:
        instance = click_data['points'][0]['pointIndex']
        llc_prediction, plotting_data, matched_instances = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
        fig =  LLCGen.plot_data(plotting_data, features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data

@app.callback(Output('explanation-plot','figure'),
              Output('chilli-prediction', 'children'),
                Input('data-plot', 'clickData'),
                Input('feature-selection', 'value'),
                Input('instance-selection', 'value'),
                Input('instance-plot', 'n_clicks'),
                Input('show-clustering', 'n_clicks'),
                Input('reset-button', 'n_clicks'))

def update_explanation_plot(click_data, features_to_plot, instance, instance_button, show_clustering_button, reset_button):
    if callback_context.triggered_id == 'instance-plot' or callback_context.triggered_id == 'show-clustering':
        if instance != None:
            llc_prediction, plotting_data, matched_instances = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
            fig =  LLCExp.plot_explanation(plotting_data, features_to_plot = features_to_plot)
#        _,_, chilli_prediction = chilli_explain(model, x_train, y_train, x_test, y_test, features, instance=instance, kernel_width=10)
            chilli_prediction = 0
            return fig, [chilli_prediction]
        else:
            fig = LLCExp.plot_explanation()
            chilli_prediction = 0
            return fig, [chilli_prediction]


    elif click_data == None or callback_context.triggered_id == 'reset-button':
        fig = LLCExp.plot_explanation()
        chilli_prediction = 0
        return fig, [chilli_prediction]
    else:
        instance = click_data['points'][0]['pointIndex']
        llc_prediction, plotting_data, matched_instances = LLCExp.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
        fig =  LLCExp.plot_explanation(plotting_data, features_to_plot = features_to_plot)
#        _,_, chilli_prediction = chilli_explain(model, x_train, y_train, x_test, y_test, features, instance=instance, kernel_width=10)
        chilli_prediction = 0
        return fig, [chilli_prediction]

if __name__ == "__main__":
    app.run_server(debug=True)
