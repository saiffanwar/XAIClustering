import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, callback_context
from run_linear_clustering import GlobalLinearExplainer
from rul_phm08 import data_preprocessing, train, evaluate
import pickle as pck
import numpy as np


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
GLE = GlobalLinearExplainer(model, x_test, y_pred, features, 'PHM08', preload_explainer=True)
#instance=100
#llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
#data_instance, instance_index, local_x, local_y_pred, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data



app = Dash(__name__)


app.layout = html.Div(
    children=[
        html.H1(children="Local Linear Explanations for the PHM08 dataset"),
        html.P(
            children=(
            ),
        ),
            html.Div(id='top-bar', children=[
            html.Div(id='feature-bar', children=[
            html.H2(children="Features to display: "),
            dcc.Dropdown(id='feature-selection',
                options=GLE.features,
                value=GLE.features[:3],
                multi=True,
                         clearable=False, style={'width':'500px'}
            ),
            html.Div(id='feature-buttons', children=[
            html.Button('Deselect All Features', id='no-features', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '0px'}),
            html.Button('Select All Features', id='all-features', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px'}),
            html.Button('Show All Clusters', id='show-clustering', n_clicks=0, style={'display': 'inline-block', 'horizontal-align': 'left', 'vertical-align': 'middle', 'margin-left': '10px'}),
            html.Button('Reset Plot', id='reset-button', n_clicks=0, style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px'})
            ], style=dict(margin='10px 0'))
            ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px'}),
            dcc.Graph(id='explanation-plot', figure=GLE.plot_explanation(), style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '10px'})
            ]),
        dcc.Graph(id='data-plot',
            figure=GLE.plot_data(features_to_plot = GLE.features[:-18]),
        ),
    ]
)
@app.callback(Output('data-plot','figure'),
            Output('feature-selection', 'value'),
            Output('data-plot', 'clickData'),
            Input('data-plot', 'clickData'),
            Input('feature-selection', 'value'),
            Input('reset-button', 'n_clicks'),
            Input('all-features', 'n_clicks'),
            Input('no-features', 'n_clicks'),
            Input('show-clustering', 'n_clicks'))

def update_data_plot(click_data, features_to_plot, reset_button, all_features_button, no_features_button, show_clustering_button):

    if callback_context.triggered_id == 'all-features':
        features_to_plot = GLE.features
        if click_data != None:
            instance = click_data['points'][0]['pointIndex']
            llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
            fig =  GLE.plot_data(plotting_data, features_to_plot = features_to_plot)
        else:
            fig = GLE.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data
    elif callback_context.triggered_id == 'no-features':
        features_to_plot = ['cycle']
        if click_data != None:
            instance = click_data['points'][0]['pointIndex']
            llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
            fig =  GLE.plot_data(plotting_data, features_to_plot = features_to_plot)
        else:
            fig = GLE.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data
    elif callback_context.triggered_id == 'show-clustering':
        print('clsutering button')
        fig = GLE.plot_all_clustering(features_to_plot)
        return fig, features_to_plot, None
    elif click_data == None or callback_context.triggered_id == 'reset-button':
        fig = GLE.plot_data(features_to_plot = features_to_plot)
        return fig, features_to_plot, None
    else:
        instance = click_data['points'][0]['pointIndex']
        llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
        fig =  GLE.plot_data(plotting_data, features_to_plot = features_to_plot)
        return fig, features_to_plot, click_data

@app.callback(Output('explanation-plot','figure'),
                Input('data-plot', 'clickData'),
                Input('feature-selection', 'value'),
                Input('reset-button', 'n_clicks'))

def update_explanation_plot(click_data, features_to_plot, reset_button):
    if click_data == None or callback_context.triggered_id == 'reset-button':
        fig = GLE.plot_explanation()
        return fig
    else:
        instance = click_data['points'][0]['pointIndex']
        llc_prediction, plotting_data = GLE.generate_explanation(x_test[instance], instance, y_pred[instance], y_test[instance])
        fig =  GLE.plot_explanation(plotting_data, features_to_plot = features_to_plot)
        return fig

if __name__ == "__main__":
    app.run_server(debug=True)
