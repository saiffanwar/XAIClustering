import pickle as pck
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plotly
from pprint import pprint

exp =  pck.load(open('saved/explanation.pck', 'rb'))

exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions = exp[0], exp[1], exp[2], exp[3]
instance = 35
targetFeature = 'RUL'



def interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature):

    exp_list = exp.as_list()
    explained_features = [i[0] for i in exp_list][:12]
    all_features = ['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
#    all_features = self.features

    explained_feature_indices = [all_features.index(i) for i in explained_features]
    explained_feature_perturbations = np.array(perturbations)[:,explained_feature_indices]
    perturbations = explained_feature_perturbations
    feature_contributions = [i[1] for i in exp_list][:12]



    fig = make_subplots(rows=5, cols=3, column_widths=[0.33, 0.33, 0.33], row_heights =[0.33, 0.16, 0.16, 0.16, 0.16],
                        specs = [
                            [{},{'colspan':2},None],
                            [{}, {}, {}],
                            [{}, {}, {}],
                            [{}, {}, {}],
                            [{}, {}, {}],
                            ], subplot_titles=['', 'Feature Contributions']+explained_features,
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

    fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=2)


    instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), np.array(model_perturbation_predictions[0]), np.array(exp_perturbation_predictions[0])
    perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
    perturbations_model_y = [int(i) for i in perturbations_model_y]
    explanation_error = mean_squared_error(perturbations_model_y, perturbations_exp_y)



    axes = [[row, col] for row in range(2,6) for col in range(1,4)]


    for i, ax in enumerate(axes):
#        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='orange', size=3)), row=ax[0], col=ax[1])
        pprint(list(zip(perturbations_model_y, perturbations_exp_y)))
        if i==0:
            showlegend=True
        else:
            showlegend=False
        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_model_y, mode='markers', marker = dict(color='orange', size=3, opacity=0.9), showlegend=showlegend, name='Model Perturbation Predictions'), row=ax[0], col=ax[1])
        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='green', size=3, opacity=0.9), showlegend=showlegend, name='Explanation Perturbation Predictions'), row=ax[0], col=ax[1])
#        fig.add_trace(go.Scatter(x=[instance_x[i]],y=instance_model_y, mode='markers', marker = dict(color='red', size=20) ), row=ax[0], col=ax[1])
    fig.update_layout(title_text=f'Explanation for instance {instance} \n Explanation Error = {explanation_error:.2f}',
                      font=dict(size=18),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),)
    fig.write_html(f'Figures/PHM08/Explanations/instance_{instance}_explanation.html', auto_open=True)

interactive_perturbation_plot(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature)
