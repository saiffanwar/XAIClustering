import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from  sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular as lt
from chilli import CHILLI

'''
The task is to prediction the attrition of the employees in the company.
'''

def data_preprocessing():
    data = pd.read_csv('Data/ibmhr.csv')
    data.columns.to_series().groupby(data.dtypes).groups

    # Removing columns that don't provide any information.'
    data = data.drop(columns = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'])

    # tranform binary feature into 0 and 1
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})


    y = data['Attrition']
    xdata = data.drop(columns = ['Attrition'])
    xdata = pd.get_dummies(xdata, columns = ['BusinessTravel', 'Gender','MaritalStatus'], drop_first = True)
    xdata = pd.get_dummies(xdata)

    col_tobe_standard = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction','HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

#    scaler = StandardScaler()
#    for col in col_tobe_standard:
#        xdata[col] = xdata[col].astype(float)
#        xdata[[col]] = scaler.fit_transform(xdata[[col]])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(xdata, y, test_size = 0.25, random_state=40, stratify=y)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = data_preprocessing()

def plot_data(x, y):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    axes.scatter(x, y, color='blue', alpha=0.5)
    fig.savefig('Figures/IBM/ageY.pdf')


# Fitting model and checking accuracy.
model = XGBClassifier(subsample = 0.8, reg_lambda = 0.5, reg_alpha = 0.05, n_estimators = 80, max_depth = 1, learning_rate = 0.5, eta = 0.6)
model.fit(np.array(x_train.values), np.array(y_train.values))
y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:,0]
plot_data(x_test['OverTime'], y_pred_prob)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Explaining the model's predictions using lime


#explainer = lt.LimeTabularExplainer(x_train.values, feature_names = x_train.columns, class_names = ['No', 'Yes'])
#
#instance = 35
#
#exp = explainer.explain_instance(x_test.values[instance], model.predict_proba, num_features = 8, top_labels = 1)
#
#exp.as_pyplot_figure(0)
#exp.save_to_file('Figures/IBM/IBM_lime.html')

chilli = CHILLI(model, x_train, y_train, x_test, y_test)
explainer = chilli.build_explainer()
instance = 85
exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error = chilli.make_explanation(explainer, instance=instance)
chilli.plot_explanation(instance, exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature='Attrition')


