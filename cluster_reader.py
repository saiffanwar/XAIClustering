import pickle as pck
import numpy as np


with open('saved/feature_ensembles/MIDAS_feature_ensembles_full_0.05_0.1_10_0.1.pck', 'rb') as f:
#with open('saved/feature_ensembles/PHM08_feature_ensembles_full_0.05_0.1_10_0.1.pck', 'rb') as f:
    data = pck.load(f)

features_ensemble = data

print(features_ensemble['heathrow wind_speed'])
