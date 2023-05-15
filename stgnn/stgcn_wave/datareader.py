import pandas as pd
import numpy as np
import h5py
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
token = open(".mapboxtoken").read() # you will need your own token

filename = 'data/metr-la.h5'
df = pd.read_hdf(filename)

locationdata = pd.read_csv('data/sensor_graph/graph_sensor_locations.csv')

fig=px.scatter_mapbox(locationdata, lat ='latitude', lon='longitude', hover_name = 'sensor_id')
fig.update_layout(mapbox_style='light', mapbox_accesstoken=token)
fig.show()


fig, axes = plt.subplots(1,1, figsize=(10,10))
axes.scatter(df.index.tolist(), df['773869'])
fig.savefig('Figures/singleMTER-LAsisingleMTER-LAsite.png')
