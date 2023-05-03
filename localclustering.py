#!/usr/bin/env python
# coding: utf-8

# In[2]:


from LocalLinearRegression import LocalLinearRegression
import pickle as pck
import numpy as np
import random
import matplotlib.pyplot as plt


# Contextual Distance Functions

NumSamples = 5000

# MIDAS DATA:

# with open('allMIDASdata.pck', 'rb') as file:
#     data = pck.load(file)

# xdata,ydata = data[0], data[1]

# feature = 8
# xdata, ydata = xdata[:,feature], ydata
# xdata,ydata = zip(*random.sample(list(zip(xdata, ydata)), NumSamples))


# # WebTRIS DATA:
with open('WebtrisData.pck', 'rb') as file:
    data = pck.load(file)[0]

xdata, ydata = data['Time Interval'].values[:NumSamples], data['Total Volume'].values[:NumSamples]

def cyclic(x1, x2, possValues=np.arange(0,96,1)):
    # x1, x2 = [x*len(possValues) for x in [x1,x2]]
    
    diff = abs(x1-x2)
    return min(len(possValues) - diff, diff)

maxVal = max(xdata)
def euclidean(x1, x2):
    return abs(x1-x2)/maxVal

distanceFunction = cyclic

LLR = LocalLinearRegression(xdata,ydata)

w1, w2, w, MSE = LLR.calculateLocalModels(distanceFunction)
D, xDs= LLR.computeDistanceMatrix(w1, w2, w, MSE, distanceFunction)



# In[3]:


K = 6
clusteredData = LLR.KMedoidClustering(K, D)
for cluster in clusteredData:
    print(min(cluster[0]), max(cluster[0]))

linearParams = LLR.LinearModelsToClusters(clusteredData)


# In[4]:


fig, axes = plt.subplots(1,1,figsize=(10,10))
axes = fig.axes
colours = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'brown', 'grey', 'cyan', 'magenta']
for i in range(len(clusteredData)):
    w,b = linearParams[i]
    colour = np.random.rand(1,3)
    colour = colours[i]
    print
    axes[0].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data', c=colour)
    axes[0].plot(np.array(clusteredData[i][0]).flatten(), np.array([w*j+b for j in clusteredData[i][0]]).flatten(), c=colour, linewidth=5)
    # axes[i+1].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data', c=colour)
    # axes[i+1].plot(np.array(clusteredData[i][0]).flatten(), np.array([w*j+b for j in clusteredData[i][0]]).flatten(), c=colour, linewidth=5)
    # axes[i+1].set_xlim(0,1)   
    # axes[i+1].set_ylim(0,1)

    
fig.savefig('WebTRISdata'+str(K)+'Medoids.pdf')


# In[14]:


sin_time = np.sin(2*np.pi*xdata/96)
cos_time = np.cos(2*np.pi*xdata/96)

plt.scatter(np.arange(0,5000,1),sin_time+cos_time)
# plt.show()
# plt.plot(np.arange(0,5000,1), cos_time, 'o')
# plt.show()
# plt.plot(sin_time, cos_time, 'o')


# In[8]:


# Plot using plotly
import plotly
import plotly.graph_objs as go

# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
# trace1 = go.Scatter3d(
#     x=sin_time,  # <-- Put your data instead
#     y=cos_time,  # <-- Put your data instead
#     z=ydata,  # <-- Put your data instead
#     mode='markers',
#     marker={
#         'size': 2,
#         'opacity': 0.8,
#     }
# )

data = []
for i in range(len(clusteredData)):
    w,b = linearParams[i]
    colour = np.random.rand(1,3)
    colour = colours[i]
    
    sin_val, cos_val = np.sin(2*np.pi*np.array(clusteredData[i][0])/96), np.cos(2*np.pi*np.array(clusteredData[i][0])/96)
   
    # Configure the trace.
    trace1 = go.Scatter3d(
        x=sin_val, 
        y=cos_val, 
        z=clusteredData[i][1],  
        mode='markers',
        marker={
            'size': 2,
            'opacity': 0.8,
            'color': colour
        }
    )
    if i != 0:
        lindata = np.array([w*j+b for j in clusteredData[i][0]]).flatten()
    else:
        print(i)
        lindata = np.array([w*j+b for j in clusteredData[i][0]]).flatten()
    trace2 = go.Scatter3d(
        x=sin_val, 
        y=cos_val, 
        # z=np.array([w*j+b for j in clusteredData[i][0]]).flatten(), 
        z=lindata, 
        mode='markers',
        marker={'size': 12, 'color': colour}
        # marker={
        #     'size': 2,
        #     'opacity': 0.8,
        # }
    ) 

    
    # axes[0].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data', c=colour)
    # axes[0].plot(np.array(clusteredData[i][0]).flatten(), np.array([w*j+b for j in clusteredData[i][0]]).flatten(), c=colour, linewidth=5)

    data.append(trace1)
    data.append(trace2)
#Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

# data = [trace1, trace2]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)

