from LocalLinearRegression import LocalLinearRegression
import pickle as pck
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


NumSamples = 500

def fetchData(dataset='webtris'):


    def cyclic(x1, x2, possValues=np.arange(0,96,1)):
        diff = abs(x1-x2)
        return min(len(possValues) - diff, diff)
    def euclideanDefine(xdata):
        maxVal = max(xdata)
        euclidean = lambda x1,x2: abs(x1-x2)/maxVal
        return euclidean

    # WebTRIS DATA:
    if dataset == 'webtris':
        with open('WebtrisData.pck', 'rb') as file:
            data = pck.load(file)[0]
        xdata, ydata = data['Time Interval'].values[:NumSamples], data['Total Volume'].values[:NumSamples]
        distFunction = cyclic

    # MIDAS Data
    elif dataset == 'midas':
        with open('allMIDASdata.pck', 'rb') as file:
             data = pck.load(file)
        xdata,ydata = data[0], data[1]
        featureNum = 8
        xdata, ydata = xdata[:,featureNum], ydata
        xdata,ydata = zip(*random.sample(list(zip(xdata, ydata)), NumSamples))

        distFunction = euclideanDefine(xdata)

    return xdata, ydata, distFunction


#if __name__ == "__main__":

dataset = 'webtris'
print('Fethcing Data')
xdata, ydata, distFunction = fetchData(dataset)

print('Performing Local Linear Regression')
# Perform LocalLinear regression on fetched data
LLR = LocalLinearRegression(xdata,ydata)
w1, w2, w, MSE = LLR.calculateLocalModels(distFunction)

print('Calculating Distances')
# Use local models to compute distance matrix for all points (slow)
D, xDs= LLR.computeDistanceMatrix(w1, w2, w, MSE, distFunction)

print('Doing K-medoids-clustering')
# Define number of medoids and perform K medoid clustering.
K = 6
clusteredData = LLR.KMedoidClustering(K, D)

linearParams = LLR.LinearModelsToClusters(clusteredData)




print('Starting Plotting')
fig, axes = plt.subplots(1,1,figsize=(10,10))
axes = fig.axes
colours = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'brown', 'grey', 'cyan', 'magenta']
for i in range(len(clusteredData)):
    w,b = linearParams[i]
    colour = np.random.rand(1,3)
    colour = colours[i]
    axes[0].scatter(clusteredData[i][0], clusteredData[i][1], s=5, marker='o', label='data', c=colour)
    axes[0].plot(np.array(clusteredData[i][0]).flatten(), np.array([w*j+b for j in clusteredData[i][0]]).flatten(), c=colour, linewidth=5)

fig.savefig(dataset+'data'+str(K)+'Medoids.pdf')



sin_time = np.sin(2*np.pi*xdata/96)
cos_time = np.cos(2*np.pi*xdata/96)

timeFig, timeaxes = plt.subplots(1,1,figsize=(5,5))
timeaxes.scatter(sin_time,cos_time)
timeFig.savefig('timeData.png')

# Plot using plotly

data = []
print(len(clusteredData))
for i in range(len(clusteredData)):
    w,b = linearParams[i]
    colour = np.random.rand(1,3)
    colour = colours[i]
    print(colour)
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

#data = [trace1, trace2]

plot_figure = go.Figure(data=data, layout=layout)
# Render the plot.
plotly.offline.iplot(plot_figure)

