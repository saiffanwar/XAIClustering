import numpy as np
import math
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pickle as pck
from sklearn.linear_model import LinearRegression
import random
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#rc('text',usetex=True)

# rc('font',**{'family':'serif','serif':['Times']})
# plt.rcParams['text.usetex'] = True

NumSamples=5000

def cyclic(x1, x2, possValues=96):
    diff = abs(x1-x2)
    return min(possValues - diff, diff)


def fetchData():
    with open('WebtrisData.pck', 'rb') as file:
        data = pck.load(file)[0]
    xdata, ydata = data['Time Interval'].values[:NumSamples], data['Total Volume'].values[:NumSamples]
    distFunction = cyclic

#    return xdata, ydata, distFunction
    reducedxs = []
    idxs = []
    [(reducedxs.append(int(x)),idxs.append(int(idx))) for idx, x in enumerate(xdata) if ((x>=80)or(x<=15))]
    reducedys = ydata[idxs]
    return reducedxs, reducedys, distFunction



# Calculating sin and cos components of time interval.
def  calcComponents(xdata):
    sin_time = [np.sin(2*np.pi*x/96) for x in xdata]
    cos_time = [np.cos(2*np.pi*x/96) for x in xdata]
    return sin_time, cos_time



def LinReg(featureData, ydata):
    # Linear Regression using sklearn
    if len(featureData) < 1:
        transformed_data = np.array([feature[i] for i in featureData]).T
    else:
        transformed_data = np.array(featureData[0]).reshape(-1,1)

    xtrain, xtest, ytrain, ytest = transformed_data[:int(0.8*NumSamples)], transformed_data[int(0.2*NumSamples):], ydata[:int(0.8*NumSamples)], ydata[int(0.2*NumSamples):]


    LR = LinearRegression()
#    sintransform = np.array(sin_time).reshape(-1,1)
    LR.fit(transformed_data, ydata)
    m = LR.coef_
    c = LR.intercept_

    return m, c


xdata, ydata, distFunction = fetchData()
sin_time, cos_time = calcComponents(xdata)
#xdata = [x+39 if x <= 15 else x-56 for x in xdata]

# Generate linear data that is circular about the x-axis.
#xdata = np.arange(0,96,1)
#y = [-3*x+201 for x in xdata]
#
#x1 = [(x+30) for x in xdata[:66]]
#x2 = [(x-66) for x in xdata[66:]]

#xdata = x1+x2
#ydata = y
#plt.scatter(x1+x2, y)
#plt.show()

def plotTimeComponents():
    timeFig, timeaxes = plt.subplots(2,2,figsize=(15,15))
    axes = timeFig.get_axes()

# Time data with normal Linear Regression.
    axes[0].scatter(xdata, ydata, color='blue',s=5)
    axes[0].set_xlabel('Time of Day',fontsize=20)
    axes[0].set_ylabel('Volume of Traffic',fontsize=20)
#axes[0].scatter(xdata, [((24*(np.arctan2(sin_m,cos_m)*x))/(2*np.pi)) for x in xdata], color='red')
    m, c =  LinReg([xdata], ydata)
    axes[0].scatter(xdata, [m*x+c for x in xdata], color='lime')

#Plotting sin and cos components
    axes[1].scatter(sin_time, cos_time, color='blue',s=6)
    axes[1].set_xlabel(r'sin($2\pi \frac{\mathrm{Time of Day}}{96})$',fontsize=20)
    axes[1].set_ylabel(r'cos($2\pi \frac{\mathrm{Time of Day}}{96})$',fontsize=20)
    axes[1].set_xlim(-1,1)
    axes[1].set_ylim(-1,1)

#Linear Regression on  sin component
    axes[2].scatter(sin_time, ydata, color='blue',s=5)
    axes[2].set_xlabel(r'sin($2\pi \frac{\mathrm{Time of Day}}{96})$',fontsize=20)
    axes[2].set_ylabel('Volume of Traffic',fontsize=20)
    axes[2].set_xlim(-1,1)
    sin_m, sin_c = LinReg([sin_time], ydata)
    axes[2].plot(sin_time, [(sin_m*sin +sin_c) for sin in sin_time], color='red', linewidth=10)


#Linear Regression on cos component
    axes[3].scatter(cos_time, ydata, color='blue',s=5)
    axes[3].set_xlabel(r'sin($2\pi \frac{\mathrm{Time of Day}}{96})$',fontsize=20)
    axes[3].set_ylabel('Volume of Traffic',fontsize=20)
    axes[3].set_xlim(-1,1)
    cos_m, cos_c = LinReg([cos_time], ydata)
    axes[3].plot(cos_time, [(cos_m*cos +cos_c) for cos in cos_time], color='red', linewidth=10)

    timeFig.savefig('timeData.pdf')
#
#

def plotCircularData(xdata, ydata, ys):
    data=[]
    trace1 = go.Scatter3d(
        x=sin_time,
        y=cos_time,
        z=ydata,
        mode='markers',
        marker={
            'size': 2,
            'opacity': 0.8,
            'color': 'black'
        }
    )
    lindata = np.array(ys).flatten()
    print(lindata)
    trace2 = go.Scatter3d(
        x=sin_time,
        y=cos_time,
        z=lindata,
#        z = ydata,
        mode='markers',
        marker={'size': 12, 'color': 'red'}
        # marker={
        #     'size': 2,
        #     'opacity': 0.8,
        # }
    )

    data.append(trace1)
    data.append(trace2)
    #Configure the layout.
    layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


def rotateData(xdata, rotation=10, boundary=96):
    rotatedXdata = []
    for x in xdata:
        if (x+rotation) >= boundary:
            rotatedX = (x+rotation)-boundary
            rotatedXdata.append(rotatedX)
        else:
            rotatedXdata.append(x+rotation)

    return rotatedXdata

def checkBoundary(xdata, boundary=96):
    for x in xdata:
        diff = abs(xdata[0] - x)
        if (96-diff) < diff:
            return True
    return False

def findClusterEdges(xdata):
    diffs = list(map(cyclic, [xdata[0]]*len(xdata), xdata))
    firstEdge = xdata[diffs.index(max(diffs))]

    diffs = list(map(cyclic, [firstEdge]*len(xdata), xdata))
    secondEdge = xdata[diffs.index(max(diffs))]

    return min(firstEdge, secondEdge), max(firstEdge, secondEdge)

def shiftCluster(xdata, firstEdge, secondEdge, boundary=96):
    print(firstEdge, secondEdge)
    shiftedX = []
    for x in xdata:
        if x>=secondEdge:
            shiftedX.append(x-secondEdge)
        else:
            shiftedX.append(x+(boundary-secondEdge))
    return shiftedX

def rotateRegression(xdata, ydata):
    fig, axes =  plt.subplots(1,1, figsize = (10,10))
    axesList = fig.get_axes()
    axes.scatter(xdata, ydata, c='blue', s=2)
    if checkBoundary(xdata):
        firstEdge, secondEdge = findClusterEdges(xdata)
        shiftedX = shiftCluster(xdata, firstEdge, secondEdge)
#            axes.scatter(shiftedX, ydata, c='lime', s=2)
        m, c = LinReg([shiftedX], ydata)
#        xdata = shiftedX
        ys = [(m*x+c) for x in shiftedX]
    else:
        m, c = LinReg([xdata], ydata)
        ys = [(m*x+c) for x in xdata]
    print(m, c)
#        axes].scatter(xdata, ydata, c='blue', s=2)
    axes.scatter(xdata, ys, color='red', linewidth=5)
    axes.set_xlim(0,96)
    plotCircularData(xdata, ydata, ys)
    fig.savefig('rotatedSamples.pdf')
rotateRegression(xdata, ydata)



