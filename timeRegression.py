import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pck
from sklearn.linear_model import LinearRegression

NumSamples=5000

def fetchData():

    def cyclic(x1, x2, possValues=np.arange(0,96,1)):
        diff = abs(x1-x2)
        return min(len(possValues) - diff, diff)

    with open('WebtrisData.pck', 'rb') as file:
        data = pck.load(file)[0]
    xdata, ydata = data['Time Interval'].values[:NumSamples], data['Total Volume'].values[:NumSamples]
    distFunction = cyclic

    return xdata, ydata, distFunction

xdata, ydata, distFunction = fetchData()

# Generate linear data that is circular about the x-axis.
xdata = np.arange(0,96,1)
y = [-3*x+201 for x in xdata]

x1 = [(x+30) for x in xdata[:66]]
x2 = [(x-66) for x in xdata[66:]]


plt.scatter(x1+x2, y)
plt.show()





# Calculating sin and cos components of time interval.
sin_time = np.sin(2*np.pi*xdata/96)
cos_time = np.cos(2*np.pi*xdata/96)



timeFig, timeaxes = plt.subplots(1,2,figsize=(20,10))
timeaxes[0].scatter(sin_time, ydata, color='blue',s=10)
timeaxes[1].scatter(cos_time, ydata, color='blue',s=10)
timeFig.savefig('timeData.png')


# Linear Regression using sklearn
transformed_data = np.array([sin_time, cos_time]).T
xtrain, xtest, ytrain, ytest = transformed_data[:int(0.8*NumSamples)], transformed_data[int(0.2*NumSamples):], ydata[:int(0.8*NumSamples)], ydata[int(0.2*NumSamples):]
print(xtest, ytest)
LR = LinearRegression()
LR.fit(xtrain, ytrain)
print(LR.coef_)
print(LR.score(xtest, ytest))
preds = LR.predict(xtest)

# Plotting the results
regressionFig, regressionAxes = plt.subplots(1,2,figsize=(10,5))
regressionAxes[0].scatter([x[0] for x in xtest], preds, color='blue',s=10)
regressionAxes[0].scatter([x[0] for x in xtest], ytest, color='red',s=2)
regressionAxes[1].scatter([x[1] for x in xtest], preds, color='blue',s=10)
regressionAxes[1].scatter([x[1] for x in xtest], ytest, color='red',s=2)
regressionFig.savefig('timeRegression.png')





