# This module would house all the utility functions necessary for the project
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Create the spark SQL context to be used for dataset processing
def createContext():
    sc= SparkContext()
    sqlContext = SQLContext(sc)
    return sqlContext

# Plot correlation plot
def plotCorrelation(corr):
    plt.title('Attribute correlation')
    fig = plt.figure()
    cmap = cm.get_cmap('jet', 20)
    sub1 = fig.add_subplot(111)
    #sub1.set_xticklabels(corr.columns,fontsize=6)
    #ylabel = [corr.columns[i] for i in range(len(corr.columns)-1,0)]
    #sub1.set_yticklabels(corr.columns,fontsize=6)
    #plt.matshow(corr)
    subfig=sub1.imshow(corr,interpolation="nearest", cmap=cmap)
    fig.colorbar(subfig, ticks=[.70,.75,.8,.85,.90,.95,1])
    plt.savefig("corr.png")