#This acts as the main class for building the python model for preicting if the patient would belong to sepsis class or not
#import the necessary python libraries
from model import *
from dataset import *

def main():
    print("Welcome to Sepsis Predictor")
    #Analysis over all data to idetify patterns, bins for grouping
    #demographicsAnalysis()
    #alldata = groupAverageForNone();
    #alldata=loadDatasetLocal()

    ## pre-process data
    alldata = loadDatasetMassaged()
    
    # Split the data in train, test, validate
    #X_train, X_val, X_test, Y_train, Y_val, Y_test = traintestSplit(alldata)
    ## We intend to input time series modelling for this problem with each input of the following form
    ## list of [[data for visit1 for patientx],[data for visit2 for patientx],..] or all the patients
    ## Load the above in pytorch
    ## Train the classfier algorithm based on the above visit data

    # Models for various types of algorithms
    # Logistic regression
    logmodelfile = doLogisticRegression(X_train,Y_train)
    #SGD classifier
    #sgdmodelfile =doSGDClassification(X_train,Y_train)
    #SGD regression
    #sgdrmodelfile =doSGDRegression(X_train,Y_train)
    #Gaussian classifier
    #gmodelfile =doGaussianClassification(X_train,Y_train)
    #print(logmodelfile)
    #print(sgdmodelfile)
    #print(sgdrmodelfile)
    #print(gmodelfile)
    # load the model from disk
    logmodelfile = '/training/output/sepsis.model'
    logmodel = loadModel(logmodelfile);
    applyModel(logmodel, X_train,Y_train,"validation")
    #applyModel(logmodel, X_val,Y_val,"validation")
    #applyModel(logmodel, X_test,Y_test,"test")
    print("End of Sepsis Predictor processing")

if __name__ == '__main__':
	main()