# This module would do classification modelling
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import *
from sklearn.preprocessing import Imputer
import pickle

def doLogisticRegression(X_train, Y_train):
    #Y_train = Y_train.astype('int')
    logmodel = LogisticRegression(max_iter=3000,solver='lbfgs')
    logmodel.fit(X_train,Y_train)
	# save the model to disk	
    filename = '/training/output/sepsis.model'	
    pickle.dump(logmodel, open(filename, 'wb'))	
    # predict the values for the fit model
    Y_pred = logmodel.predict(X_train)
    display_metrics("Logistic regression",Y_pred,Y_train)
    return filename

# SGD Classification
def doSGDClassification(X_train, Y_train):
    #Y_train = Y_train.astype('int')
    sgdmodel = SGDClassifier(loss='hinge',max_iter=2000,tol=0.001)
    sgdmodel.fit(X_train,Y_train)
	# save the model to disk	
    filename = '/training/output/sepsisSGD.model'	
    pickle.dump(sgdmodel, open(filename, 'wb'))	
    # predict the values for the fit model
    Y_pred = sgdmodel.predict(X_train)
    display_metrics("SGD Classification",Y_pred,Y_train)
    return filename

# SGD Regression
def doSGDRegression(X_train, Y_train):
    #Y_train = Y_train.astype('int')
    sgdmodel = SGDRegressor(loss='squared_loss',max_iter=2000,tol=0.001)
    sgdmodel.fit(X_train,Y_train)
	# save the model to disk	
    filename = '/training/output/sepsisSGDR.model'	
    pickle.dump(sgdmodel, open(filename, 'wb'))	
    # predict the values for the fit model
    Y_pred = sgdmodel.predict(X_train)
    Y_pred = [0 if y < 0.5 else 1 for y in Y_pred]
    #print(Y_pred)
    #print(Y_train)
    display_metrics("SGD Regression",Y_pred,Y_train)
    return filename

# Gaussian classification
def doGaussianClassification(X_train, Y_train):
    #Y_train = Y_train.astype('int')
    gmodel = GaussianProcessClassifier()#max_iter_predict=200)
    gmodel.fit(X_train,Y_train)
	# save the model to disk	
    filename = '/training/output/sepsisGaussian.model'	
    pickle.dump(gmodel, open(filename, 'wb'))	
    # predict the values for the fit model
    Y_pred = gmodel.predict(X_train)
    display_metrics("Gaussian Classification",Y_pred,Y_train)
    return filename

# load the pre-saved model
def loadModel(filename='/training/output/sepsis.model'):
    logmodel = pickle.load(open(filename, 'rb'))
    return logmodel


def applyModel(logmodel,X_train, Y_train,datatype):
    Y_pred = logmodel.predict(X_train)
    display_metrics("Logistic regression for " + datatype,Y_pred,Y_train)

# Calculate the performance metrics for the classifier
def classification_metrics(Y_pred, Y_true):
	accuracy = accuracy_score(Y_true, Y_pred)
	#auc = roc_auc_score(Y_true, Y_pred)
	precision = precision_score(Y_true, Y_pred)
	recall = recall_score(Y_true, Y_pred)
	f1score = f1_score(Y_true, Y_pred)
	return accuracy,"AUC commented for small dataset",precision,recall,f1score

# Show the accuracy metrics for the classifier 
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")