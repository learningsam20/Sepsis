# This module is used for all dataset processing related functions
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import glob
from utils import *
import pandas as pd
import math
#import re # may not be needed since it is trivial file name format
import six
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset from the input data in psv format
def loadDataset():
    #sqlContext = createContext()
    #alldata = sqlContext.read.options(header='true', inferschema='true').load('p00001.psv')
    #print(alldata.take(1))
    # Not able to start the spark session, so proceeding with local load for now
    print("Not implementing due to docker error")

# Load dataset in a dataframe...this can later on be converted to pig/spark or any other format
def loadDatasetLocal():
    path = "/training/*.psv"
    alldata = pd.DataFrame()
    #Taking data only for first 20 psv files for testing, remove this after done with entire coding
    for fname in glob.glob(path):
        #print(fname)
        #../training\p04992.psv
        data = pd.read_csv(fname,sep="|")
        data["PatientID"] = fname[fname.find("training")+len("training")+1 : fname.find(".psv")]
        if alldata.empty == True:
            alldata = data
        else:
            alldata = alldata.append(data,ignore_index=True)
    print("Data loading completed")
    print(alldata.head(2))
    print(alldata.shape)
    alldata.to_csv("/training/output/combined.psv", sep='|')
    print("Data written to a combined file")    
    #Descriptive analysis
    #print(alldata.describe())

    checkCorrelation(alldata)
    return alldata

# Check the correlation between various attributes loaded
def checkCorrelation(alldata: pd.DataFrame):
    #for i in alldata.columns:
    #if not( isinstance(alldata.select(i).take(1)[0][0], six.string_types)):
    #    print( "Correlation to MV for ", i, alldata.stat.corr('SepsisLabel',i))
    # Setting 100 as the min number of observations given the data volume
    # This can be changed to 5% of total length of dataframe
    # Also, using Pearson coefficient for correlation, can be changed later
    corr=alldata.corr(method="pearson",min_periods=100)
    plotCorrelation(corr)

def impute_missing(alldata):
    columns = alldata.columns
    index = alldata.index
    testdata=formBaseClusters(alldata)
    #missing = SimpleImputer(missing_values='np.nan', strategy='most_frequent')
    #missing = SimpleImputer(strategy='constant',fill_value=-0.0009)
    missing = SimpleImputer(strategy='constant',fill_value=-0.0009)
    missing.fit(alldata)
    #print(missing.transform(alldata).shape)
    alldata= pd.DataFrame(missing.transform(alldata),columns=columns,index=index)
    #print(alldata.isna().sum())
    return alldata

# Split input data into train, validation, test data
def traintestSplit(alldata):
    alldata=impute_missing(alldata)
    Y=alldata["SepsisLabel"].astype('int')
    X=alldata.drop(["SepsisLabel", "PatientID"],axis=1)
    # Data split into 30% test, 70% train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # Train data further split into 20% validation, 80% train
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    print("Number of samples in train data: " + str(len(X_train)))
    print("Number of samples in validation data: " + str(len(X_val)))
    print("Number of samples in test data: " + str(len(X_test)))
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Process the dataset based on demographics to be used for filling nan values
def formBaseClusters(alldata):
    print("pre-processing demographics based clusters and returns averages for all parameters")
    testdata=alldata
    columns = alldata.columns
    index = alldata.index

    return testdata

# Mini analysis
def demographicsAnalysis():
    alldata=pd.read_csv("/training/output/combined.psv",sep="|")
    print("data read")
    data=alldata[alldata.columns[35:43]]
    print(data.isna().sum())
    #data=data.drop("SepsisLabel",axis=1)
    print(data.shape)
    print(data.head(2))
    print(data.columns)
    #print(data.describe())
    histage=data[["Age"]].plot(kind="hist",by="Age",bins=50)
    histadmit=data[["HospAdmTime"]].plot(kind="hist",by="HospAdmTime",bins=50)
    histicu=data[["ICULOS"]].plot(kind="hist",by="ICULOS",bins=50)
    histunit1=data[["Unit1"]].plot(kind="hist",by="Unit1",bins=50)
    histunit2=data[["Unit2"]].plot(kind="hist",by="Unit2",bins=50)
    fig = histage.get_figure()
    fig.savefig('../agesepsis.png')
    fig = histadmit.get_figure()
    fig.savefig('../admitsepsis.png')    
    fig = histicu.get_figure()
    fig.savefig('../icusepsis.png')
    fig = histunit1.get_figure()
    fig.savefig('../unit1sepsis.png')
    fig = histunit2.get_figure()
    fig.savefig('../unit2sepsis.png')    

# load the massaged data
def loadDatasetMassaged():
    alldata=pd.read_csv("/training/output/massaged.psv",sep="|")
    print("loaded massaged data")
    #alldata.describe()
    #alldata.info()
    return alldata

# Group the data based on demographic clusters to fill in the na values
def groupAverageForNone():

    # Assign category ID based on Gender, Age, Unit1, Unit2, ICU and admit time bin
    # Calculate mean of the indicator variables for the category
    # Identify patient specific category
    # Calcualte the category ID along with its means
    # Calculate 
    tdata = pd.DataFrame()
    pdata = pd.DataFrame({"PatientID":["0"]})
    alldata=pd.read_csv("/training/output/combined.psv",sep="|")
    alldata = alldata.drop(alldata.columns[[0]], axis=1)  # drop first column that got created with index 
    attrcolumns = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
    print("data read")
    #print(alldata.shape)
    #data=alldata[alldata.columns[35:43]]
    print(alldata.isna().sum())
    cols = alldata.columns
    #print(alldata[alldata.SepsisLabel == 1]["SepsisLabel"].count())
    #for i in range(alldata.shape[0]):
    for rowdata in alldata.itertuples():
        
        #print(alldata.loc[i])
        #rowdata = irow
        if (pdata.empty == True) | (pdata[pdata.PatientID == getattr(rowdata,"PatientID")].shape[0]==0):
            print(getattr(rowdata,"PatientID"))
            pdata = alldata[alldata.PatientID == getattr(rowdata,"PatientID")]
            unit1mean = pdata["Unit1"].mean()
            unit2mean = pdata["Unit2"].mean()
            if math.isnan(unit1mean):
                unit1mean = math.floor(alldata["Unit1"].mean())
            else:
                unit1mean = math.floor(unit1mean)
            if math.isnan(unit2mean):
                unit2mean = math.floor(alldata["Unit2"].mean())
            else:
                unit2mean = math.floor(unit2mean)         
            alldata.at[rowdata.Index,"Unit1"] = unit1mean
            alldata.at[rowdata.Index,"Unit2"] = unit2mean
            #rowdata[37] = unit1mean
            #rowdata[38] = unit2mean
            samegender = alldata[(alldata.Gender == getattr(rowdata,"Gender")) & (alldata.PatientID != getattr(rowdata,"PatientID"))]
            sameage = samegender[(samegender.Age == getattr(rowdata,"Age"))]
            if(sameage.empty == True):
                sameage = samegender[(samegender.Age >= getattr(rowdata,"Age")-2) & (samegender.Age <= getattr(rowdata,"Age")+2)]
            if(sameage.empty == True):
                sameage = samegender
            # 6 hour of admit time
            sameadmit = sameage[(sameage.HospAdmTime == getattr(rowdata,"HospAdmTime"))]
            if(sameadmit.empty == True):
                sameadmit = sameage[(sameage.HospAdmTime >= getattr(rowdata,"HospAdmTime") - 3) & (sameage.HospAdmTime <= getattr(rowdata,"HospAdmTime") + 3)]
            if(sameadmit.empty == True):
                sameadmit = sameage
            # 4 hour of ICU time
            sameicu = sameadmit[(sameadmit.ICULOS == getattr(rowdata,"ICULOS"))]
            if(sameicu.empty == True):
                sameicu = sameadmit[(sameadmit.ICULOS >= getattr(rowdata,"ICULOS") - 2) & (sameadmit.ICULOS <= getattr(rowdata,"ICULOS") + 2)]
            if(sameicu.empty == True):
                sameicu = sameadmit[(sameadmit.ICULOS >= getattr(rowdata,"ICULOS") - 10) & (sameadmit.ICULOS <= getattr(rowdata,"ICULOS") + 10)]
            if(sameicu.empty == True):
                sameicu = sameadmit
            sameunit1 = sameicu[(sameicu.Unit1 == getattr(rowdata,"Unit1"))]
            if(sameunit1.empty == True):
                sameunit1 = sameicu
            sameunit2 = sameunit1[(sameunit1.Unit2 == getattr(rowdata,"Unit2"))]
            if(sameunit2.empty == True):
                sameunit2 = sameunit1
        
        # Processing for lab and test values
        #for j in range(36):
        for j in attrcolumns:
            jval = getattr(rowdata,j)
            if(math.isnan(jval)):
                meanval=pdata[j].mean()
                if(math.isnan(meanval)):
                    meanval = sameunit2[j].mean()
                    if(math.isnan(meanval)):
                        meanval = sameunit1[j].mean()
                    if(math.isnan(meanval)):
                        meanval = sameicu[j].mean()
                    if(math.isnan(meanval)):
                        meanval = sameadmit[j].mean()
                    if(math.isnan(meanval)):
                        meanval = sameage[j].mean()
                    if(math.isnan(meanval)):
                        meanval = samegender[j].mean()
                    if(math.isnan(meanval)):
                        meanval = 0                        
                #rowdata[j] = meanval
                alldata.at[rowdata.Index,j] = meanval

                #print(cols[j])
                #print(pdata[cols[j]].count())
                #print(pdata[cols[j]].mean())
                #print(jval)
        #alldata.update(rowdata)
        #if tdata.empty == True:
        #    tdata = pd.DataFrame(rowdata)
        #else:
        #    tdata = tdata.append(rowdata,ignore_index=True)
        #print(tdata.columns)
    #tdata.columns = alldata.columns
    tdata=alldata
    tdata.to_csv("/training/output/massaged.psv", sep='|',index=False)
        #    #print(alldata.loc[i][j])
        #if alldata.empty == True:
        #    data = rowdata
        #else:
        #    data = data.append(rowdata,ignore_index=True)        
    return alldata