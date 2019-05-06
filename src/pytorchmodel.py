# Modeling using pytorch
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import io
from torch.utils.data import TensorDataset, Dataset
import torch.nn as nn

def tensorTraining(alldata):
    ## this function is not fully developed
    #csvdatax = alldata.iloc[:,0:alldata.columns.size-2].copy().values
    #csvdatay = alldata["SepsisLabel"].copy().values
    #dataset = TensorDataset(torch.from_numpy(csvdatax.astype('float32')), torch.from_numpy(csvdatay).long())
    patients = alldata["PatientID"].unique()
    allpatientData = []
    allpatientDataOutput = []
    for p in patients:
        pdata = alldata[alldata.PatientID == p]
        allpatientDataOutput.append(pdata["SepsisLabel"].max())
        pdata.drop(["PatientID","SepsisLabel"])
        onePatientData = []
        for i,r in pdata.iterrows:
            onePatientData.append(r)
        allpatientData.append(onePatientData)
    input=torch.FloatTensor(allpatientData)
    output=torch.LongTensor(allpatientDataOutput)
    # develop a model based on input, output tensors above and using CNN classifier below
    # reuse the same methods for train, test, split as developed earlier


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		# Introducing new layer for perf improvement
		self.fcinter = nn.Linear(in_features=128, out_features=32)
		self.fc2 = nn.Linear(32, 5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fcinter(x))
		x = self.fc2(x)				
		return x