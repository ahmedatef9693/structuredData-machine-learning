import pickle
import time
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
def Hot_Encoding(DataFrame):
    #Getting Dummy Columns Means Creating Columns For the Specified List of Rows

    Columns = pd.get_dummies(DataFrame.fueltype)
    #Concatinating those Dummy Variable Columns To Original Data Frame
    NewDataFrame = pd.concat([DataFrame, Columns], axis='columns')
    EncodedDataFrame = NewDataFrame.drop(['fueltype'], axis='columns')
    return EncodedDataFrame

def Label_Encoding(Columns,Hot_Encoded):
    NewDataFrame = {"doornumber": {"two": 2, "four": 4},
                    "cylindernumber":{"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12}
                    }
    Hot_Encoded.replace(NewDataFrame,inplace = True)
    RestToBeEncoded = []

    X = Hot_Encoded

    for C in Columns:
        RestToBeEncoded.append(X[C])


    for i,c in zip (range(len(RestToBeEncoded)),Columns):
        labelencoder = LabelEncoder()
        labelencoder.fit(list(RestToBeEncoded[i].values))
        X[c] = labelencoder.transform(list(RestToBeEncoded[i].values))
    X['wheelbase'].fillna(X['wheelbase'].mean(),inplace = True)

    #Col = X.columns[X.isnull().any()]

    #null_data = X[X.isnull().any(axis=1)]
    # if X['wheelbase'].isnull().any().any():
    #     print('True')
    # else:
    #     print('False')
    #booleanvalue = pd.isnull(X)
    # if X.isnull().any().any():
    #     print('True')
    # else:
    #     print('False')
    return X

def FeatureScalerNormalization(DataFrame,X,a,b):
    Normalized_X = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i] = ((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    ListOfNamesForDataFrame = DataFrame.columns.tolist()
    DataFrame = pd.DataFrame(data =Normalized_X[0:,0:],index=[i for i in range(Normalized_X.shape[0])],columns=[C for i,C in zip(range(Normalized_X.shape[1]),ListOfNamesForDataFrame)])
    return DataFrame

def FillNullValues(Data,NullColumnNames):
    for Name in NullColumnNames:
        AverageOfColumn = Data[Name].mean()
        RoundedAverage = round(AverageOfColumn)
        Data[Name].fillna(RoundedAverage, inplace=True)
    return Data

def PickleModel(Model,num):
    if num == 1:
        with open('MileStone1FirstModel', 'wb') as f:
            pickle.dump(Model, f)
    elif num == 2:
        with open('MileStone1SecondModel', 'wb') as f:
            pickle.dump(Model, f)
























