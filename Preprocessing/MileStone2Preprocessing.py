import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
def labelencoding(Dataframe,CategoryNames):
    NewDataFrame = {"doornumber": {"two": 2, "four": 4},
                    "cylindernumber":{"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12}
                    }
    Dataframe.replace(NewDataFrame, inplace=True)
    X = Dataframe
    RestToBeEncoded = []
    for c in CategoryNames:
        RestToBeEncoded.append(X[c])
    for i, c in zip(range(len(RestToBeEncoded)), CategoryNames):
        labelencoder = LabelEncoder()
        labelencoder.fit(list(RestToBeEncoded[i].values))
        X[c] = labelencoder.transform(list(RestToBeEncoded[i].values))
    Y = X['category']
    X.drop(['category'], axis=1, inplace=True)
    return X,Y

def FillNullValues(NullColumnNames,LabeledDataFrame):
    for Name in NullColumnNames:
        AverageOfColumn = LabeledDataFrame[Name].mean()
        RoundedAverage = round(AverageOfColumn)
        LabeledDataFrame[Name].fillna(RoundedAverage, inplace=True)
    return LabeledDataFrame

def PickleModel(Model,num):
    if num == 1:
        with open('MileStone2FirstModel', 'wb') as f:
            pickle.dump(Model, f)
    elif num == 2:
        with open('MileStone2SecondModel', 'wb') as f:
            pickle.dump(Model, f)
    elif num == 3:
        with open('MileStone2ThirdModel', 'wb') as f:
            pickle.dump(Model, f)

