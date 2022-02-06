from MileStone2Preprocessing import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#----------------------------------------------------------------LogisticRegression Model---------------------------------------------
Data = pd.read_csv('CarPrice_training_classification.csv')
null_data = Data[Data.isnull().any(axis=1)]
NullColumns = Data.columns[Data.isnull().any()]
CategoryNames = ('CarName','fueltype','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem','category')
LabeledData,Y = labelencoding(Data,CategoryNames)
NullColumnNames = LabeledData.columns[LabeledData.isnull().any()]
X = FillNullValues(NullColumnNames,LabeledData)

#Feature Engineering
Correlation = Data.corr()
Indices = []
Mapping = []
ImportantFeatures = set()

for r in range(Correlation.shape[0]):
    for c in range(Correlation.shape[1]):
        if (abs(Correlation.iloc[r,c])>0.5) and(Correlation.iloc[r,c]!=1):
            Indices.append([r,c])

for index in Indices:
    ImportantFeatures.add(Correlation.columns[index[0]])
    ImportantFeatures.add(Correlation.columns[index[1]])

#Selecting Features Depending On Important Ones
for col_name in X.columns:
    if col_name not in(ImportantFeatures):
        X.drop([col_name], axis=1, inplace=True)
    else:
        continue


#Splitting Data
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)

LogisticRegressionClassifier =LogisticRegression(penalty='l1')
TrainTimeStart = time.time()
LogisticRegressionClassifier.fit(Xtrain,Ytrain)
TrainTimeEnd = time.time()


#Training Prediction
YpredTrain = LogisticRegressionClassifier.predict(Xtrain)
#Training Score
TrainingScore = metrics.accuracy_score(Ytrain,YpredTrain) *100

#Plotting Data and Decision Boundary
DataFrame = Xtrain.assign(category =Ytrain)
#splitting for plotting
Low = DataFrame.loc[Y == 1]
High = DataFrame.loc[Y == 0]

#Bias
Cofficent0 = LogisticRegressionClassifier.intercept_[0]
#Best Cofficients Of Every Feature We Have After Gradient Descent There Number Is Equal To Number Of Features used
Cofficents = np.asarray(LogisticRegressionClassifier.coef_).flatten()
Cofficent1 = Cofficents[6]
Cofficent2 = Cofficents[7]
fig= plt.figure(figsize=(10,7))
MinX,MaxX = min(Xtrain.iloc[:,6]),max(Xtrain.iloc[:,6])
MinY = (-(Cofficent1*MinX)-Cofficent0)/Cofficent2
MaxY = (-(Cofficent1*MaxX)-Cofficent0)/Cofficent2
plt.scatter(Low.iloc[:,6],Low.iloc[:,7],s=10,label='Low')
plt.scatter(High.iloc[:,6],High.iloc[:,7],s=10,label='High')
plt.plot((MinX,MaxX),(MinY,MaxY),color='red',label='Decision Boundary')
plt.xlabel('Car Length')
plt.ylabel('Car Width')
plt.show()


#Testing Prediction
TestTimeStart = time.time()
YpredTest = LogisticRegressionClassifier.predict(Xtest)
TestTimeEnd = time.time()
#Testing Score
TestingScore = metrics.accuracy_score(Ytest,YpredTest)
ConfusionMatrix = confusion_matrix(Ytest,YpredTest)
#Calculate Training Time
TrainingTime = TrainTimeEnd-TrainTimeStart
#Calculate Testing Time
TestingTime = TestTimeEnd - TestTimeStart

#Saving Model
PickleModel(LogisticRegressionClassifier , 2)

print('Training Accuracy = '+str(TrainingScore))
print('Testing Accuracy = '+str(TestingScore*100))
print('Training Time = '+str(TrainingTime))
print('Testing Time = '+str(TestingTime))
print('Confusion Matrix = '+str(ConfusionMatrix))
print('Error = '+str(1-TestingScore))

#Prediction For A Value
ActualValueOfClass = np.asarray(Ytest)[0]
PredictedValueOfClass = np.asarray(YpredTest)[0]
print('ActualValueOfClass = ',ActualValueOfClass)
print('PredictedValueOfClass = ',PredictedValueOfClass)
