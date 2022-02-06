from MileStone2Preprocessing import *
#-----------------------------------------------------------------------DecisionTree Model--------------------------------------------
Data = pd.read_csv('CarPrice_training_classification.csv')
ytree = Data['category']
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
Columnsforconversion = Xtrain.columns

TreeClassifierModel =DecisionTreeClassifier(criterion='entropy',max_depth=3)

TrainTimeStart = time.time()
TreeClassifierModel.fit(Xtrain,Ytrain)
TrainTimeEnd = time.time()


#Training Prediction
YpredTrain = TreeClassifierModel.predict(Xtrain)
#Training Score
TrainingScore = metrics.accuracy_score(Ytrain,YpredTrain) *100

#Plotting Data and Decision Boundary
DataFrame = pd.DataFrame(Xtrain,columns=Columnsforconversion)
DataFrame = DataFrame.assign(category =Ytrain)
figure = plt.figure(figsize=(10, 7))
Myvalues = set()
for val in ytree:
    Myvalues.add(val)

tree.plot_tree(TreeClassifierModel,max_depth=3,feature_names=Xtrain.columns,class_names=list(Myvalues),rounded=True,filled=True)
plt.show()

#Testing Prediction
TestTimeStart = time.time()
YpredTest = TreeClassifierModel.predict(Xtest)
TestTimeEnd = time.time()

#Testing Score
TestingScore = metrics.accuracy_score(Ytest,YpredTest)
ConfusionMatrix = confusion_matrix(Ytest,YpredTest)
#Calculate Training Time
TrainingTime = TrainTimeEnd-TrainTimeStart
#Calculate Test Time
TestingTime = TestTimeEnd - TestTimeStart

#Saving Model
PickleModel(TreeClassifierModel , 1)

print('Training Accuracy = '+str(TrainingScore))
print('Testing Accuracy = '+str(TestingScore*100))
print('Training Time = '+str(TrainingTime))
print('Testing Time = ',str(TestingTime))
print('Confusion Matrix = '+str(ConfusionMatrix))
print('Error = '+str(1-TestingScore))

#Prediction For A Value
ActualValueOfClass = np.asarray(Ytest)[0]
PredictedValueOfClass = np.asarray(YpredTest)[0]
print('ActualValueOfClass = ',ActualValueOfClass)
print('PredictedValueOfClass = ',PredictedValueOfClass)
