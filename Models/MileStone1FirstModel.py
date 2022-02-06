from MileStone1Preprocessing import *

#Multiple Linear Regression Model
Data = pd.read_csv('CarPrice_training.csv');


#PreProcessing
#Data.dropna(axis=0, how="any", thresh = 15, subset=None, inplace=True)
NewData = Data.iloc[:,:]
#null_data = NewData[NewData.isnull().any(axis=1)]
#

#One Hot Encoding for fueltype column
Hot_Encoded = Hot_Encoding(NewData)

MyColumns = ('CarName','symboling','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem')
Data = Label_Encoding(MyColumns,Hot_Encoded)
NullColumnNames = Data.columns[Data.isnull().any()]
Data = FillNullValues(Data,NullColumnNames)
Data.drop(['CarName'],axis = 1,inplace = True)
Y = Hot_Encoded['price']
Data.drop(['price'],axis = 1,inplace = True)
X = Data.iloc[:,:]


#Correlation
Correlation = NewData.corr()
TopFeatures = Correlation.index[abs(Correlation['price'] > 0.5)]
plt.subplots(figsize =(12,8))
TopCorrelation = NewData[TopFeatures].corr()
sns.heatmap(TopCorrelation , annot=True)
plt.show()

for col_name in X.columns:
    if col_name not in TopFeatures:
        if col_name == 'gas':
            continue
        else:
            X.drop([col_name], axis=1, inplace=True)



X =FeatureScalerNormalization(X,np.array(X),0,1)

#Splitting Data 70% Training and 30% Testing

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.30,shuffle=True)


MultiLinearModel = linear_model.LinearRegression()
start = time.time()
MultiLinearModel.fit(Xtrain,Ytrain)
stop = time.time()
TrainingPrediction = MultiLinearModel.predict(Xtrain)
Prediction = MultiLinearModel.predict(Xtest)
plt.figure(figsize=(10,8))
plt.scatter(Xtrain['horsepower'],Ytrain)
plt.xlabel('horsepower')
plt.ylabel('price')
plt.plot(Xtrain['horsepower'],TrainingPrediction)
plt.show()


TrainingTime = stop-start
TrainTime = str(TrainingTime)

#Saving Trained Model To A File
PickleModel(MultiLinearModel,1)

#Prediction On Testing Dataset
print('Training Time :',TrainTime + ' Seconds')
#print('Co-efficient of linear regression',MultiLinearModel.coef_)

print('Mean Square Error', metrics.mean_squared_error(Ytest, Prediction))
print('Accuracy of training = ',metrics.r2_score(Ytrain,TrainingPrediction))
print('Accuracy of testing = ',metrics.r2_score(Ytest,Prediction))
true_Car_price=np.asarray(Ytest)[0]
predicted_Car_price=Prediction[0]
print('True value for the first Car : ' + str(true_Car_price))
print('Predicted value for the first Car : ' + str(predicted_Car_price))








