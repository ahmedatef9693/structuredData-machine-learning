from MileStone1Preprocessing import*
Data = pd.read_csv('CarPrice_training.csv');

#Polynomial Regression Model
#PreProcessing
#Data.dropna(axis=0, how="any", thresh = 15, subset=None, inplace=True)
NewData = Data.iloc[:,:]
# null_data = NewData[NewData.isnull().any(axis=1)]
# print('Null Data',null_data)
#One Hot Encoding for fueltype column
Hot_Encoded = Hot_Encoding(NewData)

MyColumns = ('symboling','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem')
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
        if col_name =='gas':
            continue
        else:
            X.drop([col_name], axis=1, inplace=True)

X =FeatureScalerNormalization(X,np.array(X),0,1)


#70% Training and 30% Testing

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.30,shuffle=True)
Degree = 2
Poly_Features = PolynomialFeatures(degree=Degree)
# transforms the existing features to higher degree features.
XtrainPolynomial = Poly_Features.fit_transform(Xtrain)
# # fit the transformed features to Linear Regression
Poly_Model = linear_model.Lasso(tol=1000)
start = time.time()
Poly_Model.fit(XtrainPolynomial,Ytrain)
stop = time.time()
# predicting on training data-set
Ytrain_Predicted = Poly_Model.predict(XtrainPolynomial)


#Prediction On Testing Dataset
Prediction = Poly_Model.predict(Poly_Features.fit_transform(Xtest))

#Plotting Data
plt.figure(figsize=(10,8))
plt.scatter(Xtrain['horsepower'],Ytrain)
plt.xlabel('horsepower')
plt.ylabel('price')
plt.plot(Xtrain['horsepower'],Ytrain_Predicted,color ='red')
plt.show()



TrainTime = stop-start
TrainTime = str(TrainTime)

#Saving Trained Model To A File
PickleModel(Poly_Model , 2)


print('---------------------------Polynomial Regression With Degree = ',Degree)
print('Training Time :',TrainTime+' Seconds')
print('Mean Square Error', metrics.mean_squared_error(Ytest, Prediction))
print('Accuracy of training = ',metrics.r2_score(Ytrain,Ytrain_Predicted))
print('Accuracy of testing = ',metrics.r2_score(Ytest,Prediction))
true_Car_price=np.asarray(Ytest)[0]
predicted_Car_price=Prediction[0]
print('True value for the first Car : ' + str(true_Car_price))
print('Predicted value for the first Car : ' + str(predicted_Car_price))














