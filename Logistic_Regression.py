import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Logistic_Regression():
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate = learning_rate 
        self.no_of_iteration = no_of_iteration
    def fit(self , X,Y):
        self.m ,self.n = X.shape

        self.w = np.zeros(self.n)
        self.b=0
        self.X =X
        self.Y =Y

        for i in range(self.no_of_iteration):
            self.update_weights()
    
    def update_weights(self):
        Y_hat =1/(1+ np.exp(-(self.X.dot(self.w) + self.b)))

        dw =(1/self.m)*np.dot(self.X.T , (Y_hat - self.Y))
        db = (1/self.m)*np.sum(Y_hat -self.Y)

        self.w = self.w -self.learning_rate * dw
        self.b = self.b -self.learning_rate * db

    def predict(self,X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) )) 
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred


diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
print(diabetes_dataset.shape)
# print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())
features = diabetes_dataset.drop(columns = 'Outcome', axis=1)
target = diabetes_dataset['Outcome']

print(features)

scaler = StandardScaler()
scaler.fit(features)

standarized_data = scaler.transform(features)
# print(standarized_data)

features = standarized_data
target = diabetes_dataset['Outcome']

print(features)
print(target)

X_train , X_test , Y_train , Y_test = train_test_split(features , target , test_size =0.3 , random_state =2)
print(features.shape, X_train.shape, X_test.shape)

# Training the model

classifier = Logistic_Regression(learning_rate = 0.01 , no_of_iteration= 1000)
classifier.fit(X_train ,Y_train)

# model evalution
#  --> Accuracy score 

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score( Y_train, X_train_prediction)
print('Accuracy score of the training data : ', training_data_accuracy)

# for test data accuracy score
# X_test_prediction = classifier.predict(X_test)
# test_data_accuracy = accuracy_score( Y_test, X_test_prediction)
# print('Accuracy score of the test data : ', test_data_accuracy)