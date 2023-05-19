# Equation of the Hyperplane:
#  y = wx - b

# Gradient Descent is an optimization algorithm used for 
# minimizing the loss function in various machine learning 
# algorithms. It is used for updating the parameters of 
# the learning model.

import numpy as np

class SVM_classifier():
    def __init__(self,learning_rate , no_of_iterations , lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self,X,Y):
        # m  --> number of Data points --> number of rows
        # n  --> number of input features --> number of columns
        self.m ,self.n = X.shape

        self.w =np.zeros(self.n)
        self.b = 0
        self.X =X
        self.Y =Y
        
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        y_label =  np.where(self.Y<=0 , -1 ,1)

        # gradients ( dw, db)

        for y_i , x_i in enumerate(self.X):
            condition = y_label[y_i]*(np.dot(x_i,self.w) - self.b) >= 1
            if(condition == True):
                dw = 2* self.lambda_parameter*self.w
                db = 0
            else:
                dw = 2*self.lambda_parameter*self.w - np.dot(x_i , y_label[y_i])
                db = y_label[y_i]

            
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self,X):
        output = np.dot(X,self.w) - self.b

        predicted_labels = np.sign(output)

        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('diabetes.csv')
print(diabetes_data.head())
# print(diabetes_data.sum().isnull)

print(diabetes_data['Outcome'].value_counts())

features = diabetes_data.drop(columns='Outcome', axis=1)

target = diabetes_data['Outcome']
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)

features = standardized_data
target = diabetes_data['Outcome']

X_train ,X_test , Y_train ,Y_test = train_test_split(features , target , test_size =0.2 , random_state= 3)
print(features.shape, X_train.shape, X_test.shape)

classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction ,Y_train)

print("Accuracy socre ", training_data_accuracy)

