import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

heart_data = pd.read_csv('heart.csv')
print(heart_data.head())
print(heart_data.shape)

print(heart_data.isnull().sum())

# checking the distribution of Target Variable
heart_data['target'].value_counts()

X= heart_data.drop(columns='target' , axis=1)
Y=heart_data['target']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=3)
print(X.shape, X_train.shape, X_test.shape)

models = [LogisticRegression(max_iter=1000) ,SVC(kernel='linear') , KNeighborsClassifier(),RandomForestClassifier()]

# def compare_models_train_test():
#     for model in models:
#         model.fit(X_train,Y_train)
#         test_data_predicition = model.predict(X_test)
#         accuracy = accuracy_score(Y_test , test_data_predicition)

#         print(model ,' = ', accuracy)

# compare_models_train_test()


#  Using of cross_validation for best model selection 
def compare_models_cross_validation():

    for model in models:
        cv_score = cross_val_score(model, X,Y, cv=5)
    
        mean_accuracy = sum(cv_score)/len(cv_score)

        mean_accuracy = mean_accuracy*100

        mean_accuracy = round(mean_accuracy, 2)

        print('Cross Validation accuracies for ', model, '=  ', cv_score)
        print('Accuracy % of the ', model, mean_accuracy)
        print('----------------------------------------------')

compare_models_cross_validation()
