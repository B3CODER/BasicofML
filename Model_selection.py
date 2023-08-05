import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
heart_data = pd.read_csv('heart.csv')
print(heart_data.head())
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X = np.asarray(X)
Y = np.asarray(Y)

models = [LogisticRegression(max_iter=1000) , SVC(kernel ='linear') , KNeighborsClassifier() , RandomForestClassifier(random_state=0)]

def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model,X,Y,cv=5)
        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy= round(mean_accuracy*100,2)
    print('Cross Validation accuracies for the',model,'=', cv_score)
    print('Acccuracy score of the ',model,'=',mean_accuracy,'%')
    
print(compare_models_cross_validation())

model_list = [LogisticRegression(max_iter=1000) , SVC(), KNeighborsClassifier(), RandomForestClassifier(random_state=0)]

model_hyperparameters ={
    'log_reg_hyperparameters':{
        'C': [1,5,10,20]
    },
    
    'svc_hyperparameters' : {
        'kernel':['linear', 'poly','rbf','sigmoid'],
        'C':[1,5,10,20]
    },
    
    'KNN_hyperparameters' : {
        'n_neighbors':[3,5,10,15]
    },
    'random_forest_hyperparameters':{
        'n_estimators':[10,20,50,100]
    }
    
}
model_keys = list(model_hyperparameters.keys())
#  Applying GridSearchCV

def ModelSelection(list_of_models , hyperparameters_dictionary):
    result =[]
    i=0
    
    for model in list_of_models:
        key = model_keys[i]
        params = hyperparameters_dictionary[key]
        
        i+=1
        
        print(model)
        print(params)
        
        print()
        
        classifier = GridSearchCV(model , params , cv=5)
        
        classifier.fit(X,Y)
        
        result.append({
            'model used' : model,
            'highest score' : classifier.best_score_,
            'best Hyperparameters' : classifier.best_params_
        })
        
        result_dataframe = pd.DataFrame(result , columns =['model used','highest score','best Hyperparameters'])
        return result_dataframe
    
ModelSelection(model_list , model_hyperparameters)

