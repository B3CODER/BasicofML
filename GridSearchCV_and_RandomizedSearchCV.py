import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# print(data_frame.head())

data_frame['label'] = breast_cancer_dataset.target
print(data_frame.shape)
print(data_frame['label'].value_counts())
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X = np.asarray(X)
Y = np.asarray(Y)

#  ---> GridSearchCV
model = SVC()

parameters = {
    'kernel': ['linear','poly','rbf','sigmoid'],
    'C':[1,5,10,20]
}

classifier =GridSearchCV(model,parameters,cv=5)

classifier.fit(X,Y)

# print(classifier.cv_results_)

best_parameters = classifier.best_params_
print(best_parameters)
highest_accuracy = classifier.best_score_
print(highest_accuracy)

result = pd.DataFrame(classifier.cv_results_)

grid_search_result = result[['param_C' , 'param_kernel' ,'mean_test_score']]

print(grid_search_result)

# ---> RandomizedSearchCV
classifier2= RandomizedSearchCV(model ,parameters ,cv=5)
classifier2.fit(X,Y)

print(classifier2.cv_results_)
best_parameters = classifier.best_params_
print(best_parameters)
highest_accuracy = classifier.best_score_
print(highest_accuracy)

# loading the results to pandas dataframe
result = pd.DataFrame(classifier.cv_results_)
randomized_search_result = result[['param_C','param_kernel','mean_test_score']]

