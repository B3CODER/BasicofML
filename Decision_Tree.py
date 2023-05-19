import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

dataset = pd.read_csv('Decision_Tree_ Dataset.csv')
print(dataset.head())
# print(dataset.sum().isnull())

X = dataset.values[: , 1:5]
Y = dataset.values[:,0]

# print(X)
# print(Y)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3, random_state= 100)

clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=100 , max_depth=3 , min_samples_leaf=5)
clf_entropy.fit(X_train ,Y_train)

y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)
print("Accuracy score ", accuracy_score(Y_test , y_pred_en)*100)

