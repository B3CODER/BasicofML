import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()

X = diabetes_dataset.drop(columns='Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

classifier = KNeighborsClassifier(p=1)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

print(accuracy*100)