# from sklearn.datasets import load_svmlight_file
# X_train ,y_train = load_svmlight_file('ijcnn1.bz2')
# first_row =2500
# X_train , y_train =X_train[:first_row ,:] , y_train[:first_row,:]
# # print(X_train)
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
# hypothesis = SVC(kernel='rbf', random_state=101)
# scores = cross_val_score(hypothesis, X_train, y_train,cv=5, scoring='accuracy')
# print ("SVC with rbf kernel -> cross validation accuracy: \mean = %0.3f std = %0.3f" % (np.mean(scores), np.std(scores)))

import numpy as np
import pandas as pd
covertype_dataset = pd.read_csv('covtype.csv')
# print(covertype_dataset.head())
# print(covertype_dataset.shape)

covertype_X =covertype_dataset.data[:25000,:]
covertype_y = covertype_dataset[:25000] -1
covertypes = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine','Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
print('target freq:', list(zip(covertypes,np.bincount(covertype_y))))