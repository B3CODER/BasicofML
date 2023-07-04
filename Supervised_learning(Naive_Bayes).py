import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

data = pd.read_csv('UCI_Credit_Card.csv')
# print(data.shape)
# print(data.head())
# print(data.info())

features = data.drop(["default.payment.next.month"], axis=1)
targets = data["default.payment.next.month"]

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(features, targets)
print(gnb.score(features, targets))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(features, targets)
print(knn.score(features, targets))