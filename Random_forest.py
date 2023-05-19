import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data , columns= iris.feature_names)
print(df.head())

df['species'] = pd.Categorical.from_codes(iris.target , iris.target_names)
print(df.head())

df['is_train'] = np.random.uniform(0,1 , len(df)) <= .75

print(df.head())
print(df)

train , test = df [df['is_train']==True] , df[df['is_train']==False]

print("No of obs in the training data ", len(train))
print("No of obs in the test data ", len(test))

features = df.columns[:4]
print(features)

y= pd.factorize(train['species'])[0]
print(y)

clf = RandomForestClassifier(n_jobs= 2 , random_state=0)
clf.fit(train[features] ,y)

print(clf.fit(train[features] ,y))

clf.predict(test[features])

print(clf.predict(test[features]))

clf.predict_proba(test[features])[0:10]
print(clf.predict_proba(test[features])[0:10])

predictions = iris.target_names[clf.predict(test[features])]
print(predictions[0:30])

pd.crosstab(test['species'] , predictions , rownames = ['Actual Species'] ,colnames=['Predicted species'])

print(pd.crosstab(test['species'] , predictions , rownames = ['Actual Species'] ,colnames=['Predicted species']))


