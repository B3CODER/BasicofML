import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class Linear_Regression():
    def __init__(self ,Learning_rate , No_of_iteration) :
        self.Learning_rate = Learning_rate;
        self.No_of_iteration = No_of_iteration;

    def fit(self,X,Y):
        # number of training example & number of features
        self.m, self.n = X.shape

        #  intiating the weight and bias
        self.w =np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # implimanting gredient decent
        for i in range(self.No_of_iteration):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # gredient
        dw = -(2*(self.X.T).dot(self.Y - Y_prediction))/self.m
        db = -2*np.sum(self.Y - Y_prediction)/self.m

        #  now updating the weights

        self.w =self.w -self.Learning_rate*dw
        self.b =self.b -self.Learning_rate*db


    def predict(self,X):
        return X.dot(self.w) + self.b


salary_data = pd.read_csv('salary_data.csv')
# print(salary_data.head())
print(salary_data.shape)
print(salary_data.isnull().sum())


# Splitting the feature & target
X = salary_data.iloc[:,:-1].values      
Y = salary_data.iloc[:,1].values

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 2)
model = Linear_Regression(Learning_rate = 0.01, No_of_iteration=1000)
model.fit(X_train, Y_train)

# printing the parameter values ( weights & bias)

print('weight = ', model.w[0])
print('bias = ', model.b)

# y = 9514(x) + 23697
# salary = 9514(experience) + 23697

# train_data_prediction = model.predict(X_train)
# print(train_data_prediction)
# plt.scatter(X_train ,Y_train ,color ='red')
# plt.plot(X_train , train_data_prediction ,color ='blue')


test_data_prediction = model.predict(X_test)
print(test_data_prediction)

plt.scatter(X_test ,Y_test ,color ='red')
plt.plot(X_test , test_data_prediction ,color ='blue')
plt.xlabel('work experience')
plt.xlabel('salary')
plt.title('salary vs experience')
plt.show()