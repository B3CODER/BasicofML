import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV



data = pd.read_csv('UCI_Credit_Card.csv')
# print(data.shape)
# print(data.head())
# print(data.info())

features = data.drop(["default.payment.next.month"], axis=1)
targets = data["default.payment.next.month"]
lr_sklearn = LogisticRegression(random_state=42)
lr_sklearn.fit(features,targets)

x=lr_sklearn.score(features,targets)
# print(x)

#                             ---> Optimizing method
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
# scaled_lr_model = LogisticRegression(random_state=42)
# scaled_lr_model.fit(scaled_features, targets)

# logit_coef = np.exp(scaled_lr_model.coef_[0]) - 1
# # print(logit_coef)
# idx = abs(logit_coef).argsort()[::-1]

# plt.bar(range(len(idx)), logit_coef[idx])

# # We can see that the most important feature is PAY_0
# plt.xticks(range(len(idx)),features.columns[idx], rotation=90)
# # plt.show()

np.random.seed(42)
lr_model = sm.Logit(targets,features)
# lr_results = lr_model.fit()
# # print(lr_results.summary())


# selected_features = sm.add_constant(features).loc[:, lr_results.pvalues < 0.05]
# lr_model_trimmed = sm.Logit(targets, selected_features)
# lr_trimmed_results = lr_model_trimmed.fit()
# lr_trimmed_results.summary()

# predictions = (lr_trimmed_results.predict(selected_features) > 0.5).astype('int')

# x= accuracy_score(predictions, targets)
# # print(x)

# lr_results = lr_model.fit(method='newton', maxiter=10)
# lr_sklearn = LogisticRegression(solver='newton-cg', max_iter=1000)
# lr_sklearn.fit(features, targets)


# Regularization

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
lr_sklearn = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)
lr_sklearn.fit(scaled_features, targets)

# We could consider dropping these features going forward, especially since they also had
# large p-values from the statsmodels results.
# print(lr_sklearn.coef_)


scaled_features_df = pd.DataFrame(scaled_features,columns=features.columns,
                                  index=features.index)
lr_model = sm.Logit(targets, sm.add_constant(scaled_features_df))
reg_results = lr_model.fit_regularized(alpha=100)
# print(reg_results.summary())

lr_cv = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10, 100],
 solver='liblinear',
 penalty='l1',
 n_jobs=-1,# We also set the n_jobs parameter to -1, which tells sklearn to use all available CPU cores in parallel to run the crossvalidation process
 random_state=42)
lr_cv.fit(scaled_features, targets)
# print(lr_cv.scores_)
print(lr_cv.scores_[1].mean(axis=0))