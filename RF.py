import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from sklearn import datasets



features = pd.DataFrame("features.csv")

targets = pd.DataFrame("target.csv")

X_train, X_test, Y_train, Y_test = train_test_split(features , targets,train_size=0.8,random_state=42)  # using 80% data for training

scaler = StandardScaler().fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)


X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)

# regressor

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, Y_train)


# just for testing

predicted_test = rf.predict(X_test)
test_score = r2_score(Y_test, predicted_test)
spearman = spearmanr(Y_test, predicted_test)
pearson = pearsonr(Y_test, predicted_test)

plt.scatter(X_test,predicted_test)
plt.scatter(X_test,Y_test)
plt.show()