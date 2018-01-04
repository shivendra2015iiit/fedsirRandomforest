import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from sklearn import datasets
import array


boston = datasets.load_boston()

features = pd.DataFrame(boston.data,columns=boston.feature_names)
targets = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(features, targets, train_size=0.8, random_state=42)

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

x = [0] * len(Y_test)
x = [i for i in range(len(Y_test))]
print(test_score)
plt.scatter(x,predicted_test,marker='o')
plt.scatter(x,Y_test,marker='x')
plt.show()