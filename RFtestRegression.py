import pandas as ps
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()

features = ps.DataFrame(boston.data,columns=boston.feature_names)
targets = boston.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=42)
