import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

iris_csv = pd.read_csv('Iris.csv')
del iris_csv["Id"]

iris_csv = iris_csv.replace({'Iris-setosa': "1"})
iris_csv = iris_csv.replace({'Iris-versicolor': "2"})
iris_csv = iris_csv.replace({'Iris-virginica': "3"})

X = np.array(iris_csv.iloc[:, 0:4])
Y = np.array([[iris_csv.Species]])
Y = Y.reshape(150)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.246, random_state=0)


model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score for Naive Bayes: ", accuracy_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.32, random_state=0)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score for KNN: ", accuracy_score(y_test, y_pred))
