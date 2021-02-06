import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

os.chdir('./')
dataset = pd.read_csv('aa.csv')

X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:, 6].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 3)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Doğruluk",classifier.score(X_test,y_test))
print("Kesinlik:",metrics.precision_score(y_test, y_pred)) 
print("Recall:",metrics.recall_score(y_test, y_pred)) 
print("f1 score:",f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)


