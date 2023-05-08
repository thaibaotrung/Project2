import numpy as np
import matplotlib as plt
from sklearn import neighbors, datasets

#load và hiển thị dữ liệu trong datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

print ('Number of classes: %d' %len(np.unique(iris_Y)))
print('Number of data point %d' %len(iris_Y))

x0 = iris_X[iris_Y == 0, :]
print('\nSamples from class 0:\n', x0[:5,:])

x1 = iris_X[iris_Y==1,: ]
print('\nSamples from class 1:\n', x1[:5,:])

x2 = iris_X[iris_Y==2,: ]
print('\nSamples from class 2:\n', x2[:5,:])

#tách training và test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y,test_size=50)

print("Training size: %d" %len(Y_train))
print("Test size: %d" %len(Y_test))

#trường hợp với K = 1
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", Y_test[20:40])

#phương pháp đánh giá
from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(Y_test, y_pred)))

#trường hợp với K = 10
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", Y_test[20:40])

print("Accuracy of 10NN: %.2f %%" %(100*accuracy_score(Y_test, y_pred)))


#đánh trọng số cho các điểm lân cận
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", Y_test[20:40])

print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(Y_test, y_pred)))


#cách đánh trọng số khác 

def myweight(distances):
    sigma2 = .5
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=myweight)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", Y_test[20:40])

print("Accuracy of 10NN ( myweights): %.2f %%" %(100*accuracy_score(Y_test, y_pred)))

