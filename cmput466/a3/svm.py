from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.metrics import confusion_matrix
from keras.layers import AveragePooling2D
import numpy as np
from sklearn.model_selection import GridSearchCV
def prep_pixels(train, test):
	# normalize to range 0-1
	train_norm = train / 255.0
	test_norm = test / 255.0
	return train_norm, test_norm

def low_resolution(trainX, testX):
  avgpool = AveragePooling2D(pool_size = (2,2), strides = (2,2))
  trainX = avgpool(trainX)
  testX = avgpool(testX)
  return np.array(trainX), np.array(testX)



def mnist_train_svm(low_res):
  (train_X_org, train_Y_org), (test_X_org, test_Y_org) = mnist.load_data()
  kfold = KFold(6, shuffle=True, random_state=1)

  if low_res:
    train_X_org = train_X_org.reshape((train_X_org.shape[0], 28, 28, 1))
    test_X_org = test_X_org.reshape((test_X_org.shape[0], 28, 28, 1))
    train_X_org, test_X_org = prep_pixels(train_X_org, test_X_org)
    train_X_org, test_X_org = low_resolution(train_X_org, test_X_org)
    train_X_org = train_X_org.reshape(len(train_X_org), 14*14)
    test_X_org = test_X_org.reshape(len(test_X_org), 14*14)
  else:
    train_X_org, test_X_org = prep_pixels(train_X_org, test_X_org)
    train_X_org = train_X_org.reshape(len(train_X_org), 28*28)
    test_X_org = test_X_org.reshape(len(test_X_org), 28*28)
  for train_ix, test_ix in kfold.split(train_X_org):
    train_X = train_X_org[train_ix]
    train_Y = train_Y_org[train_ix]
    test_X = train_X_org[test_ix]
    test_Y = train_Y_org[test_ix]
    svm_clf = SVC(decision_function_shape = 'ovr', gamma = 0.001)
    svm_clf.fit(train_X, train_Y)
    print('train accuracy')
    pred = svm_clf.predict(train_X)
    print(accuracy_score(train_Y, pred))
    cm = confusion_matrix(train_Y, pred)
    plt.subplots(figsize=(10, 6))
    sb.heatmap(cm, annot = True, fmt = 'g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print('test accuracy')
    pred = svm_clf.predict(test_X)
    print(accuracy_score(test_Y, pred))
    cm = confusion_matrix(test_Y, pred)
    plt.subplots(figsize=(10, 6))
    sb.heatmap(cm, annot = True, fmt = 'g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
if __name__=="__main__":
  mnist_train_svm(low_res = True)
  mnist_train_svm(low_res = False)