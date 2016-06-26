def loadDataSet(num):
 print "digits['images'][num]] is:"
 print digits['images'][num]
 print "It's attributs are:", digits['images'][num].shape, digits['images'][num].dtype
 return digits['images'][num]

def recogImg(classifier,img):
 import numpy as np

 ss_test = np.reshape(img,(-1,64))

 y_pred = classifier.predict(ss_test)
 print("The  handwritten digit is:")

 print(y_pred[0])

if __name__ == "__main__":

 #Step 1. find the examples 
 from sklearn.datasets import load_digits
 digits = load_digits()

 #Step 2. use the first 70 examples to train our machine
 from sklearn.cross_validation import train_test_split
 X = digits.data[0:69]
 y = digits.target[0:69]
 X_train, X_test, y_train, y_test = train_test_split(X, y)

 from tensorflow.contrib import skflow
 n_classes = len(set(y_train))
 classifier = skflow.TensorFlowLinearClassifier(n_classes=n_classes)
 classifier.fit(X_train, y_train)
 
 #Step 3. Use the machine to predict the NO.101 handwritten digit:  
 img = digits['images'][100]
 recogImg(classifier,img)

 from matplotlib import pyplot as plt
 digits = load_digits()
 fig = plt.figure(figsize=(3, 3))
 plt.imshow(img, cmap="gray", interpolation='none')
 plt.show()

