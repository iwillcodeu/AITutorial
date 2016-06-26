import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


# Split the data into training/testing sets
X_train = [[0,1,2],[3,4,5]]
X_test = [[6,7,8]]

# Split the targets into training/testing sets
y_train = [[0,4,8],[12,16,20]]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
# Plot outputs
print regr.predict(X_test)

