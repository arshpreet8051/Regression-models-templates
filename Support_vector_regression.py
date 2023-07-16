# Support Vector Regression (SVR)

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the SVR model on the whole dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)) # Note: Here 'y.inverse_transform id used because of feature scaling applied earlier'
# output -> array([[170370.0204065]])

# Visualising the SVR results

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Support Vector Regression(SVR)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Support Vector Regression (SVR)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

