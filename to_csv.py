import numpy as np
import pandas as pd

x = np.load("X_train.npy")
print(x.shape)
y = np.load("y_train.npy")
print(y.shape)
x = pd.DataFrame(x, columns=['x_' + str(i) for i in range(x.shape[1])], dtype=np.float32)
y = pd.DataFrame(y, columns=['y_' + str(i) for i in range(y.shape[1])], dtype=np.float32)

x.fillna(x.mean(), inplace=True)
x.to_csv('x.csv', index=False)
y.to_csv('y.csv', index=False)