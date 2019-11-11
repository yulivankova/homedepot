from sklearn.metrics import mean_squared_error
import numpy as np

#Root mean squared error
def RMSE(y,y_pred):
    N = len(y)
    error = np.linalg.norm(y-y_pred)/np.sqrt(N)
    return error
