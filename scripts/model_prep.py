import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
def preprocesser(x_train,y_train,x_test,y_test):
    scaler=MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    X_train_scaled = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))  
    X_test_scaled = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))  
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled