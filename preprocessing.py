from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def load_diabetes_prepr():
    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names 

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y, feature_names