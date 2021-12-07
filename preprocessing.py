from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def load_diabetes_prepr():
    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_names


def load_loan_prepr():
    df = pd.read_csv('loan_data_set.csv')

    # Drop unnecessary feature
    df = df.drop("Loan_ID", axis=1)

    # Drop every instance with one or more missing features
    df = df.dropna(axis=0)

    # Define dictionaries to replace words with numeric values
    d1 = {"Y": 1, "N": 0}
    d2 = {"Yes": 1, "No": 0}
    d3 = {"Male": 1, "Female": 0}
    d4 = {"0": 0, "1": 1, "2": 2, "3+": 3}
    d5 = {"Graduate": 1, "Not Graduate": 0}
    d6 = {"Rural": 0, "Semiurban": 0.5, "Urban": 1}

    cleanup_nums = {"Loan_Status": d1, "Married": d2, "Self_Employed": d2, "Gender": d3,
    "Dependents": d4, "Education": d5, "Property_Area": d6}

    # Replace every word with a corresponding numeric value
    df.replace(cleanup_nums, inplace=True)

    # Convert everything to a NumPy array
    X = df.values[:,:-1]
    y = df.values[:,-1]
    feature_names = df.columns[:-1]

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_names


def load_wisconsin_prepr():
    wisconsin = datasets.load_breast_cancer()

    X = wisconsin.data
    y = wisconsin.target
    feature_names = wisconsin.feature_names

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_names


def load_bikes_prepr():
    df = pd.read_csv('./datasets/hour.csv')
    df = df.drop(["instant", "dteday", "casual", "registered"], axis=1)

    # Convert everything to a NumPy array
    X = df.values[:,:-1]
    y = df.values[:,-1]
    feature_names = df.columns[:-1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_names