import numpy as np
import matplotlib.pyplot as plt

def classify(output: np.ndarray):
    return (output > 0.5).astype(int)

def PDP(idx, feature_name, model, X, y, categorical=False):

    # Determine the x-axis of the PDP
    if categorical:
        values = np.unique(X[:,idx])
    else:
        values = np.linspace(np.min(X[:,idx]), np.max(X[:,idx]), 100)

    # Initialize the PD values
    PDP = list()

    # For each value on the x-axis, replace the feature value of every instance 
    # with this value and make a prediction
    for value in values:
        X_copy = np.copy(X)
        X_copy[:,idx] = value
        predictions = model.predict(X_copy)
        predictions = classify(predictions) # !!!
        PDP.append(np.mean(predictions))

    # Make a line plot if feature is not categorical and a bar plot if it is categorical
    plt.title("PDP of {}".format(feature_name))
    plt.xlabel(feature_name)
    plt.ylabel("prediction")

    if not categorical:
        plt.plot(values, PDP)
    else:
        diff = max(PDP) - min(PDP)
        if diff != 0.0:
            plt.ylim(max(min(PDP) - 0.2 * diff, 0.0), max(PDP) + 0.2 * diff)
        else:
            plt.ylim(min(PDP) - 0.001, max(PDP) + 0.001)
        plt.bar(values, PDP)
    
    return values, PDP