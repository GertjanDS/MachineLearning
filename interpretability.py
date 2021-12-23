import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def classify(output: np.ndarray):
    return (output > 0.5).astype(int)
    
def inverse_transform(X, feat_idx, scaler):
    return X * np.sqrt(scaler.var_[feat_idx]) + scaler.mean_[feat_idx]

def postproces_pred_class(y_pred):
    y_pred = classify(y_pred)
    return np.reshape(y_pred, y_pred.shape[0])

def postproces_pred_regr(y_pred):
    return np.reshape(y_pred, y_pred.shape[0])

def PDP(mode, idx, feature_name, model, X, y, categorical=False):

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

        if mode == "classification":
            predictions = postproces_pred_class(predictions)
        elif mode == "regression":
            predictions = postproces_pred_regr(predictions)

        PDP.append(np.mean(predictions))
    
    return values, PDP

def generate_counterfactuals(mode, instance, X_train, model, categorical, seed, features_to_vary="all", n_counterfactuals=10, total_time=5000, sample_time=100, limit_varied_features=True, goal_pred=None, alpha=1.0, tol=1.0):

    assert mode in ("classification", "regression")

    np.random.seed(seed)

    n_features = instance.shape[-1]

    categorical_values = list()

    for i in range(n_features):
        if categorical[i]:
            categorical_values.append(np.unique(X_train[:, i]))
        else:
            categorical_values.append(None)

    instance = np.reshape(instance, (1, n_features))

    original_pred = model.predict(instance)
    if mode == "classification":
        original_pred = postproces_pred_class(original_pred)[0]
    elif mode == "regression":
        original_pred = postproces_pred_regr(original_pred)[0]
    print(f"Original prediction: {original_pred}")

    counterfactuals = instance * np.ones((n_counterfactuals, n_features))
    counterfactual_losses = np.ones(n_counterfactuals) * 100

    update_counter = 0
    # total_counter = 0

    if features_to_vary=="all":
        features_to_vary = [True] * n_features

    # while update_counter < update_time:
    for total_counter in tqdm(range(total_time)):
        counterfactual_found = False
        sample_counter = 0
        prev_instance = np.copy(instance)

        while sample_counter < sample_time and not counterfactual_found:
            # Change a feature value randomly
            feature_idx = np.random.randint(n_features)

            while not features_to_vary[feature_idx]:
                feature_idx = np.random.randint(n_features)

            if categorical[feature_idx]:
                new_value = np.random.choice(categorical_values[feature_idx])
            else:
                minimum_value = np.min(X_train[:,feature_idx]) # extend this range ?
                maximum_value = np.max(X_train[:,feature_idx])
                new_value = np.random.random() * (maximum_value - minimum_value) + minimum_value
                
            new_instance = np.copy(prev_instance)
            new_instance[:,feature_idx] = new_value

            # Make a new prediction
            new_pred = model.predict(new_instance)
            if mode == "classification":
                new_pred = postproces_pred_class(new_pred)[0]
            elif mode == "regression":
                new_pred = postproces_pred_regr(new_pred)[0]
            
            # Check if new instance is a counterfactual
            if mode == "classification":
                counterfactual_found = (new_pred != original_pred)
            elif mode == "regression":
                counterfactual_found = (new_pred > goal_pred - tol and new_pred < goal_pred + tol)

            if counterfactual_found:
                if mode == "classification":
                    counterfactual_loss = np.sum(np.abs(new_instance - instance))
                elif mode == "regression":
                    counterfactual_loss = np.sum(np.abs(new_instance - instance)) + alpha * np.abs(new_pred - goal_pred)

                # Check if counterfactual already in the set
                # not (new_instance == counterfactuals).all(axis=1).any()

                # Standard counterfactual to replace is the one with the biggest loss value
                idx_to_replace = np.argmax(counterfactual_losses)

                # If varied features are limited we replace the instance with the same
                # varied features
                if limit_varied_features:
                    counterfactuals_varied_features = ((counterfactuals - instance) != 0.0)
                    varied_features = ((new_instance - instance) != 0.0)
                
                    counterfactuals_with_same_varied_features = (varied_features == counterfactuals_varied_features).all(axis=1)

                    if counterfactuals_with_same_varied_features.any():
                        idx_to_replace = np.where(counterfactuals_with_same_varied_features)
                
                # Check if new loss value is smaller than the one we want to replace
                if counterfactual_loss < counterfactual_losses[idx_to_replace]:
                    # Replace counterfactual in set with found instance
                    counterfactuals[idx_to_replace] = new_instance
                    counterfactual_losses[idx_to_replace] = counterfactual_loss
                    
                    # print("---------------------------------------------------------------")
                    # print(f"Counts total: {total_counter}")
                    # print(f"Counts since last update: {update_counter}")
                    # print(f"Counterfactual losses: {counterfactual_losses}")

                    update_counter = 0
            
            prev_instance = np.copy(new_instance)

            sample_counter += 1

        update_counter += 1
        # total_counter += 1
        # if not (total_counter % 1000):
            # print("---------------------------------------------------------------")
            # print(f"Total counts: {total_counter}")

    # print("---------------------------------------------------------------")
    # print(f"Total counts: {total_counter}")
    # print(f"Counts since last update: {update_counter}")

    sort_indices = np.argsort(counterfactual_losses)
    counterfactual_losses = counterfactual_losses[sort_indices]
    counterfactuals = counterfactuals[sort_indices]

    print(f"Counterfactual_losses: {counterfactual_losses}")

    return counterfactuals, counterfactual_losses