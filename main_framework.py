"""The purpose of this script is to store the customisable framework that includes adjustable parameters and uses tools
from scikit-learn for supervised learning and visualization for the machine learning aspect in assignment 2 of SCIE3100.
"""
#######################################################################################################################
# Setting up the environment #
#######################################################################################################################
# Import in-house packages
import helper_functions as hf

# Import necessary modules
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import make_scorer, f1_score, accuracy_score
#######################################################################################################################
# Section 1 - Data Transformation #
#######################################################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------------------------------------------------------------------------
# Sets training F1 score for all decision making
f1_score_positive = make_scorer(f1_score, pos_label='Primary Tumor')


# Function for data transformation
def data_transformation(X, transformation_type):
    if transformation_type == 'min-max':
        scaler = MinMaxScaler()
    elif transformation_type == 'standardization':
        scaler = StandardScaler()
    elif transformation_type == 'log2':
        X = np.log2(X + 1)  # Add 1 to avoid log(0)
        return X
    else:
        return X  # No transformation

    # Apply the selected transformation
    X_transformed = scaler.fit_transform(X)
    return X_transformed


# Function for model evaluation and plotting
# def evaluate_model(X, y, model, model_name):
#     # Perform 10-fold cross-validation
#     scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
#     f1_scores = cross_val_score(model, X, y, cv=10, scoring='f1_macro')
#
#     # Print and plot the results
#     print(f'{model_name} - Mean Accuracy: {np.mean(scores):.2f}')
#     print(f'{model_name} - Mean F1 Score: {np.mean(f1_scores):.2f}')
#
#     # Plot accuracy and F1 score
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.bar(range(1, 11), scores)
#     plt.title(f'{model_name} - Cross-Validation Accuracy')
#
#     plt.subplot(1, 2, 2)
#     plt.bar(range(1, 11), f1_scores)
#     plt.title(f'{model_name} - Cross-Validation F1 Score')
#
#     plt.tight_layout()
#     plt.show()


#######################################################################################################################
# Section 2 - Feature Selection #
#######################################################################################################################
from boruta import BorutaPy
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------------------------------------------------------------------
# Function for dimensionality reduction with Boruta
def boruta_selector(X, y):
    """
    Perform dimensionality reduction using the Boruta algorithm.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - boruta_params: Dictionary of hyperparameters for Boruta.

    Returns:
    - X_selected: Feature matrix with selected features.
    """
    # Special thanks to Wilson
    boruta_params = {
        'n_estimators': [100, 200, 300],  # Number of trees in the Random Forest
        'max_depth': [10, 20, 30, 40, 50],  # Maximum depth of the trees
        'random_state': [42],  # Random state for reproducibility (seed)
        'verbose': [1],  # Set to 0 for no output during fitting
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        # 'bootstrap': [True, False],
        # 'class_weight': [None, 'balanced']
    }

    # Create a random forest classifier
    forest = RandomForestClassifier()

    # Initialize Boruta
    boruta_rf = BorutaPy(forest, n_estimators='auto', random_state=42)

    # Passes through the random forest params from the boruta params into the boruta's random forest estimator
    reduction_rf_params = {
        'estimator__n_estimators': boruta_params['n_estimators'],
        'estimator__max_depth': boruta_params['max_depth'],
        'estimator__random_state': boruta_params['random_state'],
        'verbose': boruta_params['verbose']
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(
        estimator=boruta_rf,
        param_grid=reduction_rf_params,
        cv=10,  # Number of cross-validation folds
        scoring=f1_score_positive,  # Measure with F1 score or accuracy
        # n_jobs=-1  # Use all available CPU cores for parallelism
    )

    # Fit Boruta with grid search on the data
    grid_search.fit(X.values, y.values)

    # Get selected features mask
    selected_features_mask = grid_search.best_estimator_.support_

    # Create a feature matrix with the selected features
    X_selected = X.loc[:, selected_features_mask]

    return X_selected


def rf_selector(X, y):
    """
    Perform dimensionality reduction using Random Forest feature importances.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - reduction_rf_params: Dictionary of hyperparameters for Random Forest.

    Returns:
    - X_selected: Feature matrix with selected features.
    """
    reduction_rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, 40, 50],
        'random_state': [42],  # Random state for reproducibility (seed)
        'verbose': [1],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        # 'bootstrap': [True, False],
        # 'class_weight': [None, 'balanced']
    }

    # Create a Random Forest classifier
    rf_selector = RandomForestClassifier()

    # Fit the classifier to the data to calculate feature importances
    rf_selector.fit(X.values, y.values)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(
        estimator=rf_selector,
        param_grid=reduction_rf_params,
        cv=10,  # Number of cross-validation folds
        scoring=f1_score_positive,  # Measure with F1 score or accuracy
        # n_jobs=-1  # Use all available CPU cores for parallelism
    )

    grid_search.fit(X.values, y.values)

    # Get the best estimator with the best hyperparameters
    best_rf = grid_search.best_estimator_

    # A lazy way to select features automatically based on feature importances - normally this is manually chosen
    importances = best_rf.feature_importances_
    num_features_to_select = np.sum(importances > np.mean(importances))

    # Added threshold to deal with a stupid bug, we use max_features to select instead of setting threshold to mean
    feature_selector = SelectFromModel(best_rf, threshold=-np.inf, max_features=num_features_to_select)

    # Transform the data to select the top features
    feature_selector.fit(X.values, y.values)
    X_selected = feature_selector.transform(X)

    return X_selected


def data_reduction(X, y, reduction_type):
    """
    Perform data reduction based on the specified reduction_type.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - reduction_type: 'boruta' or 'random_forest'.

    Returns:
    - selected_features: Feature matrix with selected features.
    """
    if reduction_type == 'random_forest':
        X_selected = rf_selector(X, y)
    elif reduction_type == 'boruta':
        X_selected = boruta_selector(X, y)
    else:
        return X  # No reduction
    return X_selected


#######################################################################################################################
# Section 3 - Classification #
#######################################################################################################################
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------Defining adjustable parameters-------------------------------------------
classify_rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, 50],
    'random_state': [42],  # Random state for reproducibility (seed)
    'verbose': [1],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'bootstrap': [True, False],
    # 'class_weight': [None, 'balanced']
}

classify_dt_params = {
    'criterion': ['gini', 'entropy'],  # Split criterion: Gini impurity or entropy
    'max_depth': [None, 20, 40],  # Maximum depth of the tree (None means no limit)
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider for the best split
    'random_state': [42],  # Random state for reproducibility (seed)
}

#######################################################################################################################
# Main framework #
#######################################################################################################################
if __name__ == "__main__":
    """This script works by propagating X sets into a dictionary that also stores the processing steps it has undergone.
    This will be retreived at the end for the best model/processing combination. Note: The y label is nominal."""

    # Load gene expression data
    data_path = "G1/G1_breast_gene-expr.csv"
    X1, y = hf.load_data(data_path)

    X_sets_1, X_sets_2, X_sets_3 = {}, {}, {}

    # Section 1: Data Transformation
    print('-------------------------------------------\nTransforming data\n-------------------------------------------')
    # Notes: Can store processing_steps as list instead of dictionary - update it later
    combination_id = 0
    transformation_types = ['no_transformation', 'min-max', 'standardization', 'log2']
    for transform_type in transformation_types:
        X_transformed = data_transformation(X1, transform_type)
        X_sets_1[combination_id] = [X_transformed, transform_type]
        combination_id += 1

    # Section 2: Dimensionality Reduction with Boruta
    print('-------------------------------------------\nReducing data\n-------------------------------------------')
    combination_id = 0
    reduction_types = ['no_reduction', 'random_forest', 'boruta']
    for id in X_sets_1:
        processing_steps = X_sets_1[id]
        X = processing_steps[0]
        for reduction_type in reduction_types:
            processing_steps.append(reduction_type)
            X_reduced = data_reduction(X, y, reduction_type)
            X_sets_2[combination_id] = processing_steps
            combination_id += 1

    # e.g. processing_steps=[X_dataframe, transformation_type, reduction_type, scores_list]
    # Load real test data - For Assignment purposes ONLY. Sorts and runs data imputation.
    real_data = "G1/mystery_gene-expr.csv"
    real_X, real_y = hf.load_data(real_data)
    real_X = real_X[X1.columns]
    real_X = real_X.fillna(0)

    # Section 3: Classification with Random Forest and Decision Tree
    print('-------------------------------------------\nClassifying data\n-------------------------------------------')
    rf_classifier = RandomForestClassifier()
    dt_classifier = DecisionTreeClassifier()

    combination_id = 0
    for _id, processing_steps in X_sets_2.items():
        X = processing_steps[0]
        # Create a GridSearchCV objects for random forest and Decision Tree
        rf_grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=classify_rf_params,
            cv=10,  # Number of cross-validation folds
            scoring=f1_score_positive,  # Measure with F1 score or accuracy
            # n_jobs=-1  # Use all available CPU cores for parallelism
        )

        dt_grid_search = GridSearchCV(
            estimator=dt_classifier,
            param_grid=classify_dt_params,
            cv=10,  # Number of cross-validation folds
            scoring=f1_score_positive,  # Measure with F1 score or accuracy
            # n_jobs=-1  # Use all available CPU cores for parallelism
        )

        # Fit and run all combinations with cross validation n=10
        rf_grid_search.fit(X.values, y.values)
        dt_grid_search.fit(X.values, y.values)

        # Get the best estimators with the best hyperparameters
        best_rf = rf_grid_search.best_estimator_
        best_dt = dt_grid_search.best_estimator_

        # Make predictions on actual test data
        rf_predictions = best_rf.predict(real_X)
        dt_predictions = best_dt.predict(real_X)

        # Calculate F1 and accuracy scores
        rf_f1 = f1_score(real_y, rf_predictions, average='macro')
        rf_accuracy = accuracy_score(real_y, rf_predictions)
        rf_and_scores = [best_rf, rf_f1, rf_accuracy]

        dt_f1 = f1_score(real_y, dt_predictions, average='macro')
        dt_accuracy = accuracy_score(real_y, rf_predictions)
        dt_and_scores = [best_dt, dt_f1, dt_accuracy]

        # Store results
        X_sets_3[combination_id] = processing_steps.append(rf_and_scores)
        combination_id += 1
        X_sets_3[combination_id] = processing_steps.append(dt_and_scores)
        combination_id += 1

# ----------------------------------------Visualise for different combinations----------------------------------------
