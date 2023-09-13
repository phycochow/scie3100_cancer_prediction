"""The purpose of this script is to store the customisable framework that includes adjustable parameters and uses tools
from scikit-learn for supervised learning and visualization for the machine learning aspect in assignment 2 of SCIE3100.
"""
#######################################################################################################################
# Section 0 - House-keeping modules and functions #
#######################################################################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from boruta import BorutaPy
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Sets training F1 score for all decision making
f1_score_positive = make_scorer(f1_score, pos_label='Primary Tumor')


def load_data(file_path):
    # Load the data using Pandas (adjust the delimiter if needed)
    data = pd.read_csv(file_path, index_col=0)

    # Extract features (all columns except the label column)
    X = data.drop(columns=['Label'])  # Replace 'Label' with the actual label column name

    # Extract labels (assuming the label column name is 'Label')
    y = data['Label']

    return X, y


# Sets training F1 score for all decision making
f1_score_positive = make_scorer(f1_score, pos_label='Primary Tumor')


#######################################################################################################################
# Section 1 - Data Transformation #
#######################################################################################################################
# Function for data transformation
def data_transformation(X, transformation_type):
    if transformation_type == 'min-max':
        scaler = MinMaxScaler()
    elif transformation_type == 'standardization':
        scaler = StandardScaler()
    elif transformation_type == 'log2':
        X = pd.DataFrame(np.log2(X + 1))  # Add 1 to avoid log(0)
        return X
    else:
        return X  # No transformation

    # Apply the selected transformation
    X_transformed = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns, index=X.index)
    return X_transformed


#######################################################################################################################
# Section 2 - Feature Selection #
#######################################################################################################################
# Function for dimensionality reduction with Boruta
def boruta_selector(X, y):
    """
    Perform dimensionality reduction using the Boruta algorithm.

    Parameters:
    - X: Feature matrix (pandas DataFrame).
    - y: Target variable (pandas Series).
    - boruta_params: Dictionary of hyperparameters for Boruta.

    Returns:
    - X_selected: Feature matrix with selected features.
    """

    # Create a random forest classifier
    forest = RandomForestClassifier()
    forest.fit(X.values, y.values)

    # Initialize Boruta
    boruta_rf = BorutaPy(forest, n_estimators='auto', random_state=42)
    boruta_rf.fit(X.values, y.values)

    # Create a mask of selected features
    feature_mask = boruta_rf.support_

    # Extract the selected feature names
    selected_feature_names = X.columns[feature_mask]

    # Create a new DataFrame with only the selected features
    X_selected = X[selected_feature_names]

    return X_selected


def rf_selector(X, y):
    reduction_rf_params = {
        'n_estimators': [100],
        'max_depth': [30],
        'random_state': [42],  # Random state for reproducibility (seed)
        'verbose': [0],
        # 'max_depth': [30, 40, 50],
        # 'n_estimators': [100, 200, 300],
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
        cv=10,
        scoring=f1_score_positive,
    )

    grid_search.fit(X.values, y.values)

    # Get the best estimator with the best hyperparameters
    best_rf = grid_search.best_estimator_

    # A lazy way to select features automatically based on feature importances
    importances = best_rf.feature_importances_
    num_features_to_select = np.sum(importances > np.mean(importances))

    # Added threshold to deal with a stupid bug, we use max_features to select instead of setting threshold to mean
    feature_selector = SelectFromModel(best_rf, threshold=-np.inf, max_features=num_features_to_select)

    # Transform the data to select the top features
    X_selected = feature_selector.fit_transform(X.values, y.values)

    # Create a DataFrame with the selected features and original column names
    X_selected = pd.DataFrame(X_selected, columns=X.columns[feature_selector.get_support()])

    return X_selected


def data_reduction(X, y, reduction_type):
    """
    Perform data reduction based on the specified reduction_type.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - reduction_type: 'boruta' or 'random_forest'.

    Returns:
    - selected_features: Feature dataframe  with selected features.
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

# -------------------------------------------Defining adjustable parameters-------------------------------------------
classify_rf_params = {
    'n_estimators': [70, 90, 110, 130],
    'max_depth': [15, 30, 45, 60, 75, 90],
    # 'n_estimators': [100],
    # 'max_depth': [50],
    'random_state': [42],  # Random state for reproducibility (seed)
    'verbose': [0],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'bootstrap': [True, False],
    # 'class_weight': [None, 'balanced']
}

classify_dt_params = {
    'criterion': ['gini', 'entropy'],  # Split criterion: Gini impurity or entropy
    'max_depth': [None, 50, 100, 150, 200],  # Maximum depth of the tree (None means no limit)
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    # 'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider for the best split
    # 'criterion': ['gini'],  # Split criterion: Gini impurity or entropy
    # 'max_depth': [20],  # Maximum depth of the tree (None means no limit)
    # 'min_samples_split': [2],  # Minimum samples required to split an internal node
    # 'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    # 'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider for the best split
    'random_state': [42],  # Random state for reproducibility (seed)
}

#######################################################################################################################
# Main framework #
#######################################################################################################################
# Loading files
mrna_file_path = "G1/G1_breast_gene-expr.csv"
methyl_file_path = "G1/G1_breast_dna-meth.csv"

mrna_X, mrna_y = load_data(mrna_file_path)
methyl_X, methyl_y = load_data(methyl_file_path)

# Irrelevant at the moment
real_mrna_data = "G1/mystery_gene-expr.csv"
real_methyl_data = "G1/mystery_dna-meth.csv"

real_mrna_X, real_mrna_y = load_data(real_mrna_data)
real_methyl_X, real_methyl_y = load_data(real_methyl_data)

# Impute NaN values with 0
real_mrna_X = real_mrna_X.fillna(0)
real_methyl_X = real_methyl_X.fillna(0)

# ----------------------------Transforming data----------------------------
transformation_types = ['no_transformation', 'min-max', 'standardization', 'log2']
# log2 transformation on gene expression data and real test data
mrna_X_transformed = data_transformation(mrna_X, 'log2')
real_mrna_X_transformed = data_transformation(real_mrna_X, 'log2')
# standardization transformation on methylation data
methyl_X_transformed = data_transformation(methyl_X, 'standardization')
real_methyl_X_transformed = data_transformation(real_methyl_X, 'standardization')

# ----------------------------Reducing data----------------------------
reduction_types = ['no_reduction', 'random_forest', 'boruta']

mrna_reduced_X_dict, methyl_reduced_X_dict, reduction_X_key = {}, {}, 0
real_mrna_reduced_X_dict, real_methyl_reduced_X_dict = {}, {}

for reduction_type in reduction_types:
    reduction_X_key += 1
    # Generate the reduced/non-reduced datasets
    mrna_X_reduced = data_reduction(mrna_X_transformed, mrna_y, reduction_type)
    real_mrna_X_reduced = data_reduction(real_mrna_X_transformed, real_mrna_y, reduction_type)

    methyl_X_reduced = data_reduction(methyl_X_transformed, methyl_y, reduction_type)
    real_methyl_X_reduced = data_reduction(real_methyl_X_transformed, real_methyl_y, reduction_type)

    # Stores the datasets in the respective dicts for recalling
    mrna_reduced_X_dict[reduction_X_key] = mrna_X_reduced
    methyl_reduced_X_dict[reduction_X_key] = methyl_X_reduced

    real_mrna_reduced_X_dict[reduction_X_key] = real_mrna_X_reduced
    real_methyl_reduced_X_dict[reduction_X_key] = real_methyl_X_reduced

# ----------------------------Storing data----------------------------
for reduction_X_key in range(1, 4):
    df1 = pd.concat([pd.DataFrame(mrna_reduced_X_dict[reduction_X_key]), pd.DataFrame(mrna_y, columns=['Label'])], axis=1)
    df1.to_csv(f"G1/mrna_reduced_X{reduction_X_key}.csv")

    df2 = pd.concat([pd.DataFrame(methyl_reduced_X_dict[reduction_X_key]), pd.DataFrame(methyl_y, columns=['Label'])], axis=1)
    df2.to_csv(f"G1/methyl_reduced_X{reduction_X_key}.csv")

    df3 = pd.concat([pd.DataFrame(real_mrna_reduced_X_dict[reduction_X_key]), pd.DataFrame(real_mrna_y, columns=['Label'])], axis=1)
    df3.to_csv(f"G1/real_mrna_reduced_X{reduction_X_key}.csv")

    df4 = pd.concat([pd.DataFrame(real_methyl_reduced_X_dict[reduction_X_key]), pd.DataFrame(real_methyl_y, columns=['Label'])], axis=1)
    df4.to_csv(f"G1/real_methyl_reduced_X{reduction_X_key}.csv")

# ----------------------------Loading data----------------------------
mrna_reduced_X_dict, methyl_reduced_X_dict, real_mrna_reduced_X_dict, real_methyl_reduced_X_dict = {}, {}, {}, {}

for reduction_X_key in range(1, 4):
    # Stores the datasets in the respective dicts for recalling
    mrna_reduced_X_dict[reduction_X_key],  mrna_y = load_data(f"G1/mrna_reduced_X{reduction_X_key}.csv")
    methyl_reduced_X_dict[reduction_X_key],  methyl_y = load_data(f"G1/methyl_reduced_X{reduction_X_key}.csv")

    real_mrna_reduced_X_dict[reduction_X_key], real_mrna_y = load_data(f"G1/real_mrna_reduced_X{reduction_X_key}.csv")
    real_methyl_reduced_X_dict[reduction_X_key], real_methyl_y = load_data(f"G1/real_methyl_reduced_X{reduction_X_key}.csv")

# ----------------------------Classifying data----------------------------
results = {}
# Loop over the respective number of reduced X datasets to sort the actual test data
for i in range(1, reduction_X_key + 1):
    mrna_X_reduced = mrna_reduced_X_dict[i]
    methyl_X_reduced = methyl_reduced_X_dict[i]

    # Sort the real test dataset by the columns of the processed training dataset
    temp_real_mrna_X = real_mrna_X[mrna_X_reduced.columns]
    temp_real_methyl_X = real_methyl_X[methyl_X_reduced.columns]

    # Set up the classifiers
    rf_classifier = RandomForestClassifier()
    dt_classifier = DecisionTreeClassifier()

    # Cross-validation to find the best model (params set in Section 3)
    mrna_rf_grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=classify_rf_params,
        cv=10,  # Number of cross-validation folds
        scoring=f1_score_positive,  # Measure with F1 score or accuracy
        # n_jobs=-1  # Use all available CPU cores for parallelism
    )

    mrna_dt_grid_search = GridSearchCV(
        estimator=dt_classifier,
        param_grid=classify_dt_params,
        cv=10,  # Number of cross-validation folds
        scoring=f1_score_positive,  # Measure with F1 score or accuracy
        # n_jobs=-1  # Use all available CPU cores for parallelism
    )

    # Create duplicate search for methyl datasets
    methyl_rf_grid_search = mrna_rf_grid_search
    methyl_dt_grid_search = mrna_dt_grid_search

    # Conduct the actual searches
    mrna_rf_grid_search.fit(mrna_X_reduced, mrna_y.values)
    mrna_dt_grid_search.fit(mrna_X_reduced, mrna_y.values)

    methyl_rf_grid_search.fit(methyl_X_reduced, methyl_y.values)
    methyl_dt_grid_search.fit(methyl_X_reduced, methyl_y.values)

    # Extract the best estimators
    best_mrna_rf = mrna_rf_grid_search.best_estimator_
    best_mrna_dt = mrna_dt_grid_search.best_estimator_

    best_methyl_rf = methyl_rf_grid_search.best_estimator_
    best_methyl_dt = methyl_dt_grid_search.best_estimator_

    # ----------------------------Predicting data----------------------------
    # Make the actual predictions
    best_mrna_rf_predictions = best_mrna_rf.predict(temp_real_mrna_X)
    best_mrna_dt_predictions = best_mrna_dt.predict(temp_real_mrna_X)

    best_methyl_rf_predictions = best_methyl_rf.predict(temp_real_methyl_X)
    best_methyl_dt_predictions = best_methyl_dt.predict(temp_real_methyl_X)

    # Calculating the scores
    best_mrna_rf_f1 = f1_score(real_mrna_y.values, best_mrna_rf_predictions, average='macro')
    best_mrna_rf_accuracy = accuracy_score(real_mrna_y.values, best_mrna_rf_predictions)

    best_mrna_dt_f1 = f1_score(real_mrna_y.values, best_mrna_dt_predictions, average='macro')
    best_mrna_dt_accuracy = accuracy_score(real_mrna_y.values, best_mrna_dt_predictions)

    best_methyl_rf_f1 = f1_score(real_mrna_y.values, best_methyl_rf_predictions, average='macro')
    best_methyl_rf_accuracy = accuracy_score(real_methyl_y.values, best_methyl_rf_predictions)

    best_methyl_dt_f1 = f1_score(real_mrna_y.values, best_mrna_rf_predictions, average='macro')
    best_methyl_dt_accuracy = accuracy_score(real_methyl_y.values, best_mrna_rf_predictions)

    results[reduction_X_key] = [(best_mrna_rf, best_mrna_rf_f1, best_mrna_rf_accuracy), (best_mrna_dt, best_mrna_dt_f1, best_mrna_dt_accuracy),
                                (best_methyl_rf, best_methyl_rf_f1, best_methyl_rf_accuracy), (best_methyl_dt, best_methyl_dt_f1, best_methyl_dt_accuracy)]
    print(f"reduction_X_key: {i}")
    print(f"model: {best_mrna_rf}, best_mrna_rf_f1: {best_mrna_rf_f1}, best_mrna_rf_accuracy: {best_mrna_rf_accuracy}")
    print(f"model: {best_mrna_dt}, best_mrna_dt_f1: {best_mrna_dt_f1}, best_mrna_dt_accuracy: {best_mrna_dt_accuracy}")
    print(f"model: {best_methyl_rf}, best_methyl_rf_f1: {best_methyl_rf_f1}, best_methyl_rf_accuracy: {best_methyl_rf_accuracy}")
    print(f"model: {best_methyl_dt}, best_methyl_dt_f1: {best_methyl_dt_f1}, best_methyl_dt_accuracy: {best_methyl_dt_accuracy}")

