#%% import packages
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import f
from scipy.spatial.distance import cdist
import geopandas as gpd
from pysal.lib import weights
#%% find_best_model
def find_best_model(performance_results, performance_measure):

    """
    Find the best model based on a given performance measure.

    Args:
        performance_results (dict): A dictionary of performance results for different models.
                                    Keys are model names, and values are dictionaries of performance measures.
        performance_measure (str): The performance measure to use for model comparison.

    Returns:
        tuple: A tuple containing the best model name and its corresponding score.

    """

    best_model = None
    best_score = None

    for model_name, scores in performance_results.items():
        score = scores[performance_measure]

        if best_score is None:
            best_model = model_name
            best_score = score
        else:
            if (
                (performance_measure == 'Mean Squared Error' or performance_measure == 'Mean Absolute Error')
                and score < best_score
            ):
                best_model = model_name
                best_score = score
            elif (
                (performance_measure == 'R-squared' or performance_measure == 'Explained Variance Score')
                and score > best_score
            ):
                best_model = model_name
                best_score = score

    return best_model, best_score
#%% create_polynomials
def create_polynomials(final, features_interact, features_no_interact):

    """
    Create polynomial features from the given DataFrame.

    Args:
        final (DataFrame): The input DataFrame.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.

    Returns:
        DataFrame: A DataFrame with polynomial features.

    """

    poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly_features.fit_transform(final[features_interact])
    X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(features_interact))
    X_poly = pd.concat([X_poly, final[features_no_interact]], axis=1)

    return X_poly
#%% create_log_interactions
def create_log_interactions(df, features_interact, features_no_interact, all=True):
    
    """
    Create logarithm-interacted features from the given DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Defaults to True.

    Returns:
        DataFrame: A DataFrame with logarithm-interacted features.

    """
        
    df_log_interact = df.copy()

    # Apply logarithm transformation to interacted features
    for feature in features_interact:
        df_log_interact[feature] = np.log(df_log_interact[feature] * 100 + 1)

    # Create logarithm interactions
    poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_interact = poly_features.fit_transform(df_log_interact[features_interact])
    X_interact = pd.DataFrame(X_interact, columns=poly_features.get_feature_names_out(features_interact))

    # Apply logarithm transformation to non-interacted features if all=True
    if all:
        for feature in features_no_interact:
            if feature not in ['const','avg_height']:
                df_log_interact[feature] = np.log(df_log_interact[feature] * 100 + 1)

    # Concatenate logarithm-interacted features with transformed non-interacted features
    return pd.concat([X_interact, df_log_interact[features_no_interact]], axis=1)
#%% compute_avg_marginal_effect
def compute_avg_marginal_effect(model, final, feature, features_interact, features_no_interact, mode="poly", all=False, delta=0.001, step=0.01):
    
    """
    Compute the average marginal effect of a feature on the model's predictions.

    Args:
        model: The trained model.
        final (DataFrame): The input DataFrame.
        feature (str): The name of the feature for which to compute the marginal effect.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        mode (str, optional): The mode for feature transformation. Can be "poly" for polynomial features
                              or "log" for logarithm-interacted features. Defaults to "poly".
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Applicable only when mode="log". Defaults to False.
        delta (float, optional): The change in the feature value for computing marginal effect. Defaults to 0.001.
        step (float, optional): The step size for scaling the average marginal effect. Defaults to 0.01.

    Returns:
        float: The average marginal effect of the feature.

    """

    final_delta = final.copy()
    final_delta[feature] = final_delta[feature] + delta

    if mode == "poly":
        X = create_polynomials(final, features_interact, features_no_interact)
        X_delta = create_polynomials(final_delta, features_interact, features_no_interact)
    elif mode == "log":
        X = create_log_interactions(final, features_interact, features_no_interact, all=all)
        X_delta = create_log_interactions(final_delta, features_interact, features_no_interact, all=all)

    y_original = model.predict(X)
    y_modified = model.predict(X_delta)

    marginal_effects = (y_modified - y_original) / delta

    avg_marginal_effect = np.mean(marginal_effects) * step

    return avg_marginal_effect
#%% compute_marginal_effect_at_avg
def compute_marginal_effect_at_avg(model, final, feature, features_interact, features_no_interact, mode="poly", all=False, delta=0.001, step=0.01):
    
    """
    Compute the marginal effect of a feature at the average.

    Args:
        model: The trained model.
        final (DataFrame): The input DataFrame.
        feature (str): The name of the feature for which to compute the marginal effect.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        mode (str, optional): The mode for feature transformation. Can be "poly" for polynomial features
                              or "log" for logarithm-interacted features. Defaults to "poly".
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Applicable only when mode="log". Defaults to False.
        delta (float, optional): The change in the feature value for computing marginal effect. Defaults to 0.001.
        step (float, optional): The step size for scaling the marginal effect. Defaults to 0.01.

    Returns:
        float: The marginal effect of the feature at the average.

    """
    
    final = final[features_interact + features_no_interact]
    feature_avg = pd.DataFrame(final.mean(axis=0)).T
    feature_avg_delta = feature_avg.copy()

    feature_avg_delta[feature] += delta

    if mode == "poly":
        X = create_polynomials(feature_avg, features_interact, features_no_interact)
        X_delta = create_polynomials(feature_avg_delta, features_interact, features_no_interact)
    elif mode == "log":
        X = create_log_interactions(feature_avg, features_interact, features_no_interact, all=all)
        X_delta = create_log_interactions(feature_avg_delta, features_interact, features_no_interact, all=all)

    y_original = model.predict(X)
    y_modified = model.predict(X_delta)

    marginal_effect = (y_modified - y_original) / delta

    return marginal_effect.item()*step
#%% compute_avg_marginal_effect_adv
def compute_avg_marginal_effect_adv(model, final, feature, features_interact, features_no_interact, mode="poly", all=False, delta=0.001, step=0.01):
    
    """
    More advanved computation of average marginal effect incorporating changes in other features accounting for highly correlated features.

    Args:
        model: The trained model.
        final (DataFrame): The input DataFrame.
        feature (str): The name of the feature for which to compute the marginal effect.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        mode (str, optional): The mode for feature transformation. Can be "poly" for polynomial features
                              or "log" for logarithm-interacted features. Defaults to "poly".
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Applicable only when mode="log". Defaults to False.
        delta (float, optional): The change in the feature value for computing marginal effect. Defaults to 0.001.
        step (float, optional): The step size for scaling the average marginal effect. Defaults to 0.01.

    Returns:
        float: The average marginal effect of the feature.

    """
    
    final_delta = final.copy()
    final_delta[feature] = final_delta[feature] + delta

    for i in features_interact:
        if i != feature:
            final_delta[i] = (final_delta[i] - (delta / len(features_interact))).clip(lower=0)

    if mode == "poly":
        X = create_polynomials(final, features_interact, features_no_interact)
        X_delta = create_polynomials(final_delta, features_interact, features_no_interact)
    elif mode == "log":
        X = create_log_interactions(final, features_interact, features_no_interact, all=all)
        X_delta = create_log_interactions(final_delta, features_interact, features_no_interact, all=all)

    y_original = model.predict(X)
    y_modified = model.predict(X_delta)

    marginal_effects = (y_modified - y_original) / delta

    avg_marginal_effect = np.mean(marginal_effects) * step

    return avg_marginal_effect
#%% predict_LST
def predict_LST_example(example, features_interact, features_no_interact, model, mode="poly", all=False):

    """
    Predict the Land Surface Temperature (LST) using an example input.

    Args:
        example (DataFrame): The example input data.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        model: The trained model.
        mode (str, optional): The mode for feature transformation. Can be "poly" for polynomial features
                              or "log" for logarithm-interacted features. Defaults to "poly".
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Applicable only when mode="log". Defaults to False.

    Returns:
        float: The predicted Land Surface Temperature (LST).

    """

    if mode == "poly":
        example_trans = create_polynomials(example.reset_index(drop=True), features_interact, features_no_interact)
    elif mode == "log":
        example_trans = create_log_interactions(example.reset_index(drop=True), features_interact, features_no_interact, all=all)
    pred = model.predict(example_trans)
    return pred.item()
#%% test_joint_significance
def test_joint_significance(model_unrestricted, final, features_interact, features_no_interact, target, features_exclude=[], mode="poly", all=False):

    """
    Test the joint significance of a group of features in a model.

    Args:
        model_unrestricted: The unrestricted model.
        final (DataFrame): The input DataFrame.
        features_interact (list): A list of features to interact.
        features_no_interact (list): A list of features that should not be interacted.
        target: The target variable.
        features_exclude (list, optional): A list of features to exclude from the restricted model. Defaults to [].
        mode (str, optional): The mode for feature transformation. Can be "poly" for polynomial features
                              or "log" for logarithm-interacted features. Defaults to "poly".
        all (bool, optional): Indicates whether to apply logarithm transformation to non-interacted features.
                              Applicable only when mode="log". Defaults to False.

    Returns:
        float: The F-statistic value.
        float: The p-value.

    """
    
    # Create polynomials for the restricted model
    features_interact_restricted = [f for f in features_interact if f not in features_exclude]

    if mode == "poly":
        X_restricted = create_polynomials(final, features_interact_restricted, features_no_interact)
    elif mode == "log":
        X_restricted = create_log_interactions(final, features_interact_restricted, features_no_interact, all=all)
    
    # Fit the restricted model
    model_restricted = sm.OLS(target, X_restricted)
    results_restricted = model_restricted.fit(cov_type='HC3')

    # Perform the F-test
    rss_full = model_unrestricted.ssr
    rss_restricted = results_restricted.ssr
    dof_full = model_unrestricted.df_resid
    dof_restricted = results_restricted.df_resid
    f_statistic = ((rss_restricted - rss_full) / (dof_restricted - dof_full)) / (rss_full / dof_full)
    p_value = 1 - f.cdf(f_statistic, dof_restricted - dof_full, dof_full)

    return f_statistic, p_value
#%% get_nearest_neighbor_weights
def get_nearest_neighbor_weights(final, k=4, factor=1.95):

    """
    Compute nearest neighbor weights for spatial analysis.

    Args:
        final (DataFrame): The input DataFrame containing spatial data.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 4.
        factor (float, optional): The factor to determine the threshold for defining neighbors. Defaults to 1.95.

    Returns:
        weights.KNN: The K-nearest neighbors weights object.
        pd.Series: A series containing the divisor values for each observation.

    """

    knn = weights.KNN.from_dataframe(final, k=k, geom_col='geometry')

    points = gpd.GeoDataFrame(final.geometry).geometry.centroid
    df = pd.DataFrame({'points': points})
    coordinates = df['points'].apply(lambda point: (point.x, point.y)).tolist()

    dist_matrix = cdist(coordinates, coordinates)

    threshold = dist_matrix[0,1]*factor
    mask = (dist_matrix < threshold) & (dist_matrix > 0)
    closest_indices = [np.where(row)[0].tolist() for row in mask.T]

    for i in range(len(closest_indices)):
        knn.neighbors[i] = closest_indices[i]

    divisor = [len(knn.neighbors[i]) for i in range(len(knn.neighbors))]

    return knn, pd.Series(divisor)
#%% add_feature_lags
def add_feature_lags(final, features, k=8):

    """
    Add spatial lagged features to the input DataFrame.

    Args:
        final (DataFrame): The input DataFrame containing spatial data.
        features (list): A list of features for which to compute spatial lags.
        k (int, optional): The number of neighbors to consider for computing spatial lags. Defaults to 8.

    Returns:
        DataFrame: The input DataFrame with added spatial lagged features.

    """
    
    knn = weights.KNN.from_dataframe(final, k=k, geom_col='geometry')
    lag = final[features].apply(
    lambda y: weights.spatial_lag.lag_spatial(knn, y) / k
    ).rename(
        columns=lambda c: "lag_" + c
    )
    out = final.join(lag)

    return out