#%% import packages
import pandas as pd
import numpy as np
import pickle
import yaml
import os

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

home_directory = os.path.expanduser( '~' )
os.chdir(home_directory + '/DS_Project/modules')
config_path = 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% create_polynomials
def create_polynomials(final, features_interact, features_no_interact):

    poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly_features.fit_transform(final[features_interact])
    X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(features_interact))
    X_poly = pd.concat([X_poly, final[features_no_interact]], axis=1)

    return X_poly
#%% compute_avg_marginal_effect
def compute_avg_marginal_effect(model, final, feature, features_interact, features_no_interact, delta=0.001, step=0.01):
    
    final_delta = final.copy()
    final_delta[feature] = final_delta[feature] + delta

    X_poly = create_polynomials(final, features_interact, features_no_interact)
    X_delta_poly = create_polynomials(final_delta, features_interact, features_no_interact)

    y_original = model.predict(X_poly)
    y_modified = model.predict(X_delta_poly)

    marginal_effects = (y_modified - y_original) / delta

    avg_marginal_effect = np.mean(marginal_effects)*step

    return avg_marginal_effect
#%% compute_marginal_effect_at_avg
def compute_marginal_effect_at_avg(model, final, feature, features_interact, features_no_interact, delta=0.001, step=0.01):
    
    feature_avg = pd.DataFrame(final.mean(axis=0)).T
    feature_avg_delta = feature_avg.copy()

    feature_avg_delta[feature] += delta

    X_poly = create_polynomials(feature_avg, features_interact, features_no_interact)
    X_delta_poly = create_polynomials(feature_avg_delta, features_interact, features_no_interact)

    y_original = model.predict(X_poly)
    y_modified = model.predict(X_delta_poly)

    marginal_effect = (y_modified - y_original) / delta

    return marginal_effect.item()*step
#%%
def predict_LST(example, features_interact, features_no_interact, model):
    example_poly = create_polynomials(example.reset_index(drop=True), features_interact, features_no_interact)
    pred = model.predict(example_poly)
    return pred.item()