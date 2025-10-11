"""GPBoost model utilities and plotting helpers.

Contains helpers to fit and visualize GPBoost models, compute SHAP values,
partial dependence plots and feature importances used in the analysis.
"""

from itertools import combinations
import os
from typing import Optional
import numpy as np
import gpboost as gpb
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from pdpbox import pdp
import seaborn as sns
import shap
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils_modelling import read_and_preprocess_dataset, split_dataset, save_gpboost_summary
# #imports for SMAC-METHOD
# import smac
# from ConfigSpace import ConfigurationSpace, Integer, Categorical
# from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
# from smac import RunHistory, Scenario

PATH = 'results_model/gp_boosting/'
os.makedirs(PATH, exist_ok=True)
seeds = [27225, 32244, 34326, 92161, 99246, 108473, 117739, 147316, 235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 777646, 778572, 936518]
SEED = seeds[12]
def get_data(classification):
    df = read_and_preprocess_dataset(classification)
    X, y, group_data = split_dataset(df)
    return X
def assess_gp_boosting(setting,classification: bool = False, try_max_depth: bool = False,
                        try_num_leaves: bool = False, try_joint: bool = False, try_num_iter: bool = False, ) -> tuple:
    """Assess the GPBoost model for regression or classification tasks, create the booster and save an initial assessment of the model."""

    if classification:
        filename = f'classification_{setting}'
    
    else:   
        filename = f'regression_{setting}'

    os.makedirs(os.path.join(PATH, filename), exist_ok=True)
    setting_path = os.path.join(PATH, filename)

    df = read_and_preprocess_dataset(classification)
    X, y, group_data = split_dataset(df)

    best_iters = []
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = create_param_space(try_max_depth, try_num_leaves,try_joint,try_num_iter)
    min_score = float('inf')
    def objective_opt(params, seed = SEED):
        nonlocal min_score
        train_set = gpb.Dataset(X_train, label=y_train) 
        valid_set = gpb.Dataset(X_val, label=y_val)

        # Convert SMAC Configuration to dictionary to avoid KeyError
        params_dict = dict(params)

        # Train the model
        bst = gpb.train(
            params=params_dict, train_set=train_set, num_boost_round=1000,#change back to 1000
            valid_sets=[valid_set], early_stopping_rounds=20,
            verbose_eval=False
        )
        y_pred = bst.predict(data=X_val, pred_latent=False)
        score = root_mean_squared_error(y_val, y_pred)
        best_iters.append(bst.best_iteration)
        if score < min_score:
            min_score = score
        return score
    scenario = Scenario(params, deterministic=True, n_trials=135, seed = SEED)
    smac = HPOFacade(scenario,objective_opt,overwrite=True)
    incumbent = smac.optimize()
    rh = smac.runhistory
    # Mapping between configs and IDs
    conf2id = {conf: cid for cid, conf in rh.ids_config.items()}

    inc_id = conf2id[incumbent]
    
    # Find index of the best config (lowest cost)
    
    best_iter = best_iters[inc_id]
    
    best_config = incumbent.get_dictionary()
    best_cost = rh.get_cost(incumbent)
    train_set = gpb.Dataset(data=X, label=y)
    gp_model = gpb.GPModel(group_data=group_data, likelihood="gaussian")

    # Train the model
    bst = gpb.train(params=best_config, train_set=train_set, gp_model=gp_model, num_boost_round=best_iter, verbose_eval=False)

    save_gpboost_summary(gp_model, os.path.join(setting_path, f'model_summary_{filename}.txt'))

    y_pred = bst.predict(data=X, group_data_pred=group_data, predict_var=True, pred_latent=False)
    r_squared = r2_score(y, y_pred['response_mean'])
    rmse = root_mean_squared_error(y, y_pred['response_mean'])

    with open(os.path.join(setting_path, 'opt_params.txt'), 'w') as f:
        f.write(str(best_config))
        f.write(f"\nBest iteration: {best_iter}")
        f.write(f"\nBest RMSE: {best_cost}")
        f.write(f"\nModel R^2: {r_squared}")
        f.write(f"\nModel RMSE: {rmse}")

    # Plot predicted vs actual values
    sns.set_theme(style='dark')
    plt.figure(figsize=(10, 8))

    plt.scatter(y_pred['response_mean'], y, color='mediumslateblue', alpha=0.3, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='firebrick', linewidth=2)
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.suptitle('Predicted vs Actual values', fontsize=16)

    plt.savefig(os.path.join(setting_path, f'predicted_vs_actual.pdf'))

    return bst, X, setting_path
def create_param_space(try_max_depth, try_num_leaves,try_joint,try_num_iter):
    param = ConfigurationSpace(
        space = {
        'learning_rate': (0.001, 1.0),
        'min_data_in_leaf': (1, 1000),
        'lambda_l2':  (0.0, 100.0),
        'max_bin': (255, 10000),
        'bagging_fraction': (0.5, 1.0),
        'feature_fraction': (0.5, 1.0),
        'max_depth': [-1],
        'verbose': [-1],
        'objective': ['regression'],
        'metric': ['rmse']
    })
    if try_num_leaves and not try_num_iter:
        num_leaves = Integer(name='num_leaves',bounds = (2,1024))
        param.add([num_leaves])
    elif try_joint:
        num_leaves = Integer(name='num_leaves',bounds = (2,1024))
        max_depth = Categorical(name='max_depth',items=[-1,1,2,3,4,5,6,7,8,9,10])
        param.add([num_leaves,max_depth])
    elif try_max_depth:
        max_depth = Categorical(name='max_depth',items=[-1,1,2,3,4,5,6,7,8,9,10])
        param.add([max_depth])
    elif try_num_iter and try_num_leaves:
        num_leaves = Integer(name='num_leaves',bounds = (2,1024))
        param.add([num_leaves])
        n_iter = Integer(name='n_iter', bounds = (1,1000)) 
        param.add([n_iter])
    return param

def create_param_grid(trial,try_max_depth, try_num_leaves,try_joint,try_num_iter):
    if try_max_depth:
        param_grid ={
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
            'max_bin': trial.suggest_int('max_bin', 255, 10000),
            'max_depth': trial.suggest_int('max_depth', 1, 10), 
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
        }
        other_params ={
            'num_leaves': 1024,
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse'
        }
    elif try_num_leaves and not try_num_iter:
        param_grid ={
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
            'max_bin': trial.suggest_int('max_bin', 255, 10000),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
        }
        other_params ={
            'max_depth': -1,
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse'
        }
    elif try_joint:
        param_grid ={
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
            'max_bin': trial.suggest_int('max_bin', 255, 10000),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
            'max_depth':trial.suggest_categorical('max_depth',[-1,1,2,3,4,5,6,7,8,9,10]),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
        }
        other_params ={
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse'
        }
    elif try_num_iter and try_num_leaves:
        param_grid ={
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
            'max_bin': trial.suggest_int('max_bin', 255, 10000),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
            'n_iter': trial.suggest_int('n_iter',1,1000),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
        }
        other_params ={
            'max_depth':-1,
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse'
        }
    return param_grid, other_params

def plot_feature_importance(booster: gpb.Booster, X: pd.DataFrame, path: Optional[str]) -> None:
    """Plot feature importances for the GPBoost model."""

    title = 'classification' if 'classification' in path else 'regression'

    # Split-based feature importances
    for importance_type in ['gain', 'split']:
        importance = 'Gain' if importance_type == 'gain' else 'Split'
        feature_importances = booster.feature_importance(importance_type=importance_type)
        plt_imp = gpb.plot_importance(booster, importance_type=importance_type, color='mediumslateblue', dpi=300, precision=0, height=0.5, figsize=(12, 3), grid=False)
        plt_imp.set_title(f'{importance} feature importance for {title} tasks', fontsize=16)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

        plt.savefig(os.path.join(path, f'feature_importance_{importance_type}_gpboost.pdf'), bbox_inches='tight', dpi=300)


def plot_partial_dependence(booster: gpb.Booster, X: pd.DataFrame, path: Optional[str]) -> None:
    """Plot partial dependence plots for the GPBoost model."""

    for feature in X.columns:
        try:
            # Check if feature has enough unique values
            if X[feature].nunique() < 2:
                print(f"Skipping feature {feature}: insufficient unique values ({X[feature].nunique()})")
                continue
                
            grid_range = (0, 1) if feature in ['learning_rate'] else (0.5, 1) if feature in ['bagging_fraction', 'feature_fraction'] else None

            pdp_dist = pdp.PDPIsolate(
                model=booster, df=X.copy(), model_features=X.columns, feature=feature, feature_name=feature,
                grid_type='percentile' if grid_range is None else 'equal', n_classes=0,
                num_grid_points=min(50, X[feature].nunique()), grid_range=grid_range, predict_kwds={"ignore_gp_model": True}
            )

            fig, axes = pdp_dist.plot(engine='matplotlib', frac_to_plot=0.1, figsize=(12, 4.5))

            os.makedirs(os.path.join(path, 'partial_dependece_isolate'), exist_ok=True)
            folder = os.path.join(path, 'partial_dependece_isolate')
            plt.savefig(os.path.join(folder, f'pdp_{feature}.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
            
        except (IndexError, ValueError) as e:
            print(f"Skipping feature {feature} due to error: {e}")
            continue


def plot_interactions(booster: gpb.Booster, X: pd.DataFrame, path: Optional[str]) -> None:
    """Plot interaction plots for all pairs of features for the GPBoost model."""

    feature_pairs = list(combinations(X.columns, 2))
    
    for i, (feature1, feature2) in enumerate(feature_pairs):
        try:
            # Check if both features have enough unique values
            if X[feature1].nunique() < 2:
                print(f"Skipping interaction {feature1}-{feature2}: {feature1} has insufficient unique values ({X[feature1].nunique()})")
                continue
            if X[feature2].nunique() < 2:
                print(f"Skipping interaction {feature1}-{feature2}: {feature2} has insufficient unique values ({X[feature2].nunique()})")
                continue
                
            interact = pdp.PDPInteract(model=booster, df=X.copy(), model_features=X.columns,
                                        features=[feature1, feature2],
                                        feature_names=[feature1, feature2],
                                        n_classes=0, predict_kwds={"ignore_gp_model": True})
            fig, axes = interact.plot(engine='matplotlib', plot_type='contour', figsize=(6, 8))

            os.makedirs(os.path.join(path, 'partial_dependece_interact'), exist_ok=True)
            folder = os.path.join(path, 'partial_dependece_interact')
            plt.savefig(os.path.join(folder, f'pdp_{feature1}_{feature2}.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
            
        except (IndexError, ValueError) as e:
            print(f"Skipping interaction {feature1}-{feature2} due to error: {e}")
            continue

def plot_shap_values_with_avg(classification: bool, X: pd.DataFrame, path: Optional[str]) -> None:
    # SHAP values
    os.makedirs(os.path.join(path, 'shap_values_pub'), exist_ok=True)
    folder = os.path.join(path, 'shap_values_pub')
    if classification:
        task = 'classification'
    else:
        task = 'regression'
    shap_values = pd.read_csv(f'Results_Model/gp_boosting/{task}_try_num_leaves/shap_values/shap_values.csv')
    feature_names = [
    a + ": " + str(b) for a,b in zip(X.columns, np.abs(shap_values.values).mean(0).round(2))
]
    shap_values = shap_values.to_numpy()
    print(feature_names)
    shap.summary_plot(shap_values, X, plot_size=(12, 3), show=False, feature_names=feature_names)
    #plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'shap_summary_plot_{task}.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
def plot_shap_values(booster: gpb.Booster, X: pd.DataFrame, path: Optional[str]) -> None:
    """Plot SHAP values for the GPBoosting model."""
    # SHAP values
    os.makedirs(os.path.join(path, 'shap_values'), exist_ok=True)
    folder = os.path.join(path, 'shap_values')

    title = 'classification' if 'classification' in path else 'regression'

    plt.clf()
    shap_values = shap.TreeExplainer(booster).shap_values(X)

    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_values_df.to_csv(os.path.join(folder, 'shap_values.csv'), index=False)
    feature_names = [
    a + ": " + str(b) for a,b in zip(X.columns, np.abs(shap_values.values).mean(0).round(2))
]
    shap.summary_plot(shap_values, X, plot_size=(12, 3), show=False, feature_names=feature_names)
#    plt.title(f'SHAP values for {title} tasks', fontsize=16, loc='center')
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.tight_layout()

    plt.savefig(os.path.join(folder, 'shap_summary_plot.pdf'), bbox_inches='tight', dpi=300)

    # for feature in X.columns:
    #     shap.dependence_plot(feature, shap_values, X,show=False)
    #     plt.tight_layout()

    #     plt.savefig(os.path.join(folder, f"shap_dependence_plot_{feature}.pdf"), bbox_inches='tight', dpi=300)

    # # SHAP interaction values
    # shap_interaction_values = shap.TreeExplainer(booster).shap_interaction_values(shap_values)
    # shap.summary_plot(shap_interaction_values, X)
    # plt.tight_layout()

    # plt.savefig(os.path.join(folder, 'shap_interaction_plot_gpboost.pdf'), bbox_inches='tight', dpi=300)

    # feature_pairs = list(combinations(X.columns, 2))

    # for (feature1, feature2) in feature_pairs:
    #     shap.dependence_plot((feature1, feature2), shap_interaction_values, X, display_features=X)
    #     plt.tight_layout()

    #     plt.savefig(os.path.join(folder, f"shap_interaction_plot_{feature1}_{feature2}_gpboost.pdf"), bbox_inches='tight', dpi=300)


def main():
    param_combinations = [
        {'setting': 'try_num_leaves','try_num_leaves': True, 'try_max_depth': False, 'try_joint': False, 'try_num_iter': False},
        # {'setting': 'try_max_depth','try_num_leaves': False, 'try_max_depth': True, 'try_joint': False, 'try_num_iter': False},
        #{'setting': 'try_joint','try_num_leaves': False, 'try_max_depth': False, 'try_joint': True, 'try_num_iter': False}
        # {'setting': 'try_num_iter','try_num_leaves': True, 'try_max_depth': False, 'try_joint': False, 'try_num_iter': True},
    ]

    for classification in [False, True]:
        # for params in param_combinations:
            # booster, X, path = assess_gp_boosting(params['setting'],
            #     classification=classification,
            #     try_num_leaves=params['try_num_leaves'],
            #     try_max_depth=params['try_max_depth'],
            #     try_joint=params['try_joint'],
            #     try_num_iter=params['try_num_iter']
            # )
        X = get_data(classification)
        path = 'Additional_Shap_Plot'
        plot_shap_values_with_avg(classification=classification, X=X, path=path)


if __name__ == '__main__':
    main()