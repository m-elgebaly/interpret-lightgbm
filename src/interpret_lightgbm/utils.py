"""Utility helpers used across the package."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, accuracy_score, r2_score, log_loss,
    precision_score, recall_score, f1_score
)

from typing import Callable, Optional, Union, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Optional, Dict, Union
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_object_dtype
from pandas import CategoricalDtype
import math


from scipy.special import expit
from collections import defaultdict
from tqdm.auto import tqdm

# --- Helper Functions (Verified and Correct) ---

def _get_task_type(model_class: type) -> str:
    """Determines the task type from the model class."""
    if issubclass(model_class, RegressorMixin):
        return "regression"
    elif issubclass(model_class, ClassifierMixin):
        return "classification"
    raise ValueError("model_class must be a scikit-learn compatible classifier or regressor.")

def _calculate_classification_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculates precision, recall, and F1 score for classification tasks."""
    if num_classes == 2:
        y_pred = (y_pred_raw > 0).astype(int)
        average_method = 'binary'
    else:
        y_pred = np.argmax(y_pred_raw, axis=1)
        average_method = 'macro'
    
    precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
    
    return {"precision": precision, "recall": recall, "f1_score": f1}

def _calculate_r2_score(y_true: np.ndarray, y_pred_raw: np.ndarray, task_type: str, y_train_for_null: np.ndarray, num_classes: int) -> float:
    """Calculates R-squared for regression or McFadden's Pseudo R-squared for classification."""
    if task_type == "regression":
        return r2_score(y_true, y_pred_raw)

    if num_classes == 2:
        probabilities = 1 / (1 + np.exp(-y_pred_raw))
        probabilities = np.vstack([1 - probabilities, probabilities]).T
    else:
        probabilities = np.exp(y_pred_raw - np.max(y_pred_raw, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    unique_labels = np.unique(np.concatenate((y_true, y_train_for_null)))
    ll_full = log_loss(y_true, probabilities, normalize=False, labels=unique_labels)
    
    class_counts = np.bincount(y_train_for_null, minlength=num_classes)
    null_probs_vector = np.clip(class_counts / len(y_train_for_null), 1e-15, 1 - 1e-15)
    null_probs_matrix = np.tile(null_probs_vector, (len(y_true), 1))
    ll_null = log_loss(y_true, null_probs_matrix, normalize=False, labels=unique_labels)
    
    return 1 - (ll_full / ll_null) if ll_null != 0 else np.inf

def _run_staged_training_for_split(
    model_class: type,
    X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame, y_eval: np.ndarray,
    task_type: str, base_params: Dict[str, Any], depth_params: Dict[int, Dict[str, Any]],
    verbose: bool
) -> Tuple[np.ndarray, Dict[int, Any], Dict[int, Any]]:
    """
    Runs the staged training for a single fold, returning final predictions, models, and contributions.
    """
    num_classes = len(np.unique(np.concatenate((y_train, y_eval))))
    models, contributions = {}, {}
    previous_model = None

    if verbose:
        header = (f"{'Depth':<7} | {'Train F1':<12} | {'Eval F1':<12} | {'Train Precision':<15} | {'Eval Precision':<15} | "
                  f"{'Train Recall':<12} | {'Eval Recall':<12} | {'Train R2':<12} | {'Eval R2':<12}")
        if task_type == 'regression':
            header = f"{'Depth':<7} | {'Train MSE':<20} | {'Eval MSE':<20} | {'Train R2':<20} | {'Eval R2':<20}"
        print(header)
        print("-" * 120)

    # Step 0: Create and fit the initial bias model
    bias_params = base_params.copy()
    bias_params.update({"n_estimators": 1, "min_child_samples": len(X_train), "min_gain_to_split": 1e9})
    if task_type == 'classification':
        bias_params['objective'] = 'binary' if num_classes == 2 else 'multiclass'
        if num_classes > 2: bias_params['num_class'] = num_classes

    bias_model = model_class(**bias_params)
    bias_model.fit(X_train, y_train)
    models[0] = bias_model
    contributions[0] = bias_model.predict(X_eval, pred_contrib=True)
    previous_model = bias_model

    # Iterate through specified depths, building on previous models
    for depth in sorted(depth_params.keys()):
        params = base_params.copy()
        params.update(depth_params[depth])
        params['max_depth'] = depth
        if task_type == 'classification':
            params['objective'] = 'binary' if num_classes == 2 else 'multiclass'
            if num_classes > 2: params['num_class'] = num_classes

        model = model_class(**params)
        model.fit(X_train, y_train, init_model=previous_model)

        models[depth] = model
        contributions[depth] = model.predict(X_eval, pred_contrib=True)
        previous_model = model

        if verbose:
            # Verbose printing logic (unchanged)
            if task_type == 'classification':
                train_preds_raw = model.predict(X_train, raw_score=True)
                eval_preds_raw = model.predict(X_eval, raw_score=True)
                train_metrics = _calculate_classification_metrics(y_train, train_preds_raw, num_classes)
                eval_metrics = _calculate_classification_metrics(y_eval, eval_preds_raw, num_classes)
                train_r2 = _calculate_r2_score(y_train, train_preds_raw, task_type, y_train, num_classes)
                eval_r2 = _calculate_r2_score(y_eval, eval_preds_raw, task_type, y_train, num_classes)
                print(f"{depth:<7} | {train_metrics['f1_score']:<12.4f} | {eval_metrics['f1_score']:<12.4f} | "
                      f"{train_metrics['precision']:<15.4f} | {eval_metrics['precision']:<15.4f} | "
                      f"{train_metrics['recall']:<12.4f} | {eval_metrics['recall']:<12.4f} | "
                      f"{train_r2:<12.4f} | {eval_r2:<12.4f}")
            else:
                train_preds, eval_preds = model.predict(X_train), model.predict(X_eval)
                train_mse, eval_mse = mean_squared_error(y_train, train_preds), mean_squared_error(y_eval, eval_preds)
                train_r2, eval_r2 = r2_score(y_train, train_preds), r2_score(y_eval, eval_preds)
                print(f"{depth:<7} | {train_mse:<20.4f} | {eval_mse:<20.4f} | {train_r2:<20.4f} | {eval_r2:<20.4f}")
    
    # Return raw predictions for the final model to allow for flexible metric calculation
    final_eval_preds = previous_model.predict(X_eval, raw_score=True if task_type == 'classification' else False)
    return final_eval_preds, models, contributions

# --- Main User-Facing Function ---

def staged_depth_boosting(
    model_class: type,
    X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
    base_params: Dict[str, Any],
    depth_params: Dict[int, Dict[str, Any]],
    eval_metric: Optional[Callable] = None,
    cv_object: Optional[KFold] = None,
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    verbose: bool = True
) -> Union[Tuple[Dict[str, float], List, List], Tuple[float, Dict, Dict]]:
    
    if isinstance(y, pd.Series): y = y.values
    if y_test is not None and isinstance(y_test, pd.Series): y_test = y_test.values

    task_type = _get_task_type(model_class)
    if not depth_params: raise ValueError("'depth_params' cannot be empty.")

    if eval_metric is None:
        eval_metric = mean_squared_error if task_type == 'regression' else accuracy_score
        if verbose: print(f"Primary evaluation metric not specified. Defaulting to `{eval_metric.__name__}`.")
    
    X = pd.DataFrame(X) # Ensure X is a DataFrame
    if X_test is not None: X_test = pd.DataFrame(X_test)

    params_for_lgbm = base_params.copy()
    params_for_lgbm.setdefault('verbose', -1)
    
    num_classes = len(np.unique(y))

    def _get_final_score(y_true, y_pred_raw, current_num_classes):
        """Helper to compute the final score using the specified eval_metric."""
        if task_type == 'classification':
            y_pred_class = (y_pred_raw > 0).astype(int) if current_num_classes == 2 else np.argmax(y_pred_raw, axis=1)
            # This wrapper handles metrics that need an `average` parameter
            if eval_metric.__name__ in ['f1_score', 'precision_score', 'recall_score']:
                avg = 'binary' if current_num_classes == 2 else 'macro'
                return eval_metric(y_true, y_pred_class, average=avg, zero_division=0)
            return eval_metric(y_true, y_pred_class)
        return eval_metric(y_true, y_pred_raw)

    if cv_object is not None:
        if verbose: print(f"Running Cross-Validation with {cv_object.get_n_splits()} folds...")
        all_scores, all_models, all_contribs = [], [], []

        for fold, (train_idx, val_idx) in enumerate(cv_object.split(X, y)):
            if verbose: print(f"\n===== Fold {fold+1} =====")
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            fold_num_classes = len(np.unique(y_train))
            
            final_val_preds, models, contributions = _run_staged_training_for_split(
                model_class, X_train, y_train, X_val, y_val, task_type,
                params_for_lgbm, depth_params, verbose=verbose
            )
            score = _get_final_score(y_val, final_val_preds, fold_num_classes)
            all_scores.append(score)
            all_models.append(models)
            all_contribs.append(contributions)
        
        # For CV, returning a dictionary of stats is more informative
        results = {"mean_score": np.mean(all_scores), "std_score": np.std(all_scores)}
        if verbose:
            print("\n===== Cross-Validation Overall Summary =====")
            print(f"Mean Score ({eval_metric.__name__}): {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            
        return results, all_models, all_contribs

    elif X_test is not None and y_test is not None:
        if verbose: print("Running in Train-Test Split mode...")
        final_test_preds, models, contributions = _run_staged_training_for_split(
            model_class, X, y, X_test, y_test, task_type,
            params_for_lgbm, depth_params, verbose=verbose
        )
        final_score = _get_final_score(y_test, final_test_preds, num_classes)
        
        if verbose:
            print("\n===== Train-Test Split Summary =====")
            print(f"Final Test Score ({eval_metric.__name__}): {final_score:.4f}")
        
        # *** MODIFIED RETURN: Return the direct score, models, and contributions ***
        return final_score, models, contributions
    
    else:
        raise ValueError("You must provide either 'cv_object' or both 'X_test' and 'y_test'.")
    


def _plot_contributions(
    contributions: Union[Dict, List[Dict]],
    X_eval: pd.DataFrame,
    depth: int,
    plot_type: str = 'beeswarm',
    class_index: Optional[int] = None,
    max_display: int = 15,
    categorical_threshold: int = 10,
    cmap_pos: str = 'Greens',
    cmap_neg: str = 'Reds',
):
    """
    Plots feature contributions from models in 'beeswarm' or 'scatter' mode.
    Handles contributions as either pandas DataFrames or numpy ndarrays.

    Args:
        contributions (Union[Dict, List[Dict]]): Contributions from a model.
        X_eval (pd.DataFrame): Evaluation dataset used for contributions.
        depth (int): The specific depth/stage to plot.
        plot_type (str): The type of plot: 'beeswarm' or 'scatter'.
        class_index (Optional[int]): The class index for multiclass models.
        max_display (int): Maximum number of features to display.
        categorical_threshold (int): Threshold to treat numeric features as categorical.
        cmap_pos (str): Colormap for positive values.
        cmap_neg (str): Colormap for negative values.
    """
    if plot_type not in ['beeswarm', 'scatter']:
        raise ValueError("plot_type must be either 'beeswarm' or 'scatter'")

    # --- Step 1: Data Preparation ---
    if isinstance(contributions, list):  # Multiclass classification
        if class_index is None:
            raise ValueError("Must provide 'class_index' for multiclass models.")
        if class_index >= len(contributions):
            raise ValueError(f"class_index {class_index} is out of bounds.")
        contrib_data = contributions[class_index]
        title_class_info = f" for Class {class_index}"
    else:  # Regression or binary classification
        contrib_data = contributions
        title_class_info = ""

    if depth not in contrib_data:
        raise ValueError(f"Depth {depth} not found. Available depths: {list(contrib_data.keys())}")

    shap_values_raw = contrib_data[depth]

    if isinstance(shap_values_raw, np.ndarray):
        num_expected_cols = X_eval.shape[1]
        if shap_values_raw.shape[1] == num_expected_cols + 1:
            feature_names = list(X_eval.columns) + ['base']
        elif shap_values_raw.shape[1] == num_expected_cols:
            feature_names = list(X_eval.columns)
        else:
            raise ValueError("SHAP values array shape does not match X_eval columns.")
        shap_df = pd.DataFrame(shap_values_raw, columns=feature_names, index=X_eval.index)
    elif isinstance(shap_values_raw, pd.DataFrame):
        shap_df = shap_values_raw
    else:
        raise TypeError(f"Contributions must be a pandas DataFrame or numpy ndarray.")

    shap_values_df = shap_df.drop(columns='base', errors='ignore')
    
    mean_abs_shap = shap_values_df.abs().mean().sort_values(ascending=False)
    top_feature_names = mean_abs_shap.iloc[:max_display].index.tolist()
    
    shap_top_features_df = shap_values_df[top_feature_names]
    X_top_features = X_eval[top_feature_names]

    # --- Step 2: Plotting ---
    if plot_type == 'beeswarm':
        fig, ax = plt.subplots(figsize=(12, max_display * 0.5 + 2))
        legend_elements = []
        
        for i, feature_name in enumerate(top_feature_names):
            shap_i = shap_top_features_df[feature_name].values
            x_values_i = X_top_features[feature_name]
            
            # Updated check for categorical features
            is_categorical = (
                isinstance(x_values_i.dtype, CategoricalDtype) or
                is_string_dtype(x_values_i.dtype) or
                is_object_dtype(x_values_i.dtype) or
                (is_numeric_dtype(x_values_i.dtype) and x_values_i.nunique() < categorical_threshold)
            )
            y_jitter = np.random.uniform(-0.2, 0.2, size=len(shap_i))

            if is_categorical:
                categories = x_values_i.astype('category').cat.categories
                cmap_cat = plt.get_cmap('tab10')
                color_map = {cat: cmap_cat(j % cmap_cat.N) for j, cat in enumerate(categories)}

                # --- FIX: Iterate over categories to plot instead of using .map() ---
                for cat, color in color_map.items():
                    mask = (x_values_i == cat)
                    ax.scatter(shap_i[mask], i + y_jitter[mask], color=color, s=25, alpha=0.8, zorder=2)
                
                # Add to legend
                legend_elements.append(Line2D([0],[0], color='none', label=f'\nFeature: {feature_name}'))
                for cat, color in color_map.items():
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'  {cat}', markerfacecolor=color, markersize=8))
            else:
                norm = plt.Normalize(x_values_i.min(), x_values_i.max())
                cmap = plt.get_cmap('seismic')
                point_colors = cmap(norm(x_values_i))
                ax.scatter(shap_i, i + y_jitter, c=point_colors, s=20, alpha=0.7, zorder=2)

        ax.axvline(x=0, color='grey', linestyle='--', lw=1, zorder=1)
        ax.set_yticks(range(len(top_feature_names)))
        ax.set_yticklabels(top_feature_names)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Contribution (SHAP value)", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Beeswarm Contribution Plot (Depth {depth}{title_class_info})", fontsize=14, pad=20)
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.legend(handles=legend_elements, title='Color Legend', bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=3)
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.show()

    elif plot_type == 'scatter':
        ncols = min(4, len(top_feature_names))
        nrows = math.ceil(len(top_feature_names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
        axes = axes.flatten()

        for i, feature_name in enumerate(top_feature_names):
            ax = axes[i]
            shap_i = shap_top_features_df[feature_name].values
            x_values_i = X_top_features[feature_name]

            is_categorical = (
                isinstance(x_values_i.dtype, CategoricalDtype) or
                is_string_dtype(x_values_i.dtype) or
                is_object_dtype(x_values_i.dtype) or
                (is_numeric_dtype(x_values_i.dtype) and x_values_i.nunique() < categorical_threshold)
            )

            if is_categorical:
                categories = x_values_i.astype('category').cat.categories
                cmap_cat = plt.get_cmap('tab10')
                color_map = {cat: cmap_cat(j % cmap_cat.N) for j, cat in enumerate(categories)}
                
                # --- FIX: Iterate over categories to plot instead of using .map() ---
                for j, cat in enumerate(categories):
                    mask = (x_values_i == cat)
                    x_jitter = j + np.random.uniform(-0.15, 0.15, size=mask.sum())
                    ax.scatter(x_jitter, shap_i[mask], color=color_map[cat], s=25, alpha=0.7)
                
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
            else:
                cmap = plt.get_cmap('viridis')
                norm = plt.Normalize(x_values_i.min(), x_values_i.max())
                ax.scatter(x_values_i, shap_i, c=x_values_i, cmap=cmap, norm=norm, s=20, alpha=0.7)
            
            ax.axhline(0, color='grey', linestyle='--', lw=1)
            ax.set_title(feature_name)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("SHAP Value")

        for j in range(len(top_feature_names), len(axes)):
            axes[j].set_visible(False)
            
        fig.suptitle(f"Feature Value vs. Contribution (Depth {depth}{title_class_info})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()



def _get_feature_contributions(model, bias_model, X_test, raw=True):
    """
    Calculates feature contributions, with an option to convert them to probability space.

    When raw=False for classification, contributions are returned as sequential probability
    changes, where the sum of a row's contributions equals the final probability prediction.

    Parameters:
    - model: The main trained LightGBM model.
    - bias_model: A simple, one-tree "stump" model representing the bias.
    - X_test: DataFrame containing the test samples.
    - raw (bool): If True, returns contributions and predictions in log-odds (for classification).
                  If False (for classification), converts contributions and predictions to
                  probability space based on sequential tree evaluation.

    Returns:
    - all_tree_contributions (list): Log-odds contribution of each tree for each sample.
    - combo_df (DataFrame): A DataFrame of contributions. For raw=True, values are in
                            log-odds. For raw=False (classification), the 'bias' column
                            is the baseline probability and other columns are the marginal
                            probability changes.
    - predictions (np.array): Final predictions for each sample, with bias included.
    """
    # --- 1. Extract Bias Contribution (Log-Odds) ---
    bias_value = 0.0
    if bias_model:
        try:
            bias_model_dump = bias_model.booster_.dump_model()
            bias_value = bias_model_dump['tree_info'][0]['tree_structure']['leaf_value']
        except (KeyError, IndexError):
            print("Warning: Could not extract bias value. Defaulting to 0.")

    # --- 2. Extract Model Information ---
    trees = model.booster_.dump_model()['tree_info']
    feature_names = model.booster_.feature_name()
    leaf_indices = model.predict(X_test, pred_leaf=True)
    is_classification = getattr(model, '_estimator_type', None) == 'classifier'

    # --- 3. Calculate Log-Odds Contribution For Each Tree, For Each Sample (Preserving Order) ---
    all_tree_contributions = []

    iterator = tqdm(range(len(X_test)), desc="Calculating Contributions")
    
    for sample_idx in iterator:
        tree_contributions_for_sample = []
        for tree_idx, tree in enumerate(trees):
            leaf_idx = leaf_indices[sample_idx, tree_idx]
            path_features = set()

            def traverse(node):
                if 'split_feature' not in node:
                    return ('leaf_index' in node and node['leaf_index'] == leaf_idx) or \
                           ('leaf_index' not in node and leaf_idx == 0)

                split_feature_name = feature_names[node['split_feature']]
                sample_value = X_test[split_feature_name].iloc[sample_idx]
                
                go_left = False
                if pd.isna(sample_value):
                    go_left = node['default_left']
                elif node['decision_type'] == '==':
                    thresholds = set(node['threshold'].split('||'))
                    sample_code = str(X_test[split_feature_name].cat.codes.iloc[sample_idx]) if isinstance(X_test[split_feature_name].dtype, pd.CategoricalDtype) else str(sample_value)
                    go_left = sample_code in thresholds
                else:
                    go_left = sample_value <= node['threshold']

                if traverse(node['left_child'] if go_left else node['right_child']):
                    path_features.add(split_feature_name)
                    return True
                return False

            traverse(tree['tree_structure'])

            def find_leaf_value(node):
                if 'split_feature' not in node:
                    if ('leaf_index' in node and node['leaf_index'] == leaf_idx) or ('leaf_index' not in node and leaf_idx == 0):
                        return node.get('leaf_value', 0.0)
                    return None
                for child in ['left_child', 'right_child']:
                    val = find_leaf_value(node[child])
                    if val is not None: return val
                return None
            
            leaf_value = find_leaf_value(tree['tree_structure'])
            tree_contributions_for_sample.append((tree_idx, leaf_value, path_features))
        all_tree_contributions.append(tree_contributions_for_sample)

    # --- 4. Handle Output based on 'raw' parameter ---
    if is_classification and not raw:
        # --- A. CONVERT TO PROBABILITY SPACE SEQUENTIALLY ---
        prob_data = []
        for sample_tree_contribs in all_tree_contributions:
            prob_contribs_for_sample = defaultdict(float)
            
            # Start with the bias as the baseline
            running_log_odds = 0
            prob_contribs_for_sample['bias'] = expit(bias_value)

            # Sequentially add the probability change from each tree
            for _, leaf_value, path_features in sample_tree_contribs:
                if leaf_value is not None:
                    prob_before = expit(running_log_odds)
                    # Update the accumulator with the current tree's contribution
                    running_log_odds += leaf_value
                    prob_after = expit(running_log_odds)
                    
                    prob_change = prob_after - prob_before
                    
                    # Attribute the probability change to the features used in this tree's path
                    if path_features:
                        key = '&'.join(sorted(list(path_features)))
                        prob_contribs_for_sample[key] += prob_change
            
            prob_data.append(prob_contribs_for_sample)

        combo_df = pd.DataFrame(prob_data).fillna(0)
        
        # Calculate final predictions from the final log_odds
        total_log_odds = np.array([bias_value + sum(c[1] for c in s if c[1] is not None) for s in all_tree_contributions])
        predictions = expit(total_log_odds)

    else:
        # --- B. AGGREGATE LOG-ODDS CONTRIBUTIONS ---
        log_odds_data = []
        for sample_contribs in all_tree_contributions:
            sample_combinations = defaultdict(float)
            for _, leaf_value, path_features in sample_contribs:
                if leaf_value is not None and path_features:
                    key = '&'.join(sorted(list(path_features)))
                    sample_combinations[key] += leaf_value
            log_odds_data.append(sample_combinations)

        combo_df = pd.DataFrame(log_odds_data).fillna(0)
        combo_df['bias'] = bias_value
        predictions = combo_df.sum(axis=1).values

    # Ensure consistent column order with 'bias' first
    cols = ['bias'] + [col for col in combo_df.columns if col != 'bias']
    combo_df = combo_df[cols]

    return all_tree_contributions, combo_df, predictions


import pandas as pd
import numpy as np

def _get_predictions_model(model_or_preds, X):
    """
    Helper function to retrieve predictions.
    - If `model_or_preds` has a `.predict()` method, call it with `X`.
    - Otherwise, assume it's already a numpy array or pandas Series and return it.
    """
    if hasattr(model_or_preds, "predict"):
        return model_or_preds.predict(X)
    return np.array(model_or_preds)

def _process_contribution_with_complex(
    df, 
    y_test=None, 
    pred_simple=None, 
    pred_complex=None, 
    X_test=None,
    is_classification=False,
    metric_fn=None,
    min_abs_sum=None,
    reduction_ratio=1.0
):
    """
    Processes a single contribution DataFrame:
    1. Optionally removes columns whose total absolute sum is below a threshold (`min_abs_sum`).
    2. Optionally adds a 'complex_additive' column from the difference between complex and simple model predictions.
    3. Optionally computes a performance metric from reconstructed predictions.
    4. Calculates and displays ranking statistics for 'complex_additive'.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution DataFrame with features as columns and samples as rows.
    y_test : array-like, optional
        True target values for evaluating the metric function.
    pred_simple : array-like or model, optional
        Predictions (or model) for the simple model.
    pred_complex : array-like or model, optional
        Predictions (or model) for the complex model.
    X_test : array-like, optional
        Test features required if passing models instead of predictions.
    is_classification : bool, default=False
        Flag to indicate if the task is classification (unused in current logic).
    metric_fn : callable, optional
        Metric function accepting (y_true, y_pred).
    min_abs_sum : float, optional
        Minimum total absolute sum for a column to be retained.
    reduction_ratio : float, default=1.0
        Scaling factor for the 'complex_additive' column.

    Returns
    -------
    processed_df : pd.DataFrame
        Processed DataFrame with optional filtering and added columns.
    report : dict
        Summary report containing counts, metric value, and ranking stats.
    """

    if df.empty:
        return pd.DataFrame(), {}

    ref_index = df.index  # Store index for alignment checks

    # --- Column filtering based on absolute sum ---
    removed_count = 0
    if min_abs_sum is not None:
        abs_sums = df.abs().sum(axis=0)
        keep_cols = abs_sums[abs_sums >= min_abs_sum].index
        removed_count = df.shape[1] - len(keep_cols)
        df = df[keep_cols]

    # --- Add complex_additive column ---
    added_count = 0
    if pred_simple is not None and pred_complex is not None:
        if X_test is None and (hasattr(pred_simple, "predict") or hasattr(pred_complex, "predict")):
            raise ValueError("X_test is required when passing models for pred_simple/pred_complex.")

        preds_simple = _get_predictions_model(pred_simple, X_test)
        preds_complex = _get_predictions_model(pred_complex, X_test)

        diff = pd.Series(preds_complex, index=ref_index) - pd.Series(preds_simple, index=ref_index)
        df["complex_additive"] = diff * reduction_ratio
        added_count += 1

    # --- Compute metric if requested ---
    metric_value = None
    if metric_fn is not None and y_test is not None:
        # Prediction reconstructed from sum of contributions
        y_pred_from_contribs = df.sum(axis=1)
        metric_value = metric_fn(y_test, y_pred_from_contribs)

    # --- Ranking stats for complex_additive ---
    rank_stats = {}
    if "complex_additive" in df.columns:
        abs_contribs = df.abs()
        ranks = abs_contribs.rank(axis=1, ascending=False, method="min")
        complex_ranks = ranks["complex_additive"]

        rank_stats = {
            "complex_additive_mean_rank": round(complex_ranks.mean()),
            "complex_additive_25th_rank": round(complex_ranks.quantile(0.25)),
            "complex_additive_75th_rank": round(complex_ranks.quantile(0.75)),
        }

        # Top 10 mean absolute contributions (including complex_additive if present)
        mean_abs_contribs = abs_contribs.mean().sort_values(ascending=False)
        top_10_features = mean_abs_contribs.head(10).index

        top_10_summary = []
        for feat in top_10_features:
            avg_rank = ranks[feat].mean()
            top_10_summary.append((feat, mean_abs_contribs[feat], avg_rank))

        rank_stats["top_10_features"] = pd.DataFrame(
            top_10_summary, columns=["Feature", "MeanAbsContribution", "MeanRank"]
        )

    # --- Final report ---
    report = {
        "columns_removed": removed_count,
        "columns_added": added_count,
        "metric_value": metric_value,
        **rank_stats
    }

    # --- Output report ---
    print(report)


    return df, report

# --- Auto-generate __all__ at the very bottom ---
import inspect
__all__ = [
    name
    for name, obj in globals().items()
    if inspect.isfunction(obj) or inspect.isclass(obj)
]