"""Progressive LightGBM models: build shallow→deeper trees to enable exact contribution tracking."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .utils import *

# --- NEW: Configuration Dataclass ---
@dataclass
class _ProgressiveConfig:
    """A read-only container for model configuration parameters."""
    # Use field(default_factory=...) for mutable types like dicts
    base_params: Dict[str, Any] #= field(default_factory=dict)
    depth_params: Dict[int, Dict[str, Any]] #= field(default_factory=dict)
    eval_metric: Optional[Callable] = None
    cv_object: Optional[KFold] = None
    verbose: bool = True
        
    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.depth_params:
            raise ValueError("'depth_params' cannot be empty.")

# --- MODIFIED: Base Class Wrapper ---

class _ProgressiveLGBMBase(BaseEstimator):
    """
    Base class for Progressive LightGBM models.
    
    Warning: This class should not be used directly. Use ProgressiveLGBMClassifier
    or ProgressiveLGBMRegressor instead.
    """
    def __init__(self,
                 base_params: Dict[str, Any],
                 depth_params: Dict[int, Dict[str, Any]],
                 eval_metric: Optional[Callable] = None,
                 cv_object: Optional[KFold] = None,
                 verbose: bool = True):
        # All parameters are now stored in the immutable config object
        self.config = _ProgressiveConfig(
            base_params=base_params,
            depth_params=depth_params,
            eval_metric=eval_metric,
            cv_object=cv_object,
            verbose=verbose
        )
        self.base_params = base_params
        self.depth_params = depth_params
        self.eval_metric = eval_metric
        self.cv_object = cv_object
        self.verbose = verbose


    def _fit(self,
             model_class: Union[type[lgb.LGBMClassifier], type[lgb.LGBMRegressor]],
             X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series],
             X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
             y_test: Optional[Union[np.ndarray, pd.Series]] = None):
        
        if isinstance(y, pd.Series): y = y.values
        if y_test is not None and isinstance(y_test, pd.Series): y_test = y_test.values

        task_type = _get_task_type(model_class)
        
        # Now we get parameters from the config object
        eval_metric = self.config.eval_metric
        if eval_metric is None:
            eval_metric = mean_squared_error if task_type == 'regression' else accuracy_score
            if self.config.verbose: 
                print(f"Primary evaluation metric not specified. Defaulting to `{eval_metric.__name__}`.")
        
        X = pd.DataFrame(X)
        if X_test is not None: X_test = pd.DataFrame(X_test)

        params_for_lgbm = self.config.base_params.copy()
        params_for_lgbm.setdefault('verbose', -1)
        
        num_classes = len(np.unique(y))

        def _get_final_score(y_true, y_pred_raw, current_num_classes):
            if task_type == 'classification':
                y_pred_class = (y_pred_raw > 0.5).astype(int) if current_num_classes == 2 else np.argmax(y_pred_raw, axis=1)
                if eval_metric.__name__ in ['f1_score', 'precision_score', 'recall_score']:
                    avg = 'binary' if current_num_classes == 2 else 'macro'
                    return eval_metric(y_true, y_pred_class, average=avg, zero_division=0)
                return eval_metric(y_true, y_pred_class)
            return eval_metric(y_true, y_pred_raw)

        if self.config.cv_object is not None:
            if self.config.verbose: print(f"Running Cross-Validation with {self.config.cv_object.get_n_splits()} folds...")
            all_scores, all_models, all_contribs = [], [], []

            for fold, (train_idx, val_idx) in enumerate(self.config.cv_object.split(X, y)):
                if self.config.verbose: print(f"\n===== Fold {fold+1} =====")
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_val, y_val = X.iloc[val_idx], y[val_idx]
                
                final_val_preds, models, contributions = _run_staged_training_for_split(
                    model_class, X_train, y_train, X_val, y_val, task_type,
                    params_for_lgbm, self.config.depth_params, verbose=self.config.verbose
                )
                score = _get_final_score(y_val, final_val_preds, len(np.unique(y_train)))
                all_scores.append(score)
                all_models.append(models)
                all_contribs.append(contributions)
            
            self.final_score_ = np.mean(all_scores)
            self.std_score_ = np.std(all_scores)
            self.models_ = all_models
            self.contributions_ = all_contribs
            
            if self.config.verbose:
                print("\n===== Cross-Validation Overall Summary =====")
                print(f"Mean Score ({eval_metric.__name__}): {self.final_score_:.4f} (+/- {self.std_score_:.4f})")
                
        elif X_test is not None and y_test is not None:
            if self.config.verbose: print("Running in Train-Test Split mode...")
            final_test_preds, models, contributions = _run_staged_training_for_split(
                model_class, X, y, X_test, y_test, task_type,
                params_for_lgbm, self.config.depth_params, verbose=self.config.verbose
            )
            self.final_score_ = _get_final_score(y_test, final_test_preds, num_classes)
            self.models_ = models
            self.contributions_ = contributions
            
            final_depth = max(models.keys())
            self.final_model_ = models[final_depth]

            if self.config.verbose:
                print("\n===== Train-Test Split Summary =====")
                print(f"Final Test Score ({eval_metric.__name__}): {self.final_score_:.4f}")
        
        else:
            raise ValueError("You must provide either 'cv_object' during initialization or both 'X_test' and 'y_test' to the fit() method.")
        
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("predict() is only available when fit is called with X_test and y_test.")
        return self.final_model_.predict(X)

    def plot_contributions(
        self,
        X_eval: pd.DataFrame,
        depth: int,
        fold_index: Optional[int] = None,
        **kwargs):
        """
        Plots feature contributions from a fitted model stage.
    
        Args:
            X_eval (pd.DataFrame): The evaluation data that corresponds to the
                contributions. For a train-test split model, this is the X_test.
                For a CV model, this is the validation set for the chosen fold.
            depth (int): The specific model depth/stage to plot.
            fold_index (Optional[int]): If the model was fitted with cross-validation,
                this specifies which fold's contributions to plot. Must be provided
                for CV models.
            **kwargs: Additional keyword arguments to pass to the plotting function,
                such as 'plot_type', 'class_index', 'max_display', etc.
        """
        if not hasattr(self, 'contributions_'):
            raise RuntimeError("Model has not been fitted yet. Please call .fit() first.")
    
        # Case 1: Model was fitted with Cross-Validation
        if isinstance(self.contributions_, list):
            if fold_index is None:
                raise ValueError(
                    "Model was fitted with cross-validation. "
                    "Please provide a 'fold_index' to specify which fold to plot."
                )
            if not 0 <= fold_index < len(self.contributions_):
                raise IndexError(f"Invalid 'fold_index'. Must be between 0 and {len(self.contributions_) - 1}.")
            
            contrib_to_plot = self.contributions_[fold_index]
    
        # Case 2: Model was fitted with a single Train-Test Split
        else:
            if fold_index is not None:
                print("Warning: 'fold_index' is ignored as the model was not fitted with cross-validation.")
            contrib_to_plot = self.contributions_
        
        # Call the internal plotting function with the selected data
        _plot_contributions(
            contributions=contrib_to_plot,
            X_eval=X_eval,
            depth=depth,
            **kwargs
        )

    def get_feature_contributions(
        self,
        X_eval: pd.DataFrame,
        depth: int,
        fold_index: Optional[int] = None,
        raw: bool = True,
    ) -> Tuple[list, pd.DataFrame, np.ndarray]:
        """
        Calculates detailed feature contributions for each sample at a specific model depth.

        This method breaks down a prediction by showing how much each feature (or
        combination of features) contributes, starting from a base value.

        Args:
            X_eval (pd.DataFrame): The evaluation data for which to calculate contributions.
            depth (int): The specific model depth/stage to analyze.
            fold_index (Optional[int]): If the model was fitted with cross-validation,
                this specifies which fold's model to use. Must be provided for CV models.
            raw (bool): If True, returns contributions in log-odds (for classification) or
                as direct values (for regression). If False (for classification only),
                converts contributions to sequential probability changes.
            show_progress (bool): If True, displays a tqdm progress bar.

        Returns:
            A tuple containing:
            - all_tree_contributions (list): A nested list with the raw contribution
              of each tree for each sample.
            - combo_df (pd.DataFrame): A DataFrame where rows are samples and columns
              are features (or feature interactions), showing their total contribution.
            - predictions (np.ndarray): The final prediction for each sample, which should
              equal the sum of the contributions in combo_df.
        """
        if not hasattr(self, 'models_'):
            raise RuntimeError("Model has not been fitted yet. Please call .fit() first.")

        # Handle CV vs. single-split to get the right set of models
        if isinstance(self.models_, list):  # CV case
            if fold_index is None:
                raise ValueError("Model was fitted with cross-validation. Please provide a 'fold_index'.")
            if not 0 <= fold_index < len(self.models_):
                raise IndexError(f"Invalid 'fold_index'. Must be between 0 and {len(self.models_) - 1}.")
            models_to_use = self.models_[fold_index]
        else:  # Single-split case
            if fold_index is not None:
                print("Warning: 'fold_index' is ignored as the model was not fitted with cross-validation.")
            models_to_use = self.models_

        # Get the specific model and the bias model
        if depth not in models_to_use:
            raise ValueError(f"Depth {depth} not found. Available depths: {list(models_to_use.keys())}")
        if 0 not in models_to_use:
            print("Bias model (at depth 0) not found. The model may not have been fitted correctly.")
            bias_model = None
        else: 
            bias_model = models_to_use[0]
            
        model = models_to_use[depth]
        

        # Call the internal function and return its results
        return _get_feature_contributions(
            model=model,
            bias_model=bias_model,
            X_test=X_eval,
            raw=raw,
        )


        # --- NEW: User-Facing Delta Analysis Method ---
    def proccess_contribution_complex(
        self,
        contribution_df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        simple_model_depth: int,
        complex_model_depth: Optional[int] = None,
        fold_index: Optional[int] = None,
        metric_fn: Optional[Callable] = None,
        min_abs_sum: float = 0.0,
        reduction_ratio: float = 1,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Analyzes the change in contributions between a simple and a complex model stage.

        This method calculates the feature contributions of a simpler model and then adds a
        single 'complex_additive' feature representing the prediction difference to a
        more complex model. It helps quantify the impact of increasing model complexity.

        Args:
            X_eval (pd.DataFrame): The evaluation data used for the analysis.
            y_eval (Union[pd.Series, np.ndarray]): The true target values for evaluation.
            simple_model_depth (int): The depth of the simpler model to use as the baseline.
            complex_model_depth (Optional[int]): The depth of the more complex model.
                If None, defaults to the final, deepest model trained.
            fold_index (Optional[int]): If the model was fitted with cross-validation,
                this specifies which fold to analyze. Must be provided for CV models.
            metric_fn (Optional[Callable]): A metric function to evaluate the reconstructed
                predictions. If None, defaults to the 'eval_metric' used during fit.
            min_abs_sum (float): The minimum absolute sum for a feature's contributions
                to be included in the analysis. Defaults to 0.0 (include all).
            reduction_ratio (float): A scaling factor for the 'complex_additive' term.
                Defaults to 0.75.
            show_contrib_progress (bool): If True, displays a progress bar during the
                initial contribution calculation. Defaults to False.

        Returns:
            A tuple containing:
            - processed_df (pd.DataFrame): The processed contribution DataFrame, including
              the 'complex_additive' column.
            - report (dict): A dictionary containing the summary report that was printed.
        """
        if not hasattr(self, 'models_'):
            raise RuntimeError("Model has not been fitted yet. Please call .fit() first.")

        # Handle CV vs. single-split to get the right set of models
        if isinstance(self.models_, list):
            if fold_index is None: raise ValueError("Model was fitted with CV. Provide 'fold_index'.")
            if not 0 <= fold_index < len(self.models_): raise IndexError(f"Invalid 'fold_index'.")
            models_to_use = self.models_[fold_index]
        else:
            if fold_index is not None: print("Warning: 'fold_index' ignored...")
            models_to_use = self.models_

        # Resolve simple and complex models
        if simple_model_depth not in models_to_use: raise ValueError(f"Simple model depth {simple_model_depth} not found.")
        simple_model = models_to_use[simple_model_depth]

        if complex_model_depth is None:
            complex_model_depth = max(k for k in models_to_use if k > 0)
        if complex_model_depth not in models_to_use: raise ValueError(f"Complex model depth {complex_model_depth} not found.")
        complex_model = models_to_use[complex_model_depth]
        
        print(f"\nAnalyzing delta: Simple Model (depth={simple_model_depth}) vs. Complex Model (depth={complex_model_depth})")

        
        metric_to_use = metric_fn if metric_fn is not None else self.eval_metric
        is_classification = getattr(simple_model, '_estimator_type', None) == 'classifier'

        # Call the internal processing function
        processed_df, report = _process_contribution_with_complex(
            df=contribution_df, y_test=y_test, pred_simple=simple_model,
            pred_complex=complex_model, X_test=X_test, is_classification=is_classification,
            metric_fn=metric_to_use, min_abs_sum=min_abs_sum, reduction_ratio=reduction_ratio
        )
        return processed_df, report

class ProgressiveLGBMRegressor(RegressorMixin, _ProgressiveLGBMBase):
    """
    A LightGBM Regressor that builds models in stages of increasing `max_depth`.
    """
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_test: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Fits the staged-depth regression model.
        """
        super()._fit(lgb.LGBMRegressor, X, y, X_test, y_test)
        return self

class ProgressiveLGBMClassifier(ClassifierMixin, _ProgressiveLGBMBase):
    """
    A LightGBM Classifier that builds models in stages of increasing `max_depth`.
    """
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_test: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Fits the staged-depth classification model.
        """
        self.classes_ = np.unique(y)
        super()._fit(lgb.LGBMClassifier, X, y, X_test, y_test)
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("predict_proba() is only available when fit is called with X_test and y_test.")
        return self.final_model_.predict_proba(X)