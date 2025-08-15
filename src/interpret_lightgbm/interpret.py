import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from IPython.display import display


def prepare_local_force_data(
    contributions_df: pd.DataFrame,
    instance_index: any,
    global_base_value: Optional[float] = None,
    threshold_abs: float = 0.01
) -> Dict:
    """Prepares the data dictionary for plotting and analysis."""

    if instance_index not in contributions_df.index:
        raise ValueError(f"Index {instance_index} not found in contributions_df.")

    # --- 1. Determine base value ---
    if 'bias' in contributions_df.columns:
        base_value = contributions_df.loc[instance_index, 'bias']
    elif isinstance(global_base_value, str) and global_base_value in contributions_df.columns:
        base_value = contributions_df.loc[instance_index, global_base_value]
    elif isinstance(global_base_value, (int, float)):
        base_value = float(global_base_value)
    else:
        base_value = 0.0  # default

    # --- 2. Extract feature contributions ---
    instance_contributions_all = contributions_df.loc[instance_index]
    feature_contributions = instance_contributions_all.drop('bias', errors='ignore')
    predicted_value = base_value + feature_contributions.sum()

    # --- 3. Group contributions ---
    sorted_contribs = feature_contributions.abs().sort_values(ascending=False)
    others_sum = 0.0
    grouped_contributions = {}
    for feature_name, abs_value in sorted_contribs.items():
        original_value = feature_contributions[feature_name]
        if abs_value >= threshold_abs:
            grouped_contributions[feature_name] = original_value
        else:
            others_sum += original_value
    if others_sum != 0.0:
        grouped_contributions['Others'] = others_sum

    return {
        'grouped_contributions': grouped_contributions,
        'base_value': base_value,
        'predicted_value': predicted_value
    }


def plot_with_shap(plot_data: Dict, X_instance: pd.Series, true_value: Optional[float] = None, matplotlib: bool = False):
    """Generates and returns a SHAP force plot object with 'Others' shown and optional actual value annotation."""
    contribs = plot_data['grouped_contributions']  # Includes 'Others'
    base_value = plot_data['base_value']
    feature_names = list(contribs.keys())
    shap_values = np.array(list(contribs.values()))

    # If 'Others', give it a placeholder value so it shows in plot
    feature_values = []
    for name in feature_names:
        if name == 'Others':
            feature_values.append('Grouped features')
        else:
            feature_values.append(X_instance.get(name, 'N/A'))

    # Create the plot
    force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=feature_values,
        feature_names=feature_names,
        matplotlib=matplotlib,
        show=not matplotlib
    )

    # Annotate with actual value if available
    if matplotlib and true_value is not None:
        plt.title(f"SHAP Force Plot — Actual Value: {true_value}", fontsize=12)
    elif not matplotlib and true_value is not None:
        print(f"Actual Value: {true_value}")

    return force_plot


def display_prediction_dataframe(
    plot_data: Dict,
    X_instance: pd.Series,
    true_value: Optional[float] = None
) -> pd.DataFrame:
    """Creates, displays, and returns a styled DataFrame breaking down the prediction."""
    base_value = plot_data['base_value']
    predicted_value = plot_data['predicted_value']
    contributions = plot_data['grouped_contributions']

    # --- 1. Prepare data for the DataFrame ---
    sorted_contribs = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    data_for_df = [{'Component': 'Baseline (Dataset Mean)', 'Feature Value': '', 'Contribution': base_value}]

    for feature, contribution_value in sorted_contribs:
        if feature == 'Others':
            continue
        original_value = X_instance.get(feature, 'N/A')
        data_for_df.append({
            'Component': feature,
            'Feature Value': original_value,
            'Contribution': contribution_value
        })

    if 'Others' in contributions:
        data_for_df.append({'Component': 'Others', 'Feature Value': '', 'Contribution': contributions['Others']})

    data_for_df.append({'Component': '---', 'Feature Value': '---', 'Contribution': None})
    data_for_df.append({'Component': 'Final Prediction', 'Feature Value': '', 'Contribution': predicted_value})

    if true_value is not None:
        data_for_df.append({'Component': 'Actual Value', 'Feature Value': '', 'Contribution': true_value})

    # --- 2. Create and style DataFrame ---
    df = pd.DataFrame(data_for_df).set_index('Component')
    df['Feature Value'] = df['Feature Value'].apply(
        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
    )

    max_abs_val = df.loc[
        ~df.index.isin(['Baseline (Dataset Mean)', 'Final Prediction', 'Actual Value', '---']),
        'Contribution'
    ].abs().max()

    styled_df = df.style.background_gradient(
        cmap='RdYlGn',
        axis=0,
        subset=['Contribution'],
        vmin=-max_abs_val,
        vmax=max_abs_val
    ).format({'Contribution': '{:,.4f}'}, na_rep="").set_properties(
        **{'text-align': 'left', 'width': '150px'}
    ).set_table_styles(
        [dict(selector="th", props=[("text-align", "left")])]
    )

    print(f"\nPrediction analysis for instance: {X_instance.name}")
    display(styled_df)
    return df


def interpret_prediction(
    contributions_df: pd.DataFrame,
    instance_index: int,
    X_test: pd.DataFrame,
    y_test: Optional[Union[pd.Series, np.ndarray]]= None,
    mode: str = "forceplot",
    global_base_value: Optional[float] = None,
    threshold_abs: float = 0.01,
    matplotlib: bool = False
):
    """
    Unified wrapper for explaining a single prediction from test data.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series, np.ndarray, or None
        True target values (optional).
    contributions_df : pd.DataFrame
        SHAP contributions (must align with X_test).
    instance_index : int
        Row number from the *reset* index to explain.
    mode : {"forceplot", "dataframe"}
        Output type.
    global_base_value : float or str, optional
        Base value for SHAP if not in contributions_df.
    threshold_abs : float
        Grouping threshold for small contributions.
    matplotlib : bool
        If True and mode="forceplot", renders static matplotlib plot.
    """

    # --- Reset indexes to align everything ---
    X_test_reset = X_test.reset_index(drop=True)
    contributions_reset = contributions_df.reset_index(drop=True)
    y_test_reset = None if y_test is None else pd.Series(y_test).reset_index(drop=True)

    # --- Pull true value if available ---
    true_value = None if y_test_reset is None else y_test_reset.iloc[instance_index]

    # --- Prepare SHAP data ---
    plot_data = prepare_local_force_data(
        contributions_reset,
        instance_index,
        global_base_value=global_base_value,
        threshold_abs=threshold_abs
    )

    X_instance = X_test_reset.loc[instance_index]

    if mode == "forceplot":
        shap.initjs()
        return plot_with_shap(plot_data, X_instance, matplotlib=matplotlib, true_value=true_value)
    elif mode == "dataframe":
        return display_prediction_dataframe(plot_data, X_instance, true_value=true_value)
    else:
        raise ValueError("mode must be either 'forceplot' or 'dataframe'")
