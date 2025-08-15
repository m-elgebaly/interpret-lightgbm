# interpret-lightgbm

**interpret-lightgbm** turns powerful LightGBM models into **transparent, explainable predictors**.

The library trains LightGBM in **progressive stages** — starting with tiny trees (depth=1), then depth=2, then depth=3+.
This lets us **separate the prediction** for each sample into:
- **Main effects (depth=1)**: single-feature contributions
- **Pairwise interactions (depth=2)**: two-feature interactions
- **Higher-order interactions (depth≥3)**: complex interactions

Similar in spirit to SHAP, but instead of approximations, `interpret-lightgbm` computes **exact contributions**
by leveraging the staged model structure and LightGBM's per-feature contributions while controlling tree depth.
You see how each feature **moves the prediction away from the mean** and when interactions dominate.

**Doesn't (yet) solve multicollinearity** — planned in the roadmap.

## Features
- Works for **regression** and **classification**
- Per-sample contribution breakdown: `bias`, `main`, `pairwise`, `higher`
- Feature-importance progression across depths
- Simple plotting helpers (Matplotlib)
- Clean, documented API

## Installation
```bash
git clone https://github.com/m-elgebaly/interpret-lightgbm.git
cd interpret-lightgbm
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quickstart
```python
base_params = {
    'n_estimators': 100,
    'learning_rate': 1,
    'num_leaves': 31,
    'seed': 42,
}

# Parameters specific to each depth. Here, we can override or add parameters.
# For example, we increase `n_estimators` as the model gets deeper.

depth_params = {
    1: {"n_estimators": 100, "learning_rate": 1, "min_child_samples": 10},
    2: {"n_estimators": 50, "learning_rate": 1},
    3: {"n_estimators": 30, "learning_rate": 1},#, "reg_alpha": 0.1}
    15: {"n_estimators": 400, "learning_rate": 0.1}#, "reg_alpha": 0.1}
}
# 3. Instantiate and fit the model
# We pass the evaluation metric we care about (optional, defaults to MSE for regression)
progressive_reg = ProgressiveLGBMRegressor(
    base_params=base_params,
    depth_params=depth_params,
    eval_metric=mean_squared_error,
    verbose=True
)

# Fit the model using the train/test sets
progressive_reg.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# 4. Access the stored attributes after fitting
print("\n--- Accessing Results ---")
print(f"Final Test Score (MSE): {progressive_reg.final_score_:.4f}")

# The models_ attribute is a dictionary where keys are the depths
print(f"Models were trained for depths: {list(progressive_reg.models_.keys())}")

# The contributions_ attribute contains SHAP values for each model at each depth
# Let's inspect the shape for the model with max_depth=4
final_depth = max(depth_params.keys())
print(f"Shape of contributions for depth {final_depth} model: {progressive_reg.contributions_[final_depth].shape}")
print("(Shape corresponds to: n_samples, n_features + 1 for bias)")


# 5. Make predictions on new data using the final fitted model
new_data = X_test.head(5)
predictions = progressive_reg.predict(new_data)
print("\n--- Making Predictions ---")
print("Predictions on 5 new samples:\n", predictions)
```

## Explore and Visualize Feature contribution
Example:
<img width="414" height="430" alt="image" src="https://github.com/user-attachments/assets/f59aad0a-eed2-4ee5-a2a1-d0055d3f786f" />


See a full runnable example in [`examples/demo.py`](examples/demo.py) — it generates synthetic data for both regression and classification.

## API Overview
- `ProgressiveLGBMRegressor()`  
  Train staged LightGBM regressors; `predict` sums stage outputs.
- `ProgressiveLGBMClassifier(...)`  
  Same idea for classification with `predict_proba` and `predict`.
- `get_feature_contributions(X_test, model_depth)` → DataFrame of with ontribution `bias`, `main`, `pairwise`, `higher`
- Plotting: `plot_contributions`, `plot_feature_importance_progression`

## Roadmap
- Multicollinearity-aware attribution
- Richer visualizations
- Exportable reports

## Contributing
PRs welcome! Please open an issue first to discuss major changes.

## License
MIT — see [LICENSE](LICENSE).
