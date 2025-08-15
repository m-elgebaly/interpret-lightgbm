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
from interpret_lightgbm import ProgressiveLGBMRegressor, decompose_prediction

reg = ProgressiveLGBMRegressor(depths=(1,2,3), n_estimators_per_stage=50, learning_rate=0.05)
reg.fit(X_train, y_train)

breakdown = decompose_prediction(reg, X_test[:5])
print("Bias:", breakdown["bias"])
print("Main contributions (row 0):")
print(breakdown["main"].iloc[0].sort_values(ascending=False).head())
```

See a full runnable example in [`examples/demo.py`](examples/demo.py) — it generates synthetic data for both regression and classification.

## API Overview
- `ProgressiveLGBMRegressor(depths=(1,2,3), n_estimators_per_stage=50, learning_rate=0.05, **lgb_params)`  
  Train staged LightGBM regressors; `predict` sums stage outputs.
- `ProgressiveLGBMClassifier(...)`  
  Same idea for classification with `predict_proba` and `predict`.
- `decompose_prediction(model, X)` → dict with `bias`, `main`, `pairwise`, `higher`
- `feature_importance_progression(model)` → DataFrame of importances per stage
- Plotting: `plot_feature_importance_progression`, `plot_contribution_breakdown`

## Roadmap
- Multicollinearity-aware attribution
- Richer visualizations
- Exportable reports

## Contributing
PRs welcome! Please open an issue first to discuss major changes.

## License
MIT — see [LICENSE](LICENSE).
