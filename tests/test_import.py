def test_import():
    import interpret_lightgbm as ilgb
    assert hasattr(ilgb, "__version__")
    assert hasattr(ilgb, "ProgressiveLGBMRegressor")
    assert hasattr(ilgb, "ProgressiveLGBMClassifier")
