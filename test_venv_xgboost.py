import xgboost
print("XGBoost version:", xgboost.__version__)
model = xgboost.XGBClassifier()
model.fit([[1,2],[3,4]], [0,1], early_stopping_rounds=5)
print("Success!")
