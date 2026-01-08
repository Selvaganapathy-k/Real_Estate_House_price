def run_models(models, x_train_scaled, y_train, x_test_scaled, y_test):
    import mlflow
    import mlflow.sklearn
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    # Set fixed CV splitter for reproducibility
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Parameter grids
    param_grids = {
        "RidgeCV": {"alphas": [[0.1, 1.0, 10.0]]}, 
        "LassoCV": {"alphas": [[0.01, 0.1, 1.0]]},
        "support vector regression": {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5, 1],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        },
        "KNeighborsRegressor": {
            "n_neighbors": [1,2,3,4,5,6,7,8],
            "weights": ["uniform", "distance"]
        },
        "DecisionTreeRegressor": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "RandomForestRegressor": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20,50,60],
            "min_samples_split": [2, 5]
        },
        "GradientBoostingRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "XGBRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0]
        }
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n{name} Results:")

            if name in param_grids and param_grids[name]:
                grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='r2', n_jobs=-1)
                grid.fit(x_train_scaled, y_train)
                best_model = grid.best_estimator_
                print(f"Best Parameters: {grid.best_params_}")
                mlflow.log_params(grid.best_params_)
            else:
                best_model = model
                best_model.fit(x_train_scaled, y_train)

            y_pred = best_model.predict(x_test_scaled)
            r2 = r2_score(y_test, y_pred)
            n = x_test_scaled.shape[0]
            p = x_test_scaled.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            mlflow.log_param("model_type", type(best_model).__name__)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("adjusted_r2", adj_r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=x_train_scaled[:5])

            print(f"Run completed. Run ID: {run.info.run_id}")
