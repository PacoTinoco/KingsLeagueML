import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import numpy as np
from mlflow import MlflowClient
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor

# Configura la URL del experimento en DagsHub
DAGSHUB_URL = "https://dagshub.com/PacoTinoco/Proyecto_Kings_League"
CSV_PATH = "../data/kings.xlsx"

# Definir la función de lectura de datos
@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    # Lee el archivo de Excel en lugar de CSV
    df = pd.read_excel(file_path, engine='openpyxl')  # O usa 'xlrd' si es necesario

    # Convertir la columna 'date' a formato datetime
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")

    # Ordenar el DataFrame por la columna 'date'
    df = df.sort_values(by="date")

    # Filtrar valores anómalos
    df = df[df['Total_goals_in_dice'] >= 0]

    return df

# Definir función para agregar características
@task(name="Add Features")
def add_features(df: pd.DataFrame):
    features = ['dado_number', 'minutes_after_dice_roll', 'goals_team1', 'goals_team2']
    target = 'Total_goals_in_dice'

    X = df[features]
    y = df[target]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# Función de ajuste de hiperparámetros
@task(name="Hyperparameter Tuning")
def hyper_parameter_tuning_rf(X_train, X_val, y_train, y_val):
    mlflow.sklearn.autolog()

    def objective_rf(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "random_forest")

            rf_model = RandomForestRegressor(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=42
            )

            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mlflow.log_metric("rmse", rmse)

            return {'loss': rmse, 'status': STATUS_OK}

    search_space_rf = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    }

    with mlflow.start_run(run_name="Hyperparameter Tuning Random Forest", nested=True):
        best_params = fmin(
            fn=objective_rf,
            space=search_space_rf,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

        mlflow.log_params(best_params)

    return best_params

# Función para entrenar el mejor modelo
@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, best_params) -> None:
    with mlflow.start_run(run_name="Best Random Forest Model"):
        mlflow.log_params(best_params)

        rf_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )

        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        # Guardar el modelo RandomForest entrenado
        with open("models/KingsLeague_random_forest_model.pkl", "wb") as f_model:
            pickle.dump(rf_model, f_model)
        mlflow.log_artifact("models/KingsLeague_random_forest_model.pkl", artifact_path="model")

    return None

@task(name="Register Model")
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    df = mlflow.search_runs(order_by=['metrics.rmse'])
    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    run_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=run_uri,
        name="RF-model-prefect"
    )

    model_name = "RF-model-prefect"
    model_version_alias = "champion"

    client.set_registered_model_alias(
        name=model_name,
        alias=model_version_alias,
        version='1'
    )

# Definir el flujo principal
@flow(name="Main Flow")
def main_flow() -> None:
    dagshub.init(url=DAGSHUB_URL, mlflow=True)
    mlflow.set_experiment(experiment_name="kings-league-model-randomforest-prefect")

    df = read_data(CSV_PATH)
    X_train, X_val, y_train, y_val = add_features(df)
    best_params = hyper_parameter_tuning_rf(X_train, X_val, y_train, y_val)
    train_best_model(X_train, X_val, y_train, y_val, best_params)
    register_model()

main_flow()

