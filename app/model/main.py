import pickle
import mlflow
import pathlib
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient

# Configuración de MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/PacoTinoco/Proyecto_Kings_League.mlflow"
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Cargar artefactos
run_ = mlflow.search_runs(
    order_by=['metrics.rmse ASC'],
    output_format="list",
    experiment_names=["kings-league-model-randomforest-prefect"]
)[0]
run_id = run_.info.run_id

# Descargar preprocesador
client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

with open("preprocessor/preprocessor.b", "rb") as f_in:
    preprocessor = pickle.load(f_in)

# Cargar modelo
model_name = "RF-model"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"
champion_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Preprocesamiento
def preprocess(input_data):
    input_dict = {
        'team1': input_data.team1,
        'team2': input_data.team2,
        'dado_number': input_data.dado_number,
        'minutes_after_dice_roll': input_data.minutes_after_dice_roll
    }

    return preprocessor.transform(input_dict)

# Predicción
def predict(input_data):
    X_pred = preprocess(input_data)
    return champion_model.predict(X_pred)

# Configuración de la API
app = FastAPI()

# Definir estructura de entrada para la API
class InputData(BaseModel):
    team1: int
    team2: int
    dado_number: int
    minutes_after_dice_roll: int

@app.get("/")
def greet():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    return {"prediction": float(result)}

