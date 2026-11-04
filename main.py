from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os
from contextlib import asynccontextmanager
from typing import Callable




# app = FastAPI(title="Iris Model API")

#mlflow configuration
MLFLOW_TRACKING_URI =os.getenv(
    "MLFLOW_TRACKING_URI","http://127.0.0.1:6001"
)

#Molde URI : models:/iris-classifier@champion
MODEL_URI="models:/iris-classifier@champion"

#Tell mlflow client where the model registry lives
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


# model_path = Path("models/iris_model.joblib")
# model = joblib.load(model_path)

# try:
#     model=mlflow.sklearn.load_model(MODEL_URI)
#     client = mlflow.MlflowClient()
#     champion_version=client.get_model_version_by_alias(
#         name="iris-classifier", alias="champion"
#     )
#     model_version=champion_version.version
#     print(f"Loade model: iris-classifier{model_version}")
# except Exception as e:
#     raise RuntimeError(
#         f"Failed to load model from {MODEL_URI}. Error: {str(e)}"
#     )


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

#Application Factory
def create_app(model_loader:Callable=load_mlflow_model,
               version_loader:Callable=None
):
    
    if version_loader is None:
        def version_loader():
            
            client=mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            champion=client.get_model_version_by_alias(
                name="iris-classifier", alias="champion"
            )
            return champion.version
        
    @asynccontextmanager
    async def lifespan(app:FastAPI):
        try:
            app.state.model=model_loader()
            app.state.model_version=version_loader()

            print(
                f"Loaded iris-classifier"
                f"{app.state.model_version}"
            )

        except Exception as e:
            raise RuntimeError(
                f"failed to load MLflow champion model ..\n{e}"
            ) from e
        yield


    app=FastAPI(
        title="Iris Model API",
        description="API for serving the iris-classifier model",
        version="3.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": "iris-classifier",
            "model_version": app.state.model_version
        }

    @app.post("/predict")
    def predict(data: IrisRequest,request:Request):
        model=request.app.state.model
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = int(model.predict(features)[0])

        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]
        }

    return app


app = create_app()
