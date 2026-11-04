# Import TestClient so we can send simulated HTTP requests
from fastapi.testclient import TestClient
import pytest

from main import create_app


class FakeIrisModel:
    def predict(self,features):
        return [0]

def fake_model_loader():
    return FakeIrisModel()

def fake_version_loader():
    return "test-version"

@pytest.fixture # makes an object of client that can be used in test functions
def client():
    app=create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )
    return TestClient(app)

def test_health_endpoint(client):
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()=={"status":"healthy"}
    assert response.json()["model_version"]=="test-version"

def test_valid_iris_prediction(client):

    payload = {
    "sepal_length": 3.4,
    "sepal_width": 1.4,
    "petal_length": 3.4,
    "petal_width": 0.4
    }

    #send a simulated post request with json input
    response=client.post("/predict",json=payload)
    assert response.status_code==200

    result = response.json()
    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]
 