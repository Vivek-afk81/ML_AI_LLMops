from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

#Configure MLflow Tracking
mlflow.set_tracking_url("http://0.0.0.0:6001")

#group related training runs under 1 experiment
mlflow.set_experiment("iris-classification")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

n_estiators=100
random_state=42

#model config
model_parameters={
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estiators,
    "random_state": random_state,
    "test_size":0.2
}


#Start a tracked training run

with mlflow.start_run() as run:
    # record the traing config before model fitting
    mlflow.log_params(model_parameters)

    #build classifier using the selected parameters
    model=RandomForestClassifier(
        n_estimators=n_estiators,
        random_state=random_state
    )

    #Fit the model on training data
    model.fit(X_train,y_train)

    #generate the predictions
    y_pred=model.predict(X_test)\

    #calculate the metrics that helps us evaluate the classifier
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,average="weighted")
    recall=recall_score(y_test,y_pred,average="weighted")
    f1=f1_score(y_test,y_pred,average="weighted")


    #log the evaluated results so that the different runs can be compared
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })

    print("MLFlow run ID: ",run.info.run_id)
    print("Test Accuracy: ",accuracy)
    print("Test Precision: ",precision)
    print("Test Recall: ",recall)


    #save the trained model as a reusable artifact
    







model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model as reusable artifact
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_dir / "iris_model.joblib")
print(f"Model saved to {model_dir / 'iris_model.joblib'}")

