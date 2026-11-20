import json
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlflow

# Ensure project root is available for app imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.classification_service import classification_service


EXPERIMENT_NAME = "RecallR-Phase8"

SAMPLE_TEXT = (
    "Machine learning and MLOps help developers build, "
    "deploy, monitor, and maintain reliable AI applications."
)

ITERATIONS = 5


def measure_latency(function):
    """Measure execution time in milliseconds."""

    start = time.perf_counter()
    result = function()
    elapsed = (time.perf_counter() - start) * 1000

    return result, elapsed


def log_latency_metrics(prefix, measurements):
    """Log latency statistics to MLflow."""

    mlflow.log_metrics({
        f"{prefix}_mean_ms": statistics.mean(measurements),
        f"{prefix}_median_ms": statistics.median(measurements),
        f"{prefix}_min_ms": min(measurements),
        f"{prefix}_max_ms": max(measurements),
    })


def benchmark_service(name, function):
    """Measure cold-start and warm inference latency."""

    print(f"\nBenchmarking {name}...")

    # First call includes lazy model loading.
    _, cold_start = measure_latency(function)

    # Warm-up call: excluded from measurements.
    function()

    # Measure repeated warm inference.
    measurements = []

    for _ in range(ITERATIONS):
        _, elapsed = measure_latency(function)
        measurements.append(elapsed)

    mlflow.log_metric(
        f"{name}_cold_start_ms",
        cold_start,
    )

    log_latency_metrics(name, measurements)

    print(f"Cold start: {cold_start:.2f} ms")
    print(f"Warm mean: {statistics.mean(measurements):.2f} ms")
    print(f"Warm median: {statistics.median(measurements):.2f} ms")

    return measurements, cold_start


def main():

    embedding_config = settings["embedding_model"]
    classification_config = settings["classification_model"]

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model-inference-benchmark"):

        print("Starting RecallR MLflow benchmark...")
        print("Tracking URI:", mlflow.get_tracking_uri())

        # 1. Record model configuration.
        mlflow.log_params({
            "embedding_model": embedding_config["model_id"],
            "embedding_dimension": embedding_config["embedding_dimension"],
            "normalize_embeddings": embedding_config["normalize_embeddings"],
            "classification_model": classification_config["model_id"],
            "classification_threshold": classification_config["threshold"],
            "classification_multi_label": classification_config["multi_label"],
            "candidate_labels": ",".join(
                classification_config["candidate_labels"]
            ),
            "device": settings["inference"]["device"],
            "benchmark_iterations": ITERATIONS,
            "python_version": sys.version.split()[0],
            "mlflow_version": version("mlflow"),
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "sentence_transformers_version": version("sentence-transformers"),
        })

        # 2. Benchmark embedding model.
        embedding_function = lambda: embedding_service.generate_embedding(
            SAMPLE_TEXT
        )

        embedding_measurements,embedding_cold_start = benchmark_service(
            "embedding",
            embedding_function,
        )

        # 3. Benchmark zero-shot classification model.
        classification_function = lambda: classification_service.classify_text(
            SAMPLE_TEXT
        )

        classification_measurements,classification_cold_start = benchmark_service(
            "classification",
            classification_function,
        )

        # 4. Benchmark combined model inference.
        def inference_pipeline():
            embedding = embedding_service.generate_embedding(SAMPLE_TEXT)
            tags = classification_service.classify_text(SAMPLE_TEXT)

            return embedding, tags

        print("\nBenchmarking combined inference pipeline...")

        pipeline_measurements = []

        for _ in range(ITERATIONS):
            _, elapsed = measure_latency(inference_pipeline)
            pipeline_measurements.append(elapsed)

        log_latency_metrics("inference_pipeline", pipeline_measurements)

        print(
            "Pipeline mean:",
            f"{statistics.mean(pipeline_measurements):.2f} ms",
        )

        # 5. Record configuration as an MLflow artifact.
        config_path = PROJECT_ROOT / "configs" / "model_config.yaml"

        mlflow.log_artifact(
            str(config_path),
            artifact_path="configuration",
        )

        # 6. Save benchmark results as a JSON artifact.
        results = {
            "embedding": {
                "cold_start_ms": None,
                "warm_measurements_ms": embedding_measurements,
            },
            "classification": {
                "warm_measurements_ms": classification_measurements,
            },
            "inference_pipeline": {
                "warm_measurements_ms": pipeline_measurements,
            },
            "iterations": ITERATIONS,
            "note": (
                "Latency benchmark only. "
                "No model accuracy evaluation was performed."
            ),
        }

        results["embedding"]["cold_start_ms"] = embedding_cold_start

        results["classification"]["cold_start_ms"] = classification_cold_start

        results_path = PROJECT_ROOT / "scripts" / "benchmark_results.json"

        results_path.write_text(
            json.dumps(results, indent=4),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="benchmark_results",
        )

        results_path.unlink(missing_ok=True)

        print("\nMLflow benchmark completed successfully.")
        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
