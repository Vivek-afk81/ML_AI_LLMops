# RecallR — Your Personal Memory Vault

> **Capture it once. Recall it whenever you need it.**

RecallR is an API-first personal knowledge and memory vault. It lets you save notes, useful URLs, and URLs with your own notes, automatically process and categorize that information, and retrieve it later using natural-language semantic search.

Instead of relying only on exact keywords, RecallR uses text embeddings to find memories with similar meaning.

---

## Table of Contents

- [Why RecallR?](#why-recallr)
- [What it can do](#what-it-can-do)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Machine learning models](#machine-learning-models)
- [Project structure](#project-structure)
- [Getting started locally](#getting-started-locally)
- [Using the API](#using-the-api)
- [Memory processing statuses](#memory-processing-statuses)
- [Storage and persistence](#storage-and-persistence)
- [Docker](#docker)
- [Testing and CI/CD](#testing-and-cicd)
- [MLflow benchmarking](#mlflow-benchmarking)
- [Security considerations](#security-considerations)
- [Current limitations](#current-limitations)
- [Future improvements](#future-improvements)

---

## Why RecallR?

Useful information gets scattered across browser bookmarks, documents, chat messages, notes, code snippets, and research links. Later, finding a particular item can be difficult—especially when you remember its meaning but not its exact title or wording.

RecallR provides one place to save this information and search it by meaning.

**Example**

1. Save this memory:

   > An article explaining how Docker containers communicate using bridge networks.

2. Later, search:

   > How do containers communicate with each other?

3. RecallR converts the query into an embedding and searches stored memory vectors for semantically similar content.

The wording does not have to be identical for a relevant memory to be found.

## What it can do

The current MVP supports:

- **Plain-text memories:** save notes, ideas, learning points, and code-related text.
- **URL memories:** save a webpage URL and extract its useful text.
- **Hybrid memories:** save a URL together with a personal note. The extracted webpage content and personal note can both contribute to the memory representation.
- **Automatic multi-label categorization:** assign relevant tags from a configured set of candidate categories.
- **Semantic search:** search saved memories using natural-language queries.
- **Processing status tracking:** check whether extraction, embedding, and classification have completed or failed.
- **Persistent storage:** keep structured memory records in SQLite and vector representations in ChromaDB.

The current project is an API-first MVP. It does **not** include a web frontend.

---

## How it works

### High-level architecture

```text
                 User / API Client
                        |
                        v
                 FastAPI REST API
                        |
                        v
                   Input Router
                 /      |       \
              Text     URL    URL + Note
                \       |       /
                 \      v      /
                  Text Preparation
                        |
             URL Fetching and Extraction
                   (when needed)
                        |
             +----------+-----------+
             |                      |
             v                      v
       MiniLM Embeddings      XtremeDistil
          (384 values)       Zero-shot Tags
             |                      |
             v                      v
          ChromaDB <---------- SQLite
        Vector Storage       Memory Metadata
             |
             v
       Semantic Retrieval
             |
             v
        Relevant Memories
```

### Memory ingestion flow

1. The client submits a plain-text, URL, or hybrid memory to the API.
2. FastAPI validates the request and creates a memory record in SQLite.
3. The API schedules processing using FastAPI `BackgroundTasks` and returns an HTTP `202 Accepted` response.
4. The processing service prepares the text. For URL inputs, it fetches the page and uses Trafilatura to extract its main content.
5. MiniLM converts the prepared text into a normalized 384-dimensional embedding.
6. XtremeDistil evaluates the text against configured candidate labels and selects labels that meet the configured threshold.
7. ChromaDB stores the embedding and memory identifier.
8. SQLite is updated with extracted information, tags, and processing statuses.

### Search flow

1. The client submits a natural-language search query.
2. MiniLM converts the query into an embedding.
3. ChromaDB finds the nearest stored vectors using cosine distance.
4. RecallR uses the returned memory identifiers to retrieve the structured records from SQLite.
5. Results are returned with a similarity score derived from cosine distance.

The returned similarity score is a ranking signal. It is **not** a probability, confidence percentage, or model accuracy.

---

## Technology stack

| Area | Technology | Role |
|---|---|---|
| Language | Python 3.13 | Application development |
| API | FastAPI | REST API and interactive API documentation |
| Validation | Pydantic | Request and response schemas |
| ORM / database access | SQLAlchemy | Structured database operations |
| Structured storage | SQLite | Memory records and processing metadata |
| Vector storage | ChromaDB | Embedding persistence and similarity search |
| Embeddings | Sentence Transformers | Text-to-vector conversion |
| Classification | Hugging Face Transformers / PyTorch | Zero-shot multi-label classification |
| Web extraction | Trafilatura | Extract main text and title from webpages |
| HTTPS certificates | Certifi | Trusted CA certificates for HTTPS requests |
| Testing | pytest | Automated tests |
| HTTP testing | HTTPX / FastAPI TestClient | API and request testing |
| Containerization | Docker | Reproducible application runtime |
| CI/CD | GitHub Actions | Automated test and image publishing workflow |
| Container registry | GitHub Container Registry | Published Docker image |
| Experiment tracking | MLflow | Benchmark runs, parameters, metrics, and artifacts |
| Configuration | YAML | Model and inference configuration |

---

## Machine learning models

RecallR uses two pretrained models with separate responsibilities. The MVP does not use an LLM, task-specific fine-tuning, generated summaries, or a Cross-Encoder reranking stage.

### 1. Embedding model — MiniLM

**Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Used for:

- Creating embeddings for saved memories.
- Creating embeddings for search queries.
- Enabling semantic similarity search.

Important details:

- Produces **384-dimensional** sentence embeddings.
- Embeddings are normalized in the application.
- Configured to run on CPU.
- The same embedding model is used for ingestion and search so that vectors are comparable.

### 2. Zero-shot classifier — XtremeDistil

**Model:** [`MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33`](https://huggingface.co/MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33)

Used to assign multiple relevant categories to a memory without training a classifier on a project-specific labelled dataset.

The configured candidate labels are:

- Technology
- AI/ML
- MLOps
- DevOps
- Programming
- Research
- Career
- Finance
- Personal
- Education

The current configured classification threshold is **0.5**, and multi-label classification is enabled. These are configuration choices, not a claim that the model has achieved a particular accuracy.

**Important evaluation note:** the XtremeDistil model is treated as a lightweight candidate. Its tags can be imperfect or empty when no candidate meets the threshold. A manually labelled evaluation dataset is needed before making precision, recall, or F1-score claims.

### Why two models?

| Model | Input | Output | Responsibility |
|---|---|---|---|
| MiniLM | Text | 384-dimensional vector | Semantic retrieval |
| XtremeDistil | Text and candidate labels | Relevant category labels | Automatic organization |

One model finds memories with similar meaning; the other suggests how a memory should be categorized.

---

## Project structure

The main implemented components are organized as follows. Some supporting files may vary as the repository evolves.

```text
ML_AI_LLMops/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes/
│   │       ├── memories.py
│   │       └── search.py
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   ├── sqlite.py
│   │   └── vector_store.py
│   ├── models/
│   │   ├── database_models.py
│   │   └── schemas.py
│   └── services/
│       ├── input_router.py
│       ├── text_processor.py
│       ├── content_extractor.py
│       ├── embedding_service.py
│       ├── classification_service.py
│       ├── memory_processing_service.py
│       └── retrieval_service.py
├── configs/
│   └── model_config.yaml
├── scripts/
│   └── mlflow_benchmark.py
├── tests/
├── data/                       # Runtime SQLite and ChromaDB data
├── .github/
│   └── workflows/
│       └── ci.yaml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

Runtime data and MLflow tracking files are local artifacts and are excluded from Git where configured.

---

## Getting started locally

### Prerequisites

- Python **3.13**
- Git
- Internet access for the first model download and for URL extraction
- Enough disk space for the Python dependencies and pretrained model files

The application is configured for CPU inference. The first model load may take longer because the pretrained models need to be downloaded and initialized.

### 1. Clone the repository

```bash
git clone https://github.com/Vivek-afk81/ML_AI_LLMops.git
cd ML_AI_LLMops
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.13 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API

From the repository root, with the virtual environment activated:

```bash
uvicorn app.main:app --reload
```

The API should be available at:

- API root: <http://127.0.0.1:8000/>
- Health check: <http://127.0.0.1:8000/health>
- Interactive Swagger documentation: <http://127.0.0.1:8000/docs>

Use the Swagger page to inspect the exact request and response schemas exposed by the running application.

> **First run:** model initialization and downloads can take time. Keep the API process running while a memory is being processed.

---

## Using the API

The current API exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API root |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/memories/` | Submit a memory for processing |
| `GET` | `/api/v1/memories/{memory_id}` | Retrieve a memory and its processing status |
| `POST` | `/api/v1/search/` | Search memories semantically |

### 1. Save a plain-text memory

Send a `POST` request to:

```text
http://127.0.0.1:8000/api/v1/memories/
```

Example JSON:

```json
{
  "input_type": "text",
  "original_text": "I am learning Python, machine learning, and Docker to prepare for an AI engineering role.",
  "personal_note": null,
  "source_url": null
}
```

### 2. Save a URL

```json
{
  "input_type": "url",
  "original_text": null,
  "personal_note": null,
  "source_url": "https://example.com"
}
```

RecallR fetches the webpage and attempts to extract its title and main text.

### 3. Save a URL with a personal note

```json
{
  "input_type": "hybrid",
  "original_text": null,
  "personal_note": "Focus on the section explaining container networking. I want to revise this for interviews.",
  "source_url": "https://example.com"
}
```

The personal note is retained alongside the URL and can contribute to the prepared memory content.

### 4. Check memory processing status

Use the `memory_id` returned by the create request:

```text
GET http://127.0.0.1:8000/api/v1/memories/{memory_id}
```

The API returns the memory record and its current processing information.

### 5. Search saved memories

Send a `POST` request to:

```text
http://127.0.0.1:8000/api/v1/search/
```

Example query:

```json
{
  "query": "What am I learning for an AI engineering career?"
}
```

Check the live Swagger schema at `/docs` for any additional optional fields supported by the current `SearchRequest` model.

### Example using curl

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/memories/" \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "original_text": "I am learning Python and machine learning.",
    "personal_note": null,
    "source_url": null
  }'
```

On Windows PowerShell, Swagger UI is often the easiest way to submit JSON without shell-quoting issues.

### Asynchronous processing response

Memory creation returns **HTTP 202 Accepted** because processing is scheduled in the background. A successful create response means the request was accepted; it does not necessarily mean extraction, embedding, and classification have already finished.

Use the returned memory ID with the status endpoint to check completion.

---

## Memory processing statuses

RecallR tracks processing stages separately so that a client can understand what happened to a submitted memory.

| Status | Meaning |
|---|---|
| `pending` | Memory has been accepted and is waiting for processing |
| `processing` | Processing is currently underway |
| `completed` | The processing pipeline completed successfully |
| `extraction_failed` | URL content extraction failed |
| `embedding_failed` | Embedding generation failed |
| `classification_failed` | Classification failed |
| `failed` | A general processing failure occurred |

The record also tracks embedding and classification statuses independently.

A URL extraction failure does not necessarily mean the entire memory is unusable. For hybrid memories, a useful personal note may still provide content that can be processed, depending on the failure path and available input.

---

## Storage and persistence

RecallR uses two storage systems for different purposes.

### SQLite — structured records

SQLite stores memory information such as:

- Memory ID and input type.
- Original text and personal note.
- Source URL.
- Extracted title and content.
- Tags.
- Processing, embedding, and classification statuses.
- Error details and timestamps.

### ChromaDB — vector records

ChromaDB stores embeddings and their associated memory identifiers. It is used to find semantically similar memories.

The application uses a persistent ChromaDB collection named `recallr_memories` with cosine-distance-based retrieval.

### Local data locations

The application data is stored under the `data/` directory:

```text
data/
├── recallr.db
└── chroma_db/
```

Do not delete this directory if you want to keep locally stored memories.

---

## Docker

A RecallR Docker image has been built and published to GitHub Container Registry.

**Image:**

```text
ghcr.io/vivek-afk81/ml_ai_llmops:latest
```

The package is public. A commit-specific image tag is also published by the CI workflow.

### Run the published image

Create local directories first:

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force data, model_cache
```

Run the container:

```powershell
docker run -d `
  --name recallr-container `
  -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/model_cache:/root/.cache/huggingface" `
  ghcr.io/vivek-afk81/ml_ai_llmops:latest
```

**Linux / macOS**

```bash
mkdir -p data model_cache

docker run -d \
  --name recallr-container \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/model_cache:/root/.cache/huggingface" \
  ghcr.io/vivek-afk81/ml_ai_llmops:latest
```

Then open:

- <http://localhost:8000/health>
- <http://localhost:8000/docs>

The mounts keep application data and the Hugging Face model cache outside the container's writable layer. This helps preserve memory data and avoid repeatedly downloading model files when the same cache is reused.

Useful container commands:

```bash
docker logs -f recallr-container
docker stop recallr-container
docker start recallr-container
docker rm -f recallr-container
```

### Build the image locally

From the repository root:

```bash
docker build -t recallr:local .
```

Run it with the same persistent mounts shown above, replacing the published image name with `recallr:local`.

---

## Testing and CI/CD

### Run tests locally

Activate the project virtual environment and run:

```bash
python -m pytest -v
```

Some tests exercise real model inference and may need model files to be downloaded. Test execution time and resource use therefore depend on the environment.

### GitHub Actions

The repository includes a GitHub Actions workflow at:

```text
.github/workflows/ci.yaml
```

The workflow is configured to:

1. Run the test job on pushes to `main` and `feature/recallr`, and on pull requests targeting `main`.
2. Build and publish the Docker image to GitHub Container Registry after successful tests on pushes to `main`.
3. Publish the `latest` tag and a short commit-SHA tag.

The main-branch workflow has completed successfully, including the test job and Docker image publishing job.

---

## MLflow benchmarking

RecallR includes a benchmark script:

```text
scripts/mlflow_benchmark.py
```

It uses the existing embedding and classification services to measure model inference performance and log the results to MLflow.

### Run the benchmark

From the repository root:

```bash
python scripts/mlflow_benchmark.py
```

### Start the MLflow UI

The local MLflow tracking database and artifact directory are ignored by Git. From the repository root, run:

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>.

If your local MLflow tracking URI is configured differently, use the same tracking URI for the UI that was used by the benchmark run.

### What the benchmark measures

- Model cold-start time.
- Warm embedding inference latency.
- Warm classification inference latency.
- Combined embedding and classification inference latency.
- Mean, median, minimum, and maximum values for repeated warm measurements.
- Model/configuration parameters and benchmark artifacts.

**Scope of the current benchmark:** it measures model inference, not the complete database-to-vector-store memory ingestion pipeline.

### Latest recorded successful benchmark

The successful local benchmark run recorded the following measurements:

| Metric | Result |
|---|---:|
| Embedding cold start | 1727.01 ms |
| Embedding warm mean | 14.60 ms |
| Embedding warm median | 14.22 ms |
| Classification cold start | 1146.07 ms |
| Classification warm mean | 206.35 ms |
| Classification warm median | 210.03 ms |
| Combined inference mean | 229.42 ms |
| Combined inference median | 228.17 ms |

These are environment-specific measurements from one local benchmark run, not universal performance guarantees.

No labelled evaluation dataset has been used to establish classification precision/recall/F1 or retrieval Recall@K. No such quality metrics are claimed here.

---

## Security considerations

Users can submit URLs, so URL fetching must be treated as an untrusted-input operation.

The URL extraction implementation includes safeguards such as:

- Allowing HTTP and HTTPS schemes only.
- Rejecting URL credentials.
- Rejecting localhost and non-public/private IP destinations.
- Resolving hostnames and validating destination IP addresses.
- Using pinned connections to reduce DNS-rebinding risk.
- Keeping HTTPS certificate verification enabled using Certifi.
- Following redirects manually and validating each redirect destination.
- Limiting redirects.
- Applying a request timeout.
- Limiting response size.
- Restricting accepted response content types to HTML/XHTML.

These measures reduce SSRF and resource-exhaustion risks; they should not be interpreted as a guarantee of perfect security.

For a public production deployment, add authentication, authorization, rate limiting, HTTPS termination, monitoring, backups, and an operational security review.

---

## Current limitations

- **No frontend:** the current interface for trying the API is Swagger UI.
- **Background task durability:** FastAPI `BackgroundTasks` is suitable for this MVP, but it is not a durable job queue. A process or container crash can interrupt in-progress work.
- **Web extraction:** JavaScript-heavy, paywalled, blocked, or unusual pages may not yield useful text through Trafilatura.
- **Automatic tags:** zero-shot predictions can be imperfect. A memory may have no tags if no candidate label crosses the configured threshold.
- **Quality evaluation:** a labelled dataset is still needed for reliable classification and retrieval quality metrics.
- **CPU inference:** the first model load can be slow, and inference latency depends on the machine.
- **Single-user MVP:** authentication and multi-user access controls are not part of the current implementation.

---

## Future improvements

Potential next steps—outside the current MVP—include:

- A frontend dashboard.
- Authentication and multi-user support.
- Durable background jobs using a task queue.
- A manually labelled dataset and formal retrieval/classification evaluation.
- User-editable tags and memory management operations.
- Memory export, backup, and restore.
- Cloud deployment and operational monitoring.
- Optional integrations such as a Telegram bot.

These are future directions, not features that should be assumed to exist in the current version.

---

## Project status

The project has progressed beyond a design-only prototype. The API, text/URL/hybrid processing paths, pretrained model integration, SQLite and ChromaDB persistence, semantic search, tests, Docker packaging, GitHub Actions CI/CD, GHCR publishing, and MLflow inference benchmarking have been implemented.

The README describes the current API-first MVP. For exact request/response schemas, use the running Swagger documentation at `/docs`.

---

## License

No license is specified here. Add a repository license file before treating this project as open source for reuse or redistribution.
