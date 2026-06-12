# MLFlowProject 🍷

An end-to-end **MLOps pipeline** for predicting wine quality using ElasticNet regression — built with MLflow experiment tracking, modular pipeline stages, Docker containerization, and GitHub Actions CI/CD.

---

## Overview

This project demonstrates a production-style machine learning workflow on the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). Rather than a one-shot notebook, the codebase is organized as a proper Python package with a five-stage pipeline, centralized config management, and experiment tracking — everything an ML engineer would expect in a real deployment.

**What it does:**  
Given 11 physicochemical features of a wine (acidity, alcohol, pH, sulphates, etc.), the model predicts a quality score. More importantly, every training run is tracked, reproducible, and deployable.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn (ElasticNet), pandas, numpy |
| Experiment tracking | MLflow |
| Config management | PyYAML, python-box, ensure |
| Serialization | joblib |
| Serving | Flask, Flask-Cors |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Research | Jupyter Notebooks |

---

## Project Structure

```
MLFlowProject/
│
├── src/
│   └── mlOpsProject/           # Installable Python package
│       ├── components/         # Core logic classes (DataIngestion, ModelTrainer, etc.)
│       ├── pipeline/           # Stage wrappers (stage_01 → stage_05)
│       ├── entity/             # Dataclasses for typed config objects
│       ├── config/             # ConfigurationManager (reads all YAMLs)
│       └── utils/              # Shared helpers (read_yaml, create_dirs)
│
├── research/                   # Jupyter notebooks for EDA & prototyping
│
├── config/
│   └── config.yaml             # File paths, download URLs, artifact dirs
├── schema.yaml                 # Expected column names and dtypes
├── params.yaml                 # Model hyperparameters (alpha, l1_ratio)
│
├── main.py                     # Pipeline entry point — runs all 5 stages
├── app.py                      # Flask API for serving predictions
├── setup.py                    # Makes src/ an editable install
├── template.py                 # Project scaffolding script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
└── .github/workflows/          # GitHub Actions CI/CD
```

---

## Pipeline Stages

`main.py` runs these five stages sequentially. Each stage is independently logged and can be re-run from any point.

```
Data Ingestion → Data Validation → Data Transformation → Model Trainer → Model Evaluation
```

### Stage 1 — Data Ingestion
Downloads the wine quality CSV from the configured URL and saves it to the local artifact directory.

### Stage 2 — Data Validation
Validates the downloaded dataset against `schema.yaml` — checks that all expected columns are present and have the correct dtypes. Pipeline halts with a clear error if validation fails.

### Stage 3 — Data Transformation
Splits the validated data into train and test sets and saves them as `train.csv` and `test.csv`. This is also where feature scaling or encoding would live.

### Stage 4 — Model Trainer
Reads `train.csv` and fits an ElasticNet regression model using hyperparameters from `params.yaml`. Saves the trained model as `model.joblib`.

### Stage 5 — Model Evaluation
Loads the test set and saved model, generates predictions, and computes RMSE and R² — then logs all parameters, metrics, and the model artifact to **MLflow** for reproducibility and comparison across runs.

---

## Configuration

All behaviour is controlled through three YAML files — no hardcoded values in source code.

**`params.yaml`**
```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

**`schema.yaml`** — defines the expected dataset schema:
```yaml
COLUMNS:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64

TARGET_COLUMN:
  name: quality
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/Ayush-Debnath/MLFlowProject.git
cd MLFlowProject

# Install dependencies (including the local package in editable mode)
pip install -r requirements.txt
```

### Run the pipeline

```bash
python main.py
```

Each stage logs its start and completion. Artifacts are saved under the paths configured in `config/config.yaml`.

### View MLflow experiments

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) to compare runs, parameters, and metrics.

---

## Running with Docker

```bash
# Build the image
docker build -t mlflow-wine .

# Run the container
docker run -p 8080:8080 mlflow-wine
```

---

## API Usage

Once `app.py` is running, send a POST request with the wine's feature values to get a quality prediction:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

---

## How Config Management Works

The `ConfigurationManager` class reads all three YAML files on startup and returns strongly-typed dataclass objects (defined in `entity/`) to each pipeline stage. This means:

- Changing a file path only requires editing `config.yaml`
- Swapping hyperparameters only requires editing `params.yaml`
- No stage needs to know where another stage's files live

---

## Author

**Ayush Debnath**  
B.Tech Computer Science Engineering — Graphic Era Hill University  
[GitHub](https://github.com/Ayush-Debnath) · debnathayush48@gmail.com

---

## License

This project is open-source and available under the [MIT License](LICENSE).