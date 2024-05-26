
# Experiment tracking

[Class Slides](https://drive.google.com/file/d/1YtkAtOQS3wvY7yts_nosVlXrLQBq5q37/view)

Experiment tracking is the process of keeping track of all the relevant information from an ML experiment, which includes:

- Source code
- Environment
- Data
- Model
- Hyperparameters
- Metrics
... and more

Why is important to keep track of all this information?
In general, because of these 3 main reasons:

- Reproducibility
- Organization
- Optimization

## MLflow

MLflow is a tool that helps you keep track of all this information. It is an open-source platform for the complete machine learning lifecycle. It is designed to work with any ML library, algorithm, deployment tool or language.

In practice, it's just a Python package that can be installed with pip, and it contains
four main modules:

- Tracking
- Models
- Model Registry
- Projects

The MLflow Tracking module allows you to organize your experiments into runs, and to keep track of:

- Parameters
- Metrics
- Metadata
- Artifacts
- Models

Along with this information, MLflow automatically logs extra information about the run:

- Source code
- Version of the code (git commit)
- Start and end time
- Author

## Getting started with MLflow

Prepare your environment by installing the required packages:

```bash
pip install -r requirements.txt
```

Launch the MLflow UI:
To store the metadata, artifacts, and models in a mlflow database, you can use the following command:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

Add mlflow to your project:

To launch the MLflow UI, you can run the following command in the terminal:

```bash
mlflow server
```

---

### References

[MLFlow documentation](https://www.mlflow.org/docs/latest/index.html)
