# Workflow CI - MLflow Training Pipeline

## Deskripsi
Repository ini berisi CI/CD workflow untuk training model Wine Quality menggunakan MLflow dan Docker.

## Struktur
```
+-- .github/workflows/ci.yml    # GitHub Actions CI pipeline
+-- MLProject/
    +-- MLProject                # MLflow project config
    +-- conda.yaml               # Conda environment
    +-- modelling.py             # Training script
    +-- wine_quality_preprocessing/  # Preprocessed data
    +-- Tautan ke Docker Hub.txt # Docker Hub link
```

## CI Pipeline
1. **Train Job**: Setup Python -> Install deps -> Run MLflow training -> Upload artifacts
2. **Build Docker Job**: Build MLflow Docker image -> Push to Docker Hub

## Docker Hub
Image: [ardiradi/wine-quality-mlops](https://hub.docker.com/r/ardiradi/wine-quality-mlops)

## Secrets Required
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token
