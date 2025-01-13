# Create a new stage from the base python image
FROM python:3.12-bookworm

# Configure Poetry
ENV POETRY_VERSION=2.0.1

# Install poetry separated from system interpreter
RUN pip install -U pip setuptools
RUN pip install poetry==${POETRY_VERSION}

WORKDIR /app

# Copy Dependencies
COPY ["../student-health-predictor-service", "./"]

# [OPTIONAL] Validate the project is properly configured
RUN poetry check

# Install Dependencies
RUN poetry install --no-interaction --no-cache --without dev

# Copy Models
COPY ["../models", "./"]

ENV MODEL_NAME="RandomForestClassifier"

EXPOSE 9696

ENTRYPOINT ["poetry", "run", "python", "-m", "gunicorn", "--bind=0.0.0.0:9696", "app:app"]