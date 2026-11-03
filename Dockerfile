# Image de l'API FastAPI (ENF-05). Le serveur charge les modèles enregistrés et ne relance que
# des simulations courtes : aucun entraînement n'a lieu ici (ENF-04).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# Les dépendances sont copiées avant le code : la couche reste en cache tant que pyproject.toml
# et uv.lock ne changent pas. L'extra « ml » apporte PyTorch et Stable-Baselines3, nécessaires
# au chargement des modèles de prédiction, de substitution et des agents.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra ml

COPY configs ./configs

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "spareparts.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
