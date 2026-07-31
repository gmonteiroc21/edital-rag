FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# pdfplumber depende do pdfminer.six, que precisa de libs de imagem em runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

# Camada de dependências separada: só reconstrói quando o pyproject muda.
COPY pyproject.toml README.md ./
RUN poetry install --only main --no-root

COPY src/ ./src/
RUN poetry install --only-root

# Harness de avaliação — roda dentro do container via `make eval`.
COPY scripts/ ./scripts/

# Cache dos modelos de embedding — montado como volume no compose para que o
# download (~220MB) aconteça uma vez só, não a cada `docker compose up`.
ENV HF_HOME=/app/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# --timeout-keep-alive: o default do uvicorn é 5s, mas os navegadores mantêm
# conexões no pool por bem mais tempo (Firefox 115s, Chrome ~300s). Com 5s, o
# servidor fecha a conexão aberta no carregamento da página enquanto o navegador
# ainda a considera válida; o POST seguinte sai por um socket morto e falha com
# NetworkError — sem retry, porque navegador não reenvia POST com corpo. Basta
# o usuário levar 5s escolhendo o arquivo. 120s cobre a janela do Firefox.
CMD ["uvicorn", "edital_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "120"]
