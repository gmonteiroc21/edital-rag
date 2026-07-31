.PHONY: help install dev test smoke eval experimento lint fmt up down logs ingest ask clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências com Poetry
	poetry install

dev:  ## Sobe a API localmente com reload
	poetry run uvicorn edital_rag.api.main:app --reload --port 8000

test:  ## Roda a suíte de testes
	poetry run pytest -v

eval:  ## Mede a qualidade da recuperação (não gasta tokens)
	docker compose exec -T api python scripts/avaliar_retrieval.py

experimento:  ## Compara as 4 configs de indexação — uso: make experimento PDF=data/edital.pdf
	@test -n "$(PDF)" || (echo "uso: make experimento PDF=data/edital.pdf" && exit 1)
	docker compose exec -T api python scripts/experimento.py "$(PDF)"

smoke:  ## Testa a chamada à API isoladamente (sem subir o stack)
	@set -a; . ./.env; set +a; \
	curl -s https://api.anthropic.com/v1/messages \
		-H "x-api-key: $$ANTHROPIC_API_KEY" \
		-H "anthropic-version: 2023-06-01" \
		-H "content-type: application/json" \
		-d @tests/fixtures/smoke_api.json | python3 -m json.tool --no-ensure-ascii

lint:  ## Verifica estilo e erros estáticos
	poetry run ruff check src tests

fmt:  ## Formata o código
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

up:  ## Sobe via Docker
	docker compose up --build

down:  ## Derruba os containers
	docker compose down

logs:  ## Acompanha os logs
	docker compose logs -f api

ingest:  ## Indexa um PDF — uso: make ingest PDF=caminho/edital.pdf
	@test -n "$(PDF)" || (echo "uso: make ingest PDF=caminho/edital.pdf" && exit 1)
	curl -sS -X POST http://localhost:8000/ingest -F "arquivo=@$(PDF)" | python3 -m json.tool --no-ensure-ascii

ask:  ## Pergunta — uso: make ask Q="qual o prazo de inscrição?"
	@test -n "$(Q)" || (echo 'uso: make ask Q="sua pergunta"' && exit 1)
	curl -sS -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d "$$(python3 -c 'import json,sys; print(json.dumps({"pergunta": sys.argv[1]}))' "$(Q)")" \
		| python3 -m json.tool --no-ensure-ascii

clean:  ## Apaga o índice local
	rm -f data/*.db
