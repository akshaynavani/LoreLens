.PHONY: install dev lint test ingest-sample ask serve mcp eval docker

install:
	pip install -e .

dev:
	pip install -e ".[dev,huggingface]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q

ingest-sample:
	lorelens ingest data/sample_docs --base-url https://docs.example.com

ask:
	lorelens ask "How long does a Nimbus v2 access token last?" --show-trace

serve:
	lorelens serve --reload

mcp:
	python -m lorelens.mcp_server --transport streamable-http --port 8765

eval:
	lorelens eval --dataset data/eval/sample_eval.jsonl

docker:
	docker compose up --build
