.PHONY: install dev start test lint format health

install:
	pip install -r requirements.txt

dev:
	uvicorn src.ai_search.main:app --reload --host 127.0.0.1 --port 7777

start:
	uvicorn src.ai_search.main:app --host 127.0.0.1 --port 7777 --workers 1

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

health:
	curl -s http://localhost:7777/health | python3 -m json.tool
