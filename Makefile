.PHONY: build up down logs ask evaluate clean ui

build:
	docker compose build

up:
	docker compose up

up-d:
	docker compose up -d

down:
	docker compose down

clean:
	docker compose down -v

logs:
	docker compose logs -f app

ask:
	@curl -s -X POST http://localhost:8000/ask \
		-H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}' | python3 -m json.tool

evaluate:
	docker compose exec app python scripts/evaluate.py

ui:
	@echo "Open http://localhost:8501 in your browser"
	@command -v xdg-open >/dev/null && xdg-open http://localhost:8501 || true
