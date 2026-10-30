.PHONY: install lint fix test check run up down

install:
	python -m pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .

fix:
	ruff check . --fix

test:
	python -m pytest -q

check: lint test

run:
	python app.py

up:
	docker compose up --build -d

down:
	docker compose down
