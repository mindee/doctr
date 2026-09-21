# api setup is borrowed from https://github.com/frgfm/Holocron/blob/main/api

.PHONY: lock run stop test
# Pin the dependencies
lock:
	pip install uv
	uv pip compile pyproject.toml --universal --no-header -o requirements.txt
	uv pip compile pyproject.toml --universal --no-header --group dev -o requirements-dev.txt

# Run the docker
run:
	docker compose up -d --build

# Run the docker
stop:
	docker compose down

# Run tests for the library
test:
	docker compose up -d --build
	docker cp requirements-dev.txt api_web:/app/requirements-dev.txt
	docker compose exec -T web pip install -r requirements-dev.txt
	docker cp tests api_web:/app/tests
	docker compose exec -T web pytest tests/ -vv
	docker compose down
