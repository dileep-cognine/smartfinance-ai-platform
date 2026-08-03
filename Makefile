COMPOSE := docker compose
COMPOSE_TEST := docker compose -f docker-compose.yml -f docker/compose/docker-compose.test.yml
SERVICES := model_optimizer deep_learning pretrained_models rag_api embeddings_service agent_orchestrator langgraph_crewai monitoring security_xai explainability

.PHONY: build up down logs test lint push ps clean config

build:
	$(COMPOSE) build --no-cache

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down --volumes

logs:
	$(COMPOSE) logs -f

test:
	@for service in $(SERVICES); do \
		echo "Testing $$service"; \
		$(COMPOSE_TEST) build $$service || exit 1; \
		$(COMPOSE_TEST) run --rm --no-deps $$service pytest -v || exit 1; \
	done

lint:
	@for service in $(SERVICES); do \
		echo "Linting $$service"; \
		$(COMPOSE_TEST) build $$service || exit 1; \
		$(COMPOSE_TEST) run --rm --no-deps $$service ruff check . || exit 1; \
	done

push:
	$(COMPOSE) push

ps:
	$(COMPOSE) ps

config:
	$(COMPOSE) config --quiet

clean:
	@echo "This removes all unused Docker data, including volumes."
	$(COMPOSE) down --volumes --remove-orphans
	docker system prune -af --volumes
