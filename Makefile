# Convenience targets. Override on the command line, e.g. make build LLAMA_CPP_REF=b12377
-include .env
LLAMA_CPP_REF ?= v1.0.0
CUDA_VERSION  ?= 13.0.3
CUDA_ARCHS    ?= 121
UBUNTU_VERSION ?= 24.04
TAG           := $(LLAMA_CPP_REF)-cuda$(CUDA_VERSION)
QWEN_MODEL    ?= Qwen3.8-27B-Q4_K_M.gguf
QWEN_CTX      ?= 65536
CHAT_MODEL    ?= google_gemma-4-26B-A4B-it-Q4_K_M.gguf
QWEN_COMPOSE  := docker compose -f compose.qwen.yaml --profile qwen
BENCH_ARGS    ?= -ngl 999 -fa 1 -p 512 -n 128 -d 0,16384
BUILD_ARGS    := --build-arg LLAMA_CPP_REF=$(LLAMA_CPP_REF) --build-arg CUDA_VERSION=$(CUDA_VERSION) --build-arg CUDA_ARCHS=$(CUDA_ARCHS)

.PHONY: pull build build-base build-server models smoke up-chat up-embed up down logs \
        models-qwen smoke-qwen up-qwen down-qwen logs-qwen bench-compare

pull:   ## pre-pull CUDA base images (with retries; Docker Hub can be flaky)
	@for img in nvidia/cuda:$(CUDA_VERSION)-devel-ubuntu$(UBUNTU_VERSION) nvidia/cuda:$(CUDA_VERSION)-runtime-ubuntu$(UBUNTU_VERSION); do \
	  for i in 1 2 3 4 5; do docker pull $$img && break; echo "retry $$i for $$img"; sleep 10; done; done

build: build-base build-server

build-base:
	docker build --target base $(BUILD_ARGS) -t llama-cpp-base:$(TAG) .

build-server:
	docker build --target server $(BUILD_ARGS) -t llama-cpp-server:$(TAG) .

models:
	scripts/download-models.sh all

smoke:
	scripts/smoke-test.sh $(TAG)

up-chat:
	docker compose --profile chat up -d
up-embed:
	docker compose --profile embed up -d
up:
	docker compose --profile all up -d
down:
	docker compose --profile all down
logs:
	docker compose --profile all logs -f --tail 100

# --- competitor: Qwen3.8-27B (compose.qwen.yaml, port 8098; runs alongside llm-chat) ---
models-qwen:
	scripts/download-models.sh qwen
smoke-qwen:
	CHAT_MODEL=$(QWEN_MODEL) CHAT_CTX=$(QWEN_CTX) scripts/smoke-test.sh $(TAG)
up-qwen:
	$(QWEN_COMPOSE) up -d
down-qwen:
	$(QWEN_COMPOSE) down
logs-qwen:
	$(QWEN_COMPOSE) logs -f --tail 100
bench-compare:  ## same llama-bench run on the Gemma and Qwen GGUFs (stop the services first for clean numbers)
	@for m in $(CHAT_MODEL) $(QWEN_MODEL); do \
	  echo "=== $$m"; \
	  docker run --rm --gpus all -v $(abspath $(or $(MODELS_DIR),models)):/models:ro llama-cpp-base:$(TAG) \
	    llama-bench -m /models/$$m $(BENCH_ARGS) || exit 1; done
