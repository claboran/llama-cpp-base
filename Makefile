# Convenience targets. Override on the command line, e.g. make build LLAMA_CPP_REF=b12377
-include .env
LLAMA_CPP_REF ?= v1.0.0
CUDA_VERSION  ?= 13.0.3
CUDA_ARCHS    ?= 121
UBUNTU_VERSION ?= 24.04
TAG           := $(LLAMA_CPP_REF)-cuda$(CUDA_VERSION)
BUILD_ARGS    := --build-arg LLAMA_CPP_REF=$(LLAMA_CPP_REF) --build-arg CUDA_VERSION=$(CUDA_VERSION) --build-arg CUDA_ARCHS=$(CUDA_ARCHS)

.PHONY: pull build build-base build-server models smoke up-chat up-embed up down logs

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
