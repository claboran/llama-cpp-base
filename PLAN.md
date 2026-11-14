# llama.cpp Base Images for MSI EdgeExpert (GB10) — Plan

Goal: build llama.cpp from source into reusable Docker base images tuned for this
machine, then run concrete services (chat, embeddings) via Docker Compose.

## 1. Target platform (verified on host, 2026-11-14)

| Item              | Value                                         |
|-------------------|-----------------------------------------------|
| Machine           | MSI EdgeExpert, NVIDIA GB10 (Grace Blackwell) |
| CPU arch          | `aarch64` / `linux/arm64`, 20 cores           |
| Memory            | 121 GiB unified (CPU+GPU shared)              |
| GPU compute cap.  | 12.1 → `CMAKE_CUDA_ARCHITECTURES=121`         |
| Driver            | 580.178.04                                    |
| Host CUDA         | 13.0 (nvcc V13.0.88)                          |
| OS                | Ubuntu 24.04.5 LTS                            |
| Docker / Compose  | 29.6.2 / v5.2.0                               |
| Container toolkit | nvidia-ctk 1.20.1                             |

Host prep done: user added to the `docker` group. No `nvidia` runtime is registered
(default `runc`), but GPU access works via CDI (`/var/run/cdi/nvidia.yaml`):
`docker run --rm --gpus all ubuntu:24.04 nvidia-smi` → `NVIDIA GB10, 580.178.04`.
Docker Hub pulls hit transient TLS/connection-reset errors; retries succeeded.

## 2. Image architecture

Multistage `Dockerfile`, all stages `linux/arm64`, based on official NVIDIA CUDA 13 images.

```
nvidia/cuda:13.0.x-devel-ubuntu24.04          nvidia/cuda:13.0.x-runtime-ubuntu24.04
        │                                              │
   [build]  cmake, ninja, git, ccache,                 │
            libcurl-dev, libssl-dev                    │
            clone llama.cpp @ pinned tag/commit        │
            cmake -G Ninja -DGGML_CUDA=ON              │
              -DCMAKE_CUDA_ARCHITECTURES=121           │
              -DGGML_NATIVE=OFF (arm flags explicit)   │
              -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=ON   │
            cmake --install → /opt/llama               │
        │                                              │
        └──────── COPY /opt/llama ──────────────► [runtime]  = llama-cpp-base
                                                       │   libcurl, libgomp, ca-certs, curl
                                                       │   non-root user, /models volume
                                                       │   bins: llama-server, llama-cli,
                                                       │         llama-bench, llama-quantize …
                                                       │
                                               ┌───────┴────────┐
                                           [server]          (later: other
                                     ENTRYPOINT llama-server  tools/variants)
                                     HEALTHCHECK /health
```

Build args to expose: `LLAMA_CPP_REF` (tag/commit), `CUDA_VERSION`,
`CUDA_ARCHS` (default `121`), `UBUNTU_VERSION`, extra `CMAKE_ARGS`.

Tags, e.g. `llama-cpp-base:<llama-ref>-cuda13.0-arm64` and `llama-cpp-server:<…>`.

Points to verify during the build:
- Which exact `nvidia/cuda:13.0.*` tag exists for arm64 / ubuntu24.04 (match host 13.0 or newer minor that the 580 driver supports).
- sm_121 support in CUDA 13.0 nvcc (expected yes); fall back to `121a`/`120` if needed.
- Grace CPU flags: `GGML_NATIVE=OFF` with explicit `-mcpu=neoverse-v2` vs. `GGML_NATIVE=ON` (only valid if built on this host — fine here, but less portable).
- Use ccache / BuildKit cache mounts to make rebuilds fast.

## 3. Runtime layer: Docker Compose

`compose.yaml` builds/uses the base image and defines concrete services:

| Service       | Purpose                     | Key flags                                                                 |
|---------------|-----------------------------|---------------------------------------------------------------------------|
| `llm-chat`    | OpenAI-compatible chat API  | `-m /models/<chat>.gguf -ngl 999 -c <ctx> --jinja -fa on --host 0.0.0.0 --port 8080` |
| `llm-embed`   | Embeddings API              | `-m /models/<embed>.gguf --embeddings --pooling <mean/cls> -ngl 999 --port 8081` |

Common to all services:
- `gpus: all` (or `deploy.resources.reservations.devices`), `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- Model dir bind-mounted read-only: `./models:/models:ro` (or `-hf` download into a cache volume)
- `healthcheck` on `/health`, `restart: unless-stopped`
- Config via `.env` (model file, context size, ports, parallel slots)
- Profiles (`chat`, `embed`) so services can be started individually.

Unified memory note: with 121 GiB shared, run models fully offloaded (`-ngl 999`);
keep `--mlock` / `--no-mmap` as tunables to test.

## 3a. Models (first choice)

| Role       | Model                                  | GGUF source (candidates)                                   | Settings |
|------------|----------------------------------------|------------------------------------------------------------|----------|
| Chat       | `google/gemma-4-26B-A4B-it` (MoE, ~4B active) | `ggml-org/gemma-4-26B-A4B-it-GGUF`, `unsloth/…-GGUF`, `bartowski/google_gemma-4-26B-A4B-it-GGUF` | ctx 65536 → try 131072; `--jinja`; `-ngl 999`; quant Q4_K_M |
| Embeddings | `BAAI/bge-m3` (1024-dim dense, 8192 tok max) | community GGUFs (e.g. `KimChen/bge-m3-GGUF`) or convert ourselves with `convert_hf_to_gguf.py` | `--embeddings --pooling cls`, `-c 8192`, `-ub 8192` (batch must hold a full input) |

Notes / things to verify:
- Gemma 4: chat template must be correct → use `--jinja` (check whether the GGUF template is current or needs `--chat-template-file`).
- Gemma 4 MTP (speculative decoding) is reportedly not in llama.cpp master yet → out of scope for now.
- bge-m3 pooling: BGE-M3 dense embedding = CLS pooling; verify GGUF metadata
  (`bert.pooling_type`) and compare against a reference vector from `sentence-transformers` to be sure.
- Embedding output should be L2-normalized (llama-server does this by default for `/v1/embeddings`; verify).
- Context 64k vs 128k: measure KV-cache memory and tokens/s for both (`llama-bench -d`), keep as `.env` setting.

## 3b. llama.cpp version strategy

"Find out what works": start from the **latest release tag** (`bNNNN`, Gemma 4 support exists since at
least `b8746`), build it, run the test drive with both models. Keep a small compatibility log
(`COMPAT.md`: llama.cpp ref × CUDA version × model → works / issue / tok/s). Pin `LLAMA_CPP_REF` to the
newest tag that passes; bump deliberately by rebuilding with a new tag and re-running the smoke test.

## 4. Test drive (before Compose)

1. **Build**: `docker buildx build --platform linux/arm64 --target runtime -t llama-cpp-base:dev .`
2. **GPU visible**: `docker run --rm --gpus all llama-cpp-base:dev nvidia-smi`
3. **Binary sanity**: `docker run --rm --gpus all llama-cpp-base:dev llama-cli --version` and
   `llama-server --list-devices` → must show the CUDA GB10 device.
4. **Smoke inference** with a small model (e.g. Qwen 0.5B–1.5B Q4 GGUF):
   `docker run --rm --gpus all -v ./models:/models llama-cpp-base:dev llama-cli -m /models/<small>.gguf -p "Hello" -n 32 -ngl 999`
   → check log for layers offloaded to CUDA.
5. **Benchmark**: `llama-bench -m /models/<small>.gguf -ngl 999` → record pp/tg tokens/s as baseline.
6. **Server**: run `llama-server`, then
   `curl localhost:8080/health` and `curl localhost:8080/v1/chat/completions …`.
7. **Embeddings**: start with `--embeddings`, `curl localhost:8081/v1/embeddings …`.

Only after this passes → wire up `compose.yaml`.

## 5. Planned repository layout

```
llama-cpp-base/
├── PLAN.md
├── README.md               # usage, build, run
├── Dockerfile              # multistage: build → runtime → server
├── .dockerignore
├── compose.yaml            # llm-chat, llm-embed services
├── .env.example            # model paths, ports, ctx size, LLAMA_CPP_REF
├── Makefile                # build / test / up / down / bench shortcuts
├── scripts/
│   └── smoke-test.sh       # steps 2–7 of the test drive, automated
└── models/                 # .gitignore'd, GGUF files
```

## 6. Milestones

- [x] M0 – Host prep: docker group, GPU-in-container check (CDI)
- [x] M0b – `git init`, `.gitignore`
- [x] M1 – Dockerfile build stage compiles llama.cpp with CUDA 13 for sm_121 (v1.0.0, CUDA 13.0.3, ~2.5 min)
- [x] M2 – Runtime image (4.5 GB); GPU visible; `--list-devices` shows GB10
- [x] M3 – Test drive: smoke inference + llama-bench baseline recorded (see COMPAT.md)
- [x] M4 – `compose.yaml` with chat + embedding services, healthchecks, `.env`
- [x] M5 – README, Makefile, smoke-test script; llama.cpp pinned to `v1.0.0`

## 7. Open questions

- ~~Which models first?~~ → Gemma 4 26B-A4B (chat), bge-m3 (embeddings), see 3a
- ~~Pin or track master?~~ → empirical: latest tag, pin what passes, see 3b
- Push images to a registry, or local-only?
- Need other backends later (e.g. `llama-swap` for multi-model routing, reranking endpoint)?
