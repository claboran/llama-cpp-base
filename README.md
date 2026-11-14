# llama-cpp-base

Build [llama.cpp](https://github.com/ggml-org/llama.cpp) from source with **CUDA 13** into small,
reusable Docker images, and run a **chat model** and an **embedding model** as OpenAI-compatible
API services with Docker Compose.

- Multistage build: full CUDA toolchain only in the build stage, slim CUDA runtime image in the end
- Pinned llama.cpp version, GPU architecture and CUDA version as build arguments
- Portable CPU backend (picked at runtime), builds on **arm64** and **x86_64**
- Compose services for chat (Gemma 4 26B-A4B) and embeddings (bge-m3), with health checks,
  optional API key and `.env`-based configuration
- Smoke test script that checks GPU access, benchmarks the model and calls both APIs

Developed and tested on an NVIDIA **GB10** (Grace Blackwell, arm64, compute capability 12.1).
Results per tested combination are in [COMPAT.md](COMPAT.md).

---

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Building for other GPUs / versions](#building-for-other-gpus--versions)
- [Using the API](#using-the-api)
- [Security](#security)
- [Performance](#performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Licenses](#licenses)

---

## Architecture

```
 nvidia/cuda:<ver>-devel-ubuntu24.04          nvidia/cuda:<ver>-runtime-ubuntu24.04
 ┌──────────────────────────────────┐         ┌───────────────────────────────────┐
 │ build                            │         │ base     → llama-cpp-base         │
 │  gcc 14, cmake, ninja, ccache    │  /opt/  │  all llama.cpp binaries + libs    │
 │  git fetch llama.cpp @ REF       │ ──────► │  non-root user, tini, /models     │
 │  cmake -DGGML_CUDA=ON            │  llama  ├───────────────────────────────────┤
 │        -DCMAKE_CUDA_ARCHITECTURES│         │ server   → llama-cpp-server       │
 │        -DGGML_CPU_ALL_VARIANTS   │         │  ENTRYPOINT llama-server          │
 └──────────────────────────────────┘         │  HEALTHCHECK /health              │
                                              └───────────────┬───────────────────┘
                                                              │ docker compose
                                              ┌───────────────┴───────────────────┐
                                              │ llm-chat   :8099  Gemma 4 26B-A4B │
                                              │ llm-embed  :8901  bge-m3          │
                                              └───────────────────────────────────┘
```

| Image                            | Contents |
|----------------------------------|----------|
| `llama-cpp-base:<ref>-cuda<ver>`   | CUDA runtime + all llama.cpp tools: `llama-cli`, `llama-server`, `llama-bench`, `llama-quantize`, `llama-perplexity`, `llama-fit-params`, … (`/opt/llama/BUILD_INFO` records ref, commit, CUDA version, archs) |
| `llama-cpp-server:<ref>-cuda<ver>` | `base` with `llama-server` as entrypoint and a `/health` health check |

The **base** image is meant for experiments and tooling (`docker run … llama-bench …`), the
**server** image for long-running services.

## Requirements

| Component | Version / note |
|-----------|----------------|
| NVIDIA GPU | compute capability ≥ 7.5 (CUDA 13 dropped older architectures) |
| NVIDIA driver | ≥ 580 for CUDA 13.0 (check with `nvidia-smi`) |
| Docker Engine | with BuildKit (default since Docker 23) |
| Docker Compose | v2 plugin (`docker compose`) |
| NVIDIA Container Toolkit | `docker run --rm --gpus all ubuntu nvidia-smi` must work |
| Disk | ~15 GB for CUDA images, ~18 GB for the default models |

GPU memory for the default setup: about **20 GB** (chat model with 64k context plus embedding
model), see [Performance](#performance).

Registering `nvidia` as a Docker runtime is **not** needed; `--gpus all` / `gpus: all` is enough.

Your user needs access to the Docker daemon:

```bash
sudo usermod -aG docker "$USER"
# then log out completely and back in (or reboot), see Troubleshooting
```

## Quick start

```bash
git clone <this-repo> llama-cpp-base && cd llama-cpp-base
cp .env.example .env              # adjust if needed (ports, context size, API key, …)

make pull                         # pull CUDA base images (with retries)
make build                        # build llama-cpp-base and llama-cpp-server images
make models                       # download the default GGUF models into ./models
make smoke                        # test drive: GPU, benchmark, chat + embeddings
make up                           # start both services
```

Then:

- Chat API + web UI: `http://<host>:8099`
- Embeddings API: `http://<host>:8901`

Other targets: `make up-chat`, `make up-embed`, `make logs`, `make down`.
Without `make`, the same works with `docker build --target base|server …` and
`docker compose --profile all up -d`.

### Default models

| Service | Model | File | Size |
|---------|-------|------|------|
| chat | [Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (MoE, ~4B active params) | [`bartowski/google_gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF) Q4_K_M | 17 GB |
| embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1024 dims, up to 8192 tokens, multilingual) | [`ggml-org/bge-m3-Q8_0-GGUF`](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) Q8_0 | 0.6 GB |

`scripts/download-models.sh` resumes interrupted downloads and verifies the SHA-256 checksums.
For gated repositories set `HF_TOKEN`. Any other GGUF works: put it into `./models` and set
`CHAT_MODEL` / `EMBED_MODEL` in `.env`.

## Configuration

All runtime settings live in `.env` (template: [`.env.example`](.env.example)).

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLAMA_CPP_REF` | `v1.0.0` | llama.cpp tag, branch or commit to build |
| `CUDA_VERSION` | `13.0.3` | `nvidia/cuda` image version (must be supported by the host driver) |
| `CUDA_ARCHS` | `121` | GPU architecture(s), see below |
| `API_KEY` | *(empty)* | Bearer token(s), comma-separated; empty = no authentication |
| `CORS_ORIGINS` | `*` | allowed browser origins (`*`, `localhost`, or a list) |
| `MODELS_DIR` | `./models` | host directory with the GGUF files (mounted read-only) |
| `CHAT_MODEL` | `google_gemma-4-26B-A4B-it-Q4_K_M.gguf` | chat model file |
| `CHAT_ALIAS` | `gemma-4-26b-a4b` | model name reported by the API |
| `CHAT_CTX` | `65536` | context size in tokens (shared by all slots) |
| `CHAT_PARALLEL` | `1` | parallel request slots |
| `CHAT_CACHE_TYPE` | `f16` | KV cache type: `f16`, `q8_0`, `q4_0`, `f8_e4m3` |
| `CHAT_BIND` / `CHAT_PORT` | `0.0.0.0` / `8099` | host address and port for the chat API |
| `EMBED_MODEL` | `bge-m3-q8_0.gguf` | embedding model file |
| `EMBED_ALIAS` | `bge-m3` | model name reported by the API |
| `EMBED_CTX` | `8192` | max input length; also used as batch size (required for embeddings) |
| `EMBED_PARALLEL` | `1` | parallel request slots |
| `EMBED_BIND` / `EMBED_PORT` | `0.0.0.0` / `8901` | host address and port for the embeddings API |

Further llama-server options can be added to `command:` in [`compose.yaml`](compose.yaml).
Write option and value as **separate list items** (`- --ctx-size` / `- "65536"`);
`--ctx-size=65536` is rejected by llama-server.

## Building for other GPUs / versions

```bash
make build LLAMA_CPP_REF=b12377            # newer llama.cpp build
make build CUDA_VERSION=13.5.0             # newer CUDA (driver must support it)
make build CUDA_ARCHS="89;120"             # several architectures in one image
```

Common values for `CUDA_ARCHS` ([full list](https://developer.nvidia.com/cuda-gpus)):

| Value | GPUs |
|-------|------|
| `121` | GB10 (DGX Spark and similar) |
| `120` | RTX 50 series, RTX PRO Blackwell |
| `100` | B200 / GB200 |
| `90`  | H100 / H200 / GH200 |
| `89`  | RTX 40 series, RTX 4000/6000 Ada, L4, L40S |
| `86`  | RTX 30 series, A10, A40 |
| `80`  | A100 |

Image tags contain the llama.cpp ref and CUDA version (`v1.0.0-cuda13.0.3`), but not the
architecture. Use different tags when you build for several GPUs on one host.

The build uses BuildKit cache mounts (apt, ccache), so rebuilds after changing the ref are much
faster. A full build on a GB10 takes about 2–3 minutes.

## Using the API

Both services speak the OpenAI API (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`) plus llama.cpp's own endpoints (`/health`, `/metrics`, `/props`,
`/tokenize`, …). If `API_KEY` is set, add `-H "Authorization: Bearer $API_KEY"`.

### Chat

```bash
curl http://localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Explain KV caching in two sentences."}],
  "max_tokens": 1024
}'
```

Gemma 4 **thinks before answering** by default. The thinking text is returned separately in
`choices[0].message.reasoning_content`, the answer in `content`. Give it enough `max_tokens`,
or switch thinking off per request:

```json
{ "messages": [...], "chat_template_kwargs": { "enable_thinking": false } }
```

Streaming (`"stream": true`), tool calls and JSON output (`response_format`) work as with the
OpenAI API (the server runs with `--jinja`).

### Embeddings

```bash
curl http://localhost:8901/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["The cat sits on the mat.", "Eine Katze sitzt auf der Matte."]}'
```

Vectors have 1024 dimensions and are L2-normalized, so cosine similarity is a dot product.
bge-m3 uses CLS pooling (`--pooling cls`, matches the model's GGUF metadata). Only the dense
embedding of bge-m3 is provided, not its sparse or multi-vector outputs.

### Python (OpenAI client)

```python
from openai import OpenAI

chat = OpenAI(base_url="http://localhost:8099/v1", api_key="<API_KEY or anything>")
r = chat.chat.completions.create(
    model="gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)

emb = OpenAI(base_url="http://localhost:8901/v1", api_key="<API_KEY or anything>")
v = emb.embeddings.create(model="bge-m3", input=["hello world"]).data[0].embedding
print(len(v))  # 1024
```

## Security

By default both services listen on **all interfaces without authentication**. Before they are
reachable from an untrusted network:

1. **Set an API key** in `.env`: `API_KEY=$(openssl rand -hex 32)`. Requests without it get
   `401`; `/health` stays open for health checks.
2. **Restrict CORS** (`CORS_ORIGINS`) if browsers on other sites should not call the API.
3. **Firewall**: ports published by Docker bypass `ufw`/`firewalld` rules on the host. To
   limit access, bind to a specific address (`CHAT_BIND=127.0.0.1` or a VPN/LAN IP) or put a
   reverse proxy with TLS in front.
4. The containers run as an unprivileged user and mount the models read-only.

## Performance

Measured on NVIDIA GB10, llama.cpp `v1.0.0`, CUDA 13.0.3, flash attention on
(`llama-bench`, tokens per second):

| Model | Prompt (512 tokens) | Generation | Prompt @ 16k context | Generation @ 16k context |
|-------|--------------------:|-----------:|---------------------:|-------------------------:|
| Gemma 4 26B-A4B Q4_K_M | 3167 | 69.8 | 2568 | 60.5 |

GPU memory (projected by `llama-fit-params`, one slot):

| Setting | Memory |
|---------|-------:|
| Gemma 4 26B-A4B Q4_K_M, 64k context, KV f16 | 18.4 GB |
| Gemma 4 26B-A4B Q4_K_M, 128k context, KV f16 | 19.8 GB |
| Gemma 4 26B-A4B Q4_K_M, 128k context, KV q8_0 | 18.5 GB |
| bge-m3 Q8_0, 8192 tokens | ~1 GB |

Gemma 4 uses sliding-window attention for most layers, so longer contexts cost little memory.

Benchmark your own setup:

```bash
docker run --rm --gpus all -v ./models:/models:ro llama-cpp-base:v1.0.0-cuda13.0.3 \
  llama-bench -m /models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf -ngl 999 -fa 1 -d 0,16384
```

## Testing

`make smoke` (or `scripts/smoke-test.sh <tag>`) runs, using plain `docker run` and temporary
ports 18080/18081 on localhost:

1. GPU visible in the container (`nvidia-smi`)
2. Build info and version
3. llama.cpp lists the CUDA device
4. `llama-bench` on the chat model (empty and 16k context)
5. Chat server answers a question correctly
6. Embedding server returns 1024-dim normalized vectors with sensible similarities

Steps 4–6 are skipped if the model files are missing. Record new results in
[COMPAT.md](COMPAT.md).

## Troubleshooting

**`permission denied while trying to connect to the docker API`**
Your shell does not have the `docker` group yet. Check with `id`; if `docker` is missing, log
out of **all** sessions (including tmux/screen and the desktop session) or reboot. For the
current shell only: `newgrp docker`.

**Pulling `nvidia/cuda` images fails with TLS handshake timeouts**
Retry; `make pull` does this automatically. Once pulled, builds use the local images.

**Chat answers are empty, `finish_reason: "length"`**
The model spent `max_tokens` on thinking. Raise `max_tokens` or disable thinking (see above).

**Container keeps restarting with `error: invalid argument: --x=y`**
Split option and value into separate items in `compose.yaml`.

**`CUDA driver version is insufficient`**
The host driver is older than the CUDA version of the image. Use a lower `CUDA_VERSION` or
update the driver.

**`no CUDA devices found` / model runs on CPU**
Check `docker run --rm --gpus all ubuntu nvidia-smi`. If that fails, install or configure the
NVIDIA Container Toolkit. Make sure `CUDA_ARCHS` matches your GPU.

## Repository layout

```
├── Dockerfile               # multistage: build → base → server
├── compose.yaml             # llm-chat and llm-embed services
├── .env.example             # configuration template (copy to .env)
├── Makefile                 # pull / build / models / smoke / up / down / logs
├── scripts/
│   ├── download-models.sh   # resumable, checksum-verified GGUF download
│   └── smoke-test.sh        # test drive
├── COMPAT.md                # tested combinations, benchmarks, findings
├── PLAN.md                  # original planning notes
└── models/                  # GGUF files (not in git)
```

## Licenses

This repository contains build and configuration files only. The software and models it uses
come with their own licenses:

- llama.cpp: [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)
- NVIDIA CUDA container images: [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license)
- Gemma 4: see the [model card](https://huggingface.co/google/gemma-4-26B-A4B-it) for its terms of use
- bge-m3: [MIT](https://huggingface.co/BAAI/bge-m3)
