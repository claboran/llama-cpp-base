# Compatibility log

One row per tested combination. Host: MSI EdgeExpert, GB10 (sm_121), driver 580.178.04.

| Date       | llama.cpp ref (commit) | CUDA image | Archs | Model                         | Result | Notes |
|------------|------------------------|------------|-------|-------------------------------|--------|-------|
| 2026-11-14 | v1.0.0 (b1b770d)       | 13.0.3     | 121   | — (build + device check)      | ✅     | build ~2.5 min; `CUDA0: NVIDIA GB10` listed |
| 2026-11-14 | v1.0.0 (b1b770d)       | 13.0.3     | 121   | bge-m3 Q8_0 (ggml-org)        | ✅     | 1024 dims, L2-normalized, GGUF `pooling_type=2` (CLS) |
| 2026-11-14 | v1.0.0 (b1b770d)       | 13.0.3     | 121   | Gemma 4 26B-A4B Q4_K_M (bartowski) | ✅ | pp512 3167 t/s, tg128 69.8 t/s; @16k depth: pp 2568, tg 60.5; server ~65 t/s |

## Memory (projected device memory, `llama-fit-params`, 1 slot, flash-attn on)

| Model / setting                          | Memory   |
|------------------------------------------|----------|
| Gemma 4 26B-A4B Q4_K_M, ctx 64k, KV f16  | 18.4 GB  |
| Gemma 4 26B-A4B Q4_K_M, ctx 128k, KV f16 | 19.8 GB  |
| Gemma 4 26B-A4B Q4_K_M, ctx 128k, KV q8_0| 18.5 GB  |
| bge-m3 Q8_0, ctx/batch 8192 (measured via nvidia-smi after a 7.5k-token request) | ~0.95 GB |

## Findings / gotchas

- **Gemma 4 thinks by default**: output goes to `reasoning_content` first. Give enough `max_tokens`
  or disable per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- **llama-server rejects `--opt=value`** (v1.0.0): pass option and value as separate argv items.
- `--embeddings` server forces `n_batch = n_ubatch`; set both to the max input length (8192 for bge-m3).
- Version string shows `1.0.0-dev (build 1)` because of the shallow clone; `BUILD_INFO` in the image has the real ref/commit.
- `llama-embedding` example binary is not in the image (`LLAMA_BUILD_EXAMPLES=OFF`); use `llama-server --embeddings`.
- CPU backend picked at runtime on GB10: `libggml-cpu-armv8.6_2.so`.
- Docker Hub pulls are flaky on this network (TLS handshake timeouts) → `make pull` retries.
