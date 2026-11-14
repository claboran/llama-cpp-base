# llama.cpp built from source with CUDA 13.
#
# Targets:
#   build   – toolchain + compiled llama.cpp (not meant to be run)
#   base    – slim CUDA runtime + all llama.cpp binaries   (llama-cpp-base)
#   server  – base with llama-server as entrypoint          (llama-cpp-server)
#
# Key build args:
#   LLAMA_CPP_REF  git tag / branch / commit of ggml-org/llama.cpp
#   CUDA_VERSION   nvidia/cuda image version (must be supported by the host driver)
#   CUDA_ARCHS     CMAKE_CUDA_ARCHITECTURES, e.g. "121" (GB10), "120", "89", "120;89"
#   CMAKE_EXTRA    additional -D flags for cmake

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.3
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

############################################################################
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION=14
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=v1.0.0
ARG CUDA_ARCHS=121
ARG CMAKE_EXTRA=""

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential \
        cmake ninja-build ccache git ca-certificates libssl-dev libgomp1 pkg-config python3

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION} \
    CCACHE_DIR=/ccache

# Shallow fetch of exactly one ref (works for tags, branches and full commit SHAs)
WORKDIR /src
RUN git init -q . && \
    git remote add origin "${LLAMA_CPP_REPO}" && \
    git fetch -q --depth 1 origin "${LLAMA_CPP_REF}" && \
    git checkout -q FETCH_HEAD && \
    git log -1 --format='%H %cI' > /tmp/commit

# GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS: the CPU backend is picked at runtime for the
# actual CPU (Grace/Neoverse on arm64, AVX2/AVX512 on x86_64) -> portable image.
RUN --mount=type=cache,target=/ccache \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        ${CMAKE_EXTRA} && \
    cmake --build build -j"$(nproc)" && \
    ccache -s

# Collect binaries + shared libs (ggml backends are dlopen'ed from the binary's directory)
RUN mkdir -p /opt/llama && \
    cp -P build/bin/* /opt/llama/ && \
    find build -name '*.so*' -exec cp -P {} /opt/llama/ \; && \
    printf 'LLAMA_CPP_REF=%s\nLLAMA_CPP_COMMIT=%s\nCUDA_VERSION=%s\nCUDA_ARCHS=%s\n' \
        "${LLAMA_CPP_REF}" "$(cut -d' ' -f1 /tmp/commit)" "${CUDA_VERSION}" "${CUDA_ARCHS}" \
        > /opt/llama/BUILD_INFO && \
    cat /opt/llama/BUILD_INFO && ls /opt/llama

############################################################################
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

ARG LLAMA_CPP_REF=v1.0.0
ARG CUDA_ARCHS=121
LABEL org.opencontainers.image.title="llama-cpp-base" \
      org.opencontainers.image.description="llama.cpp (CUDA) built from source" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.version="${LLAMA_CPP_REF}" \
      llama.cpp.cuda_archs="${CUDA_ARCHS}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libssl3 curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama /opt/llama

# Unprivileged runtime user; /models (read-only mount) and /cache (HF downloads via -hf)
RUN useradd --system --uid 10001 --create-home --home-dir /home/llama llama && \
    mkdir -p /models /cache && chown llama:llama /models /cache

ENV PATH=/opt/llama:${PATH} \
    LD_LIBRARY_PATH=/opt/llama:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH} \
    LLAMA_CACHE=/cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER llama
WORKDIR /home/llama
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["llama-cli", "--version"]

############################################################################
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=10m --retries=3 \
    CMD curl -fsS "http://localhost:${LLAMA_ARG_PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "llama-server"]
CMD []
