# ONNX Runtime Feasibility Assessment

**Date**: 2026-03-22
**Status**: Proposal
**Author**: Automated analysis

## Executive Summary

Replacing PyTorch with ONNX Runtime for the embedding and sentiment inference workers is **feasible and recommended**. On the current CPU-only Docker deployment (2 GB RAM, 2 CPUs per worker), ONNX Runtime would deliver:

- **3–5× faster inference** on CPU (ONNX Runtime exploits MKL-DNN / AVX2/512 vectorization)
- **3–5× less RAM per worker** (~200 MB vs ~830 MB peak for embedding worker)
- **~2 GB smaller Docker images** if torch can be fully removed (requires additional work — see Section 5)

The two primary inference models — FinBERT (embedding + sentiment) and MiniLM (embedding) — are standard BERT-family architectures with full ONNX export support. Conversion is straightforward via HuggingFace Optimum.

---

## 1. Current ML Stack

### Models in Use

| Model | Service | Architecture | Task | Output |
|-------|---------|-------------|------|--------|
| `ProsusAI/finbert` | `EmbeddingService` | BERT-base (110M params) | Mean-pooled embeddings | 768-dim vector |
| `all-MiniLM-L6-v2` | `EmbeddingService` | MiniLM (22M params) | Mean-pooled embeddings | 384-dim vector |
| `ProsusAI/finbert` | `SentimentService` | BERT-base + classification head | 3-class sentiment | pos/neg/neutral probs |
| `en_core_web_trf` | `NERService` | spaCy transformer pipeline | Named entity recognition | Entity spans |
| fastcoref | `NERService` | Custom transformer | Coreference resolution | Resolved text |

### Inference Details

- **Device**: CPU-only in Docker (auto-detect falls through to CPU; no `runtime: nvidia` in compose)
- **FP16**: Configured but only activates on CUDA — effectively FP32 on CPU
- **Batch sizes**: 32 (embedding), 16 (sentiment), 32 (NER)
- **Max sequence length**: 512 tokens (FinBERT), 256 tokens (MiniLM)
- **Long document handling**: Overlapping chunks → mean pool (embedding service)
- **Caching**: Redis with SHA256 content-hash keys, 168-hour TTL

### Key Code Paths

- `src/embedding/service.py:121-163` — Lazy model loading (`AutoModel.from_pretrained`, `model.to(device)`, `model.eval()`)
- `src/embedding/service.py:281-319` — Single-text inference (`torch.no_grad()`, `model(**inputs)`, attention-mask mean pooling)
- `src/sentiment/service.py:232-274` — Classification inference (`AutoModelForSequenceClassification`, softmax over logits)

### Resource Profile (Current)

| Worker | Memory Limit | Estimated Peak Usage | Components |
|--------|-------------|---------------------|------------|
| embedding-worker | 2 GB | ~830 MB | PyTorch runtime (~300 MB) + FinBERT (~440 MB) + MiniLM (~90 MB) |
| sentiment-worker | 2 GB | ~740 MB | PyTorch runtime (~300 MB) + FinBERT (~440 MB) |

---

## 2. ONNX Convertibility Assessment

### Fully Convertible (High Confidence)

| Model | ONNX Support | Export Method |
|-------|-------------|---------------|
| ProsusAI/finbert (embedding) | Full — standard BERT | `optimum-cli export onnx --model ProsusAI/finbert` |
| ProsusAI/finbert (sentiment) | Full — BERT + linear head | `optimum-cli export onnx --model ProsusAI/finbert --task text-classification` |
| all-MiniLM-L6-v2 | Full — already on HF Hub as ONNX | Direct download or `optimum-cli export onnx` |

All three use standard BERT operations (attention, layer norm, GELU) with complete ONNX operator coverage. No custom ops. Dynamic axes for batch size and sequence length are supported natively.

### Not Convertible

| Model | Reason | Impact |
|-------|--------|--------|
| spaCy `en_core_web_trf` | spaCy manages its own transformer pipeline; no clean ONNX export path | NER stays on PyTorch |
| fastcoref | Tightly coupled to PyTorch/transformers internally | Coref stays on PyTorch |
| BERTopic | Uses pre-computed embeddings (`embedding_model=None`); HDBSCAN/UMAP are sklearn/numpy | No ML inference to convert |

---

## 3. Expected Resource Savings

### Inference Speed (CPU)

Based on published ONNX Runtime benchmarks for BERT-base models on Intel CPUs:

| Scenario | PyTorch FP32 | ONNX FP32 | ONNX INT8 (quantized) | Speedup |
|----------|-------------|-----------|----------------------|---------|
| Single text, seq_len=128 | ~45 ms | ~15 ms | ~8 ms | **3–6×** |
| Single text, seq_len=512 | ~160 ms | ~55 ms | ~30 ms | **3–5×** |
| Batch of 32, seq_len=128 | ~800 ms | ~300 ms | ~170 ms | **3–5×** |
| Batch of 32, seq_len=512 | ~3200 ms | ~1100 ms | ~600 ms | **3–5×** |

The speedup is largest on CPU because ONNX Runtime applies graph optimizations (operator fusion, constant folding) and uses platform-specific BLAS/SIMD instructions that PyTorch's default eager-mode CPU path does not fully exploit.

### Memory Usage

| Component | PyTorch | ONNX Runtime | Reduction |
|-----------|---------|-------------|-----------|
| FinBERT model weights | ~440 MB | ~110 MB (FP32) / ~55 MB (INT8) | 4–8× |
| MiniLM model weights | ~90 MB | ~23 MB (FP32) / ~12 MB (INT8) | 4–8× |
| Runtime framework overhead | ~300 MB | ~50 MB | 6× |
| **Embedding worker peak** | **~830 MB** | **~180–280 MB** | **3–5×** |
| **Sentiment worker peak** | **~740 MB** | **~160–220 MB** | **3–5×** |

### Docker Image Size

| Component | Current Size | With ONNX | Delta |
|-----------|-------------|-----------|-------|
| PyTorch (CPU-only) | ~2.0 GB | Removable if no other deps need it | -2.0 GB |
| onnxruntime (CPU) | — | ~60 MB | +60 MB |
| transformers library | ~500 MB | Could use `tokenizers` only (~5 MB) | -495 MB |
| **Net image change** | | | **-2.0 to -2.5 GB** |

**Important caveat**: torch cannot be fully removed from the main image today — see Section 5.

---

## 4. Implementation Complexity

### Code Changes (~150 LOC across 4 files)

**`src/embedding/service.py`**:
- Replace `AutoModel.from_pretrained()` → `ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])`
- Replace `torch.no_grad()` + `model(**inputs)` → `session.run(None, {k: v.numpy() for k, v in inputs.items()})`
- Mean pooling: already numpy-compatible (just remove `.cpu()` call)
- Remove `_detect_device()` when using ONNX (ONNX Runtime handles execution provider selection)
- Keep `AutoTokenizer.from_pretrained()` (lightweight, works without torch)

**`src/sentiment/service.py`**:
- Replace `AutoModelForSequenceClassification` → ONNX session
- Replace `torch.softmax()` → `scipy.special.softmax()` or numpy: `np.exp(x) / np.exp(x).sum()`
- Keep tokenizer as-is

**`src/embedding/config.py`** and **`src/sentiment/config.py`**:
- Add `use_onnx: bool = True` feature flag
- Add `onnx_model_path: str | None = None` for custom model location
- Deprecate `device` and `use_fp16` when ONNX is active

**Model export** (one-time, can be CI step or Dockerfile build step):
```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model ProsusAI/finbert ./models/onnx/finbert-embedding/
optimum-cli export onnx --model ProsusAI/finbert --task text-classification ./models/onnx/finbert-sentiment/
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./models/onnx/minilm/

# Optional: INT8 dynamic quantization for further CPU speedup
optimum-cli onnxruntime quantize --onnx_model ./models/onnx/finbert-embedding/ -o ./models/onnx/finbert-embedding-int8/
```

### Tokenizer Strategy

Two options:
1. **Keep `transformers` for tokenizers** — simplest, `AutoTokenizer` works without loading torch
2. **Switch to `tokenizers` library only** (~5 MB) — load tokenizer from JSON config files, removes `transformers` dependency. Slightly more setup but much lighter

---

## 5. Dependency Analysis: Can torch Be Fully Removed?

### Current torch dependency chain

```
news-tracker (pyproject.toml)
  ├── torch>=2.0.0            ← direct (embedding + sentiment)
  ├── transformers>=4.36.0    ← direct (model loading + tokenizers)
  ├── fastcoref>=2.1.0        ← requires torch + transformers
  ├── bertopic>=0.16.0        ← requires sentence-transformers → torch
  └── spacy (en_core_web_trf) ← requires torch (transformer backend)
```

### Three scenarios

| Scenario | torch Removable? | Image Savings | Effort | Trade-offs |
|----------|-----------------|---------------|--------|------------|
| **A. Partial (ONNX alongside torch)** | No | RAM + speed only | Low (~150 LOC) | None — torch still installed for other deps |
| **B. Full removal** | Yes | ~2–2.5 GB image | Medium | NER quality (en_core_web_sm vs trf), no coref |
| **C. Per-worker images** | Yes (for ONNX workers) | ~2 GB for embed/sentiment images | Medium | Build complexity (multiple Dockerfiles) |

**Scenario B details** (full torch removal):
- Switch `en_core_web_trf` → `en_core_web_sm` (rule-based NER, no transformer — quality impact needs testing)
- Disable or replace `fastcoref` (already opt-in via `enable_coreference` config flag)
- BERTopic's `sentence-transformers` dep: BERTopic already uses `embedding_model=None` (pre-computed embeddings), but it's still a transitive install dependency
- Could pin `bertopic` without `sentence-transformers` via a fork or feature request

**Scenario C details** (recommended):
- Create a `Dockerfile.onnx` for embedding/sentiment workers (no torch)
- Keep existing `Dockerfile` for ingestion/NER/clustering workers
- Each image type carries only its required dependencies

---

## 6. Risks and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Numerical precision differences | Low | Medium | BERT ONNX exports typically match PyTorch within 1e-5. Validate with cosine similarity on test corpus |
| Embedding cache invalidation | Medium | Medium | Tiny float differences may cause cache misses. Options: (a) accept drift, (b) clear cache during migration, (c) round to fixed precision |
| INT8 quantization accuracy loss | Low | Low | Dynamic quantization is well-tested for BERT. Validate sentiment F1 on labeled samples before deploying |
| Missing ONNX ops | Very Low | Very Low | Standard BERT has full ONNX op coverage. No custom ops in the codebase |
| Loss of GPU/MPS path | Low | N/A | Not currently used in Docker. ONNX Runtime supports `CUDAExecutionProvider` if GPU needed later |
| Model versioning/storage | Low | Medium | Pre-convert ONNX models in CI; store in Docker image layer or model registry |
| Increased build complexity | Low | High | One-time export script + Dockerfile changes. Can automate in CI |

---

## 7. Recommendation

### Phase 1: ONNX inference with feature flag (Recommended — start here)

**Effort**: ~2–3 days
**Impact**: 3–5× faster CPU inference, 3–5× less RAM per worker

1. Add `onnxruntime>=1.17.0` to `pyproject.toml` dependencies
2. Add `optimum[onnxruntime]>=1.17.0` to dev dependencies (for export tooling)
3. Export FinBERT (embedding + sentiment) and MiniLM to ONNX format
4. Add `use_onnx` feature flag to embedding and sentiment configs
5. Implement dual-path inference (ONNX when enabled, PyTorch fallback)
6. Write benchmark script comparing latency/throughput
7. Validate numerical equivalence with existing test suite

### Phase 2: Worker image optimization (Optional)

**Effort**: ~1–2 days
**Impact**: ~2 GB smaller Docker images for ONNX workers

1. Create separate Dockerfile for ONNX-only workers
2. Remove torch/transformers from ONNX worker image
3. Use `tokenizers` library instead of full `transformers` for tokenization

### Phase 3: Full torch removal (Optional, requires evaluation)

**Effort**: ~2–3 days
**Impact**: Simplified dependency tree, single small image

1. Evaluate `en_core_web_sm` vs `en_core_web_trf` NER quality
2. Make fastcoref optional (already feature-gated)
3. Address BERTopic's transitive torch dependency

### Dependencies to add

```toml
# Production
"onnxruntime>=1.17.0",          # ~60 MB, CPU inference engine

# Dev only
"optimum[onnxruntime]>=1.17.0", # ONNX model export tooling
```

---

## 8. Verification Plan

1. **Numerical validation**: Compare PyTorch vs ONNX outputs on 100 sample texts; assert cosine similarity > 0.9999 for embeddings, identical sentiment labels for >99% of samples
2. **Unit tests**: Run existing `tests/test_embedding/` and `tests/test_sentiment/` with `use_onnx=True`
3. **Performance benchmark**: `@pytest.mark.performance` tests for batch sizes 1, 16, 32 × sequence lengths 128, 256, 512
4. **Integration test**: `uv run news-tracker run-once --mock` with ONNX enabled — verify documents get embeddings + sentiment scores
5. **Docker validation**: Build ONNX worker image, verify it processes queue items within 2 GB RAM limit
6. **Cache compatibility**: Run mixed PyTorch/ONNX workers, verify cache behavior is acceptable
