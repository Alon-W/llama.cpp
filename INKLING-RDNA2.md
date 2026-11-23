# Inkling 2 Mini on RDNA2 (Vulkan)

This branch (`inkling-rdna2`) is llama.cpp tuned to run Thinking Machines' **Inkling 2 Mini** fast on an
**AMD RX 6700 XT** (gfx1031, 12 GB, RADV Vulkan). It sits on top of the `inkling2` model support
(looped transformer, depth attention, per-head relative-position bias) and adds:

- Vulkan kernels and RDNA2 tuning for the quant types in the Unsloth `UD-Q2_K_XL` GGUF;
- fused ops for Inkling's depth attention and loop control;
- relative-position bias computed inside flash attention, so the ~800 MB per-head mask is never stored;
- the official **DSpark** drafter (speculative decoding) for Inkling 2 Mini, with a converter.

## Results

RX 6700 XT, `Inkling-2-Mini-UD-Q2_K_XL.gguf`, q8_0 KV cache, 32K context, 2 slots, llama-server router.
Each number is a router-gate median. Before is the first Inkling build here; after is this branch with ubatch 1024.

| | Before | After |
|---|---|---|
| Decode, 1 user | 44.8 tok/s | 52–53 tok/s |
| Prefill, 512-token prompt | 297 tok/s | ~420 tok/s |
| Prefill, 4K / 16K prompt | 373 / 369 tok/s | ~590 / ~585 tok/s |
| 23.5K-token prompt: time to first token | 67.2 s | 43.5 s |
| 23.5K-token prompt: decode | 38.6 tok/s | 47.3 tok/s |
| Compute buffer at 32K context (ubatch 512) | 1,023 MiB | 245 MiB |
| Coding agent (opencode, temp 0): decode with DSpark | 50.5 tok/s, no drafter | 71.1 tok/s (79% of drafts accepted) |

For comparison, on the same card and file, Unsloth's b12649 Vulkan build measured 40 tok/s decode
and about half this prefill.

**Quality.** Most changes produce the same logits as before (KL divergence 0.000000, same top token 100%).
A few change the order of float sums: the fused depth attention and the GQA flash-attention path.
Those measure KL ≈ 0.002 against the previous build, with perplexity within ±0.1%.
Speculative decoding keeps the target model's sampling distribution.

## What changed

**Vulkan (`ggml/src/ggml-vulkan`)**
- MoE matrix-vector (MMVQ): 8 rows per workgroup for the IQ2 types on RDNA2. Each workgroup copies the
  IQ2 lookup grid into shared memory, so fewer, larger workgroups copy it less often.
- NVFP4: one 32-bit load per 4 values in dequantization; 8-row float MoE path for nvfp4, q2_0 and mxfp4.
- Scalar flash attention: a per-head mask stride, so grouped-query attention works with per-head masks.
  RDNA2 prefill tiles are 16×64 with row split 4 at head size 128.
- Flash attention can take the relative-position bias table and distances (bindings 8–10) and build the mask
  per tile. Each value gets the same f16 rounding as a stored mask, so outputs are unchanged.
- New kernels for Inkling ops: depth scores and mix, loop gate and apply, rel-pos mask, fused SiTU and
  loop-control elementwise chains, L2 clip, conv step, ablation-form LoRA, and top-k MoE with sigmoid + bias + eps.

**ggml (`ggml/include/ggml.h`, `ggml/src/ggml.c`, CPU)**
- Ops with CPU reference implementations: `DEPTH_SCORES`, `DEPTH_MIX`, `LOOP_GATE`, `LOOP_APPLY`, `REL_POS_MASK`.
- `ggml_flash_attn_ext_set_rel_pos()`: a backend can gather the bias inside attention instead of reading a
  `[n_kv, n_q, n_head]` mask. The CPU backend refuses it. The graph uses it only when the GPU backend reports support.

**llama (`src/`)**
- `inkling2`: uses the fused ops. For batches over 8 tokens, attention takes the bias table directly when supported.
- DSpark drafter for Inkling 2: target features are tapped at visits 5, 23 and 41 plus the final norm, with a
  ring buffer of RNN features (`--spec-type draft-dflash`).
- `tools/inkling-dspark-convert`: converts the official `mtp.safetensors` and `config.json` to a drafter GGUF.

## Build

The same as upstream Vulkan builds:

```bash
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
```

## Run

Main model (server or router preset):

```ini
[inkling-2-mini]
model = Inkling-2-Mini-UD-Q2_K_XL.gguf
ctx-size = 32768
parallel = 2
kv-unified = true
ubatch-size = 1024
cache-type-k = q8_0
cache-type-v = q8_0
temp = 1.0
top-k = 0
top-p = 1.0
min-p = 0.0
```

Coding/agent entry with the DSpark drafter. It pays off on code edits and tool calls; on plain chat it is slower.

```ini
[inkling-2-mini-code]
; same as above, plus:
temp = 0.0
spec-type = draft-dflash
spec-draft-n-max = 3
model-draft = inkling-dspark-q8.gguf
gpu-layers-draft = 99
cache-type-k-draft = q8_0
cache-type-v-draft = q8_0
; the drafter injects prefill features in 64-row chunks, which keeps its compute buffer small
spec-draft-n-ubatch = 64
```

Making the drafter GGUF (354 MB, Q8_0) from the official release:

```bash
llama-inkling-dspark-convert mtp.safetensors config.json Inkling-2-Mini-UD-Q2_K_XL.gguf inkling-dspark-q8.gguf
```

## Tried and rejected

| Experiment | Why it was rejected |
|---|---|
| Query-chunked attention to shrink the mask | Prefill 5–11% slower, and the compute buffer grew |
| DSpark on general chat at temp 1.0 | ~30% of drafts accepted; 13% slower |
| 4 or more drafts | Verify cost grows linearly with tokens: each token reads its own 8 experts |
| q2_0 MoE through the 8-bit activation path | Not lossless (KL 0.002) |
| Unsloth's padded-vocab drafter | Its markov table has 201,024 rows; the tokenizer has 200,058 |
