# GLM-5.2 DCP post-#35 research snapshot

> **Research archive, not a merge proposal.** The paths in this branch are
> disabled by default and produced negative end-to-end throughput results.
> Do not include them in a release image without a new design and matched
> validation.

This branch preserves the follow-up investigation performed after
[rtx6kpro issue #35](https://github.com/local-inference-lab/rtx6kpro/issues/35).
Issue #35 established the retained DCP prefill design: exact row-owner top-k,
partial indexer replication, and a bounded depth-1 full-CKV prefetch. This
follow-up asked whether communication could be reduced further by retaining
query ownership through attention or by making sparse attention consume only
selected remote CKV records.

The answer for the tested GLM-5.2 PCIe stack was no. All tested variants were
numerically correct, but each moved more data or introduced more lifecycle
work than the existing local-CKV decode path.

## Exact source identity

- vLLM base: `dev/gilded-gnosis` at
  `89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58`.
- This branch:
  `research/glm52-dcp-post35-poc-20260726`.
- SparkInfer dependency branch:
  [`research/glm52-dcp-post35-poc-20260726`](https://github.com/local-inference-lab/sparkinfer/tree/research/glm52-dcp-post35-poc-20260726).
- Measured local image:
  `local/vllm:gilded-gnosis-selected-record-consistent-bulk-poc-20260726`.
- The measured image was mechanically compared with both source worktrees;
  every overlaid runtime file matched.

The branch intentionally also contains the earlier prefill experiments from
which the clean production PRs were extracted. The final commit adds the
post-#35 direct remote-record consumers. Use the commit history to separate
the phases.

## What was tested

### Query ownership retained through attention

The normal optimized prefill path distributes indexer scoring, merges exact
top-k, gathers CKV, and then restores the normal query/head layout. The POC
kept row ownership through attention to avoid the final top-k restoration.
Both an NCCL implementation and a SparkInfer transpose implementation were
bit-exact.

This did not reduce the dominant communication. It introduced two full BF16
activation exchanges per sparse layer. Skipping the final top-k gather had no
measurable effect.

| Path | 64k prefill | 400k prefill | Delta at 64k | Delta at 400k |
|---|---:|---:|---:|---:|
| Replicated-indexer reference | 5,810 | 5,006.3 | reference | reference |
| Sharding through attention | 4,328 | 3,798.3 | -25.5% | -24.1% |
| Same, skip final top-k gather | 4,325 | 3,798.7 | -25.6% | -24.1% |

### Materialized selected-record decode

Each rank formed the exact union of top-k token positions, exchanged native
368-byte or 656-byte CKV records, materialized a dense selected-record slab,
and ran sparse attention over that slab. Copy-engine, direct peer-write,
per-layer prefetch, and three-layer bulk-prefetch variants were implemented.

The stock decode path is already communication-efficient: CKV remains local,
and ranks exchange compact query, LSE, and output tensors. Selected-record
decode instead moved up to 2,048 complete records per destination plus routing
metadata. Observed PCIe traffic rose from about 13-14 GB/s to 31-33 GB/s.

TP4/DCP4 NF3, active per-user decode:

| MTP | Path | ctx0 | ctx64k | ctx256k |
|---:|---|---:|---:|---:|
| 0 | Stock local CKV | 57.72 | 56.34 | 55.61 |
| 0 | Packet copy engine | 51.52 | 44.23 | 29.39 |
| 0 | Packet direct peer write | 28.9 | 17.7 | 16.6 |
| 0 | Materialized per layer | 51.2 | 48.8 | 46.5 |
| 0 | Materialized three-layer bulk | 51.16 | 48.5 | 46.9 |
| 3 | Stock local CKV | 107.44 | 102.75 | not measured |
| 3 | Materialized per layer | 87.84 | 86.07 | not measured |
| 3 | Materialized three-layer bulk | 92.11 | 88.09 | not measured |

Bulk prefetch improved the rejected MTP3 selected-record path by about 4.9%,
but remained about 14% below stock.

### Direct remote pointer consumption

The final POC removed selected-record materialization. SparkInfer supplied an
address table that pointed either into copy-engine packet storage, peer
staging storage, or the model-owned remote KV cache. Sparse MLA loaded records
through those addresses. The implementation included graph-stable pointer
rings, explicit release ordering, persistent storage mappings, native
FP8-RoPE records, and exact eager/graph replay tests.

This was correct but slower. Indirect remote loads destroyed the coalesced
local access pattern and paid PCIe latency inside the attention kernel. Direct
model-storage pointers measured 29.5/18.2/17.1 tok/s at ctx0/64k/256k, only a
small improvement over packet peer pointers and far below stock.

## Why the hypothesis failed

Logical sparsity is not enough. The useful cost comparison is:

1. bytes crossing PCIe;
2. number of launches and synchronization points;
3. whether the consumer reads contiguous local memory; and
4. whether work is repeated between draft layers or query tiles.

For decode, the stock algorithm wins because it communicates small dense
state and leaves CKV sharded. Selected-record transport communicates sparse
positions but large records. Direct pointer consumption saves a copy while
turning the attention kernel into an irregular remote-memory consumer.

For prefill, selected-record unions were also too dense. For one 8,192-row
chunk at TP8/DCP4, union coverage was 97.6% at 64k, 86.1% at 128k, 70.0% at
256k, and 58.2% on the final 400k chunk. Measured break-even for selected
transport was about 40-45%, before accounting for repeated records across
smaller query tiles.

## Correctness evidence

- vLLM selected-record policy suite: 134 passed, 1 skipped.
- vLLM topology/indexer suite: 80 passed, 17 skipped.
- CUDA runtime-row union test: 1 passed.
- SparkInfer contiguous/pointer sparse-MLA equivalence: 3 passed.
- SparkInfer world-size-2 368-byte storage-pointer eager and alternating CUDA
  graph replay: 1 passed in 65.73 seconds.
- E2E output remained coherent for all reported paths.

Passing correctness tests does not make these paths release candidates. The
matched E2E regressions are the rejection criterion.

## Reproduction

Machine-readable measurements are in
[`benchmarks/research/glm52_dcp_post35/results.csv`](../../benchmarks/research/glm52_dcp_post35/results.csv).
The same directory contains an overlay Dockerfile and a launcher for the
TP4/DCP4 NF3 decode comparison.

Clone both branches next to each other, then build:

```bash
docker buildx build --load \
  --build-context vllm_src=/path/to/vllm \
  --build-context sparkinfer_src=/path/to/sparkinfer \
  -f /path/to/vllm/benchmarks/research/glm52_dcp_post35/Dockerfile.overlay \
  -t local/vllm:glm52-dcp-post35-poc \
  /path/to/vllm
```

Run matched cases only after the model is fully loaded. Do not load another
model while benchmarking:

```bash
cd /path/to/vllm
MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  MODE=stock \
  benchmarks/research/glm52_dcp_post35/run_decode_case.sh

MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  MODE=materialized-bulk \
  benchmarks/research/glm52_dcp_post35/run_decode_case.sh

MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  MODE=remote-storage \
  benchmarks/research/glm52_dcp_post35/run_decode_case.sh
```

The launcher only starts the server. Use the same client for every case; the
reported numbers used `llm_decode_bench.py`, active per-user throughput, and
separate ctx0/64k/256k runs.

## Release disposition

No production change resulted from this post-#35 POC:

- `c456c470ba`'s process-group correction is already present in the more
  complete production PR #179.
- Lazy full-gather pool binding is only needed when a selected-record-only
  state is allowed before the normal pool exists.
- Remote pointer/storage consumption, bulk selected-record prefetch, and
  sharding-through-attention remain research-only and disabled by default.
- No new v20 image or production PR is justified by this branch.

The existing v20 image and merge checklist remain authoritative. This branch
exists so future work can start from tested code and known negative results
instead of repeating the same implementation.

## Viable future direction

Revisit this only if the communication geometry changes, not merely the
transport implementation. A credible successor would need to fuse selection,
record fetch, and attention consumption so that it avoids materializing full
records, avoids irregular per-record remote loads, and demonstrably transfers
less data than compact query/LSE/output exchange. Any such design must retain
the stock path as an oracle and report PCIe bytes, kernel time, KV capacity,
tail latency, and E2E throughput.
