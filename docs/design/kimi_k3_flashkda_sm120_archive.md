# Kimi K3 FlashKDA SM120 archive

Status: tested research branch; intentionally excluded from the production HH
Kimi K3 runtime.

## Scope

Commit `721e515069f8dd5b3162c1760cd78ac0f6ec3531` patches the pinned FlashKDA
source (`a3e42bbbece3bb38f7c426b880315294a336e82f`) for SM120. It replaces the
non-tensor raw bulk G2S loads used for the swizzled recurrence workspace with
raw-byte TMA tensor maps. The output layout and recurrence stay unchanged.

The branch also contains:

- the external-source patch and idempotent CMake application helper;
- `VLLM_FLASHKDA_KIMI_K3_ONLY`, which builds only the Kimi K3 specialization;
- `benchmarks/kernels/benchmark_flashkda_memory.py` for first-launch memory,
  persistent-memory, correctness-fingerprint, and kernel-time measurements.

## Measured result on RTX PRO 6000 Blackwell

| Measurement | Unpatched | Patched raw-TMA |
|---|---:|---:|
| First-launch device-memory delta | 4,078,960,640 B (3.80 GiB) | 153,092,096 B (0.143 GiB) |
| Persistent delta after `empty_cache()` | 4,053,794,816 B (3.78 GiB) | 127,926,272 B (0.119 GiB) |
| 65,536-token kernel time | 4.5349 ms | 4.3208 ms |

The patch removed the SM120 multi-GiB first-launch allocation and improved the
isolated kernel by about 4.7% at 65,536 tokens.

## End-to-end result

Apples-to-apples full-MXFP4 TP16/DCP8 prefill with a reduced KV allocation and
`max_num_batched_tokens=4096`:

| Backend | 8,192-token prefill median |
|---|---:|
| Triton KDA | 3,957.394 tok/s |
| Patched FlashKDA | 3,942.716 tok/s |

FlashKDA was 0.37% slower in this end-to-end test. A physical-1M run did start
and serve after the memory fix, but the 1M + MBT2048 configuration remained too
tight for a later 224 MiB transient allocation (85.25 MiB was free).

## Decision

The production HH branch uses `KDA_PREFILL_BACKEND=triton` because patched
FlashKDA did not improve end-to-end prefill and would add a 321-line patch to an
external project. This choice affects KDA prefill only. Kimi K3 decode uses the
separate fused `_kimi_k3_kda_ops` path.

Keep this branch as the exact code archive for future FlashKDA/SM120 work. Do
not merge it into HH without a new end-to-end result that beats Triton and a
memory test at the intended KV/cache scheduler settings.
