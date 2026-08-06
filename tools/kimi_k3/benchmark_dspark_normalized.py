#!/usr/bin/env python3
"""Measure DSpark output throughput independently from target-cycle rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
POSITION_COUNTER = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _labels(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        result[key] = value.strip('"')
    return result


def _metrics(url: str, model: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in _read_url(f"{url}/metrics").decode().splitlines():
        if not line or line.startswith("#") or "{" not in line:
            continue
        metric, rest = line.split("{", 1)
        labels_text, value_text = rest.rsplit("}", 1)
        labels = _labels(labels_text)
        if labels.get("model_name") != model:
            continue
        if metric in COUNTERS:
            values[metric] = float(value_text)
        elif metric == POSITION_COUNTER:
            position = labels.get("position")
            if position is not None:
                values[f"{metric}:{position}"] = float(value_text)
    missing = [name for name in COUNTERS if name not in values]
    if missing:
        raise RuntimeError(f"missing DSpark metrics: {missing}")
    return values


def _stream_request(
    *,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, int], list[float], str, float, float]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 1,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    event_times: list[float] = []
    output_parts: list[str] = []
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                event_times.append(time.perf_counter())
                output_parts.append(choices[0].get("text") or "")
    ended = time.perf_counter()
    return usage, event_times, "".join(output_parts), started, ended


def _run(
    *,
    name: str,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    temperature: float,
    output_dir: Path,
) -> dict[str, object]:
    before = _metrics(url, model)
    usage, event_times, output, started, ended = _stream_request(
        url=url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    after = _metrics(url, model)
    if len(event_times) < 2:
        raise RuntimeError(f"{name}: insufficient streamed choice events")

    deltas = {key: after.get(key, 0.0) - value for key, value in before.items()}
    drafts = deltas[COUNTERS[0]]
    draft_tokens = deltas[COUNTERS[1]]
    accepted = deltas[COUNTERS[2]]
    if drafts <= 0 or draft_tokens <= 0:
        raise RuntimeError(f"{name}: invalid speculative deltas {deltas}")

    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = event_times[-1] - event_times[0]
    emitted_per_cycle = 1.0 + accepted / drafts
    output_path = output_dir / f"{name}.txt"
    output_path.write_text(output, encoding="utf-8")
    position_deltas = {
        key.rsplit(":", 1)[1]: value
        for key, value in deltas.items()
        if key.startswith(f"{POSITION_COUNTER}:")
    }
    result: dict[str, object] = {
        "completion_tokens": completion_tokens,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "draft_acceptance_rate": accepted / draft_tokens,
        "mean_accepted_draft_tokens_per_cycle": accepted / drafts,
        "mean_emitted_tokens_per_target_cycle": emitted_per_cycle,
        "output_file": str(output_path),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt))),
        "temperature": temperature,
        "run": name,
        "speculative_counter_delta": {
            "accepted_per_position": position_deltas,
            COUNTERS[2]: accepted,
            COUNTERS[1]: draft_tokens,
            COUNTERS[0]: drafts,
        },
        "target_cycles_per_second_counter": drafts / decode_seconds,
        "target_cycles_per_second_normalized": (
            (completion_tokens - 1) / emitted_per_cycle / decode_seconds
        ),
        "timed_events": len(event_times),
        "ttft_seconds": event_times[0] - started,
        "wall_seconds": ended - started,
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompt = json.loads(args.token_file.read_text(encoding="utf-8"))
    prompt = prompt[: args.prompt_tokens]
    if len(prompt) != args.prompt_tokens or not all(
        isinstance(token, int) for token in prompt
    ):
        raise ValueError("token file lacks the requested integer-token prefix")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.warmups):
        _run(
            name=f"warmup-{index + 1}",
            url=args.url,
            model=args.model,
            prompt=prompt,
            max_tokens=min(args.max_tokens, 128),
            temperature=args.temperature,
            output_dir=args.output_dir,
        )

    results = [
        _run(
            name=f"run-{index + 1}",
            url=args.url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_dir=args.output_dir,
        )
        for index in range(args.runs)
    ]
    summary = {
        "decode_tokens_per_second_median": statistics.median(
            float(result["decode_tokens_per_second"]) for result in results
        ),
        "draft_acceptance_rate_median": statistics.median(
            float(result["draft_acceptance_rate"]) for result in results
        ),
        "max_tokens": args.max_tokens,
        "mean_emitted_tokens_per_target_cycle_median": statistics.median(
            float(result["mean_emitted_tokens_per_target_cycle"]) for result in results
        ),
        "model": args.model,
        "output_sha256s": [result["output_sha256"] for result in results],
        "prompt_tokens": args.prompt_tokens,
        "runs": args.runs,
        "temperature": args.temperature,
        "target_cycles_per_second_median": statistics.median(
            float(result["target_cycles_per_second_normalized"]) for result in results
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
