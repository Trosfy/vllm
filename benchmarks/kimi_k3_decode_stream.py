#!/usr/bin/env python3
"""Measure true decode throughput from streamed completion-token timestamps."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Kimi-K3-EXL3-3p09-MXFP4-44p0")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    prompt = json.loads(args.token_file.read_text(encoding="utf-8"))
    prompt = prompt[: args.prompt_tokens]
    if len(prompt) != args.prompt_tokens or not all(
        isinstance(token, int) for token in prompt
    ):
        raise ValueError("token file lacks the requested integer token prefix")
    payload = json.dumps(
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 1,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    token_times: list[float] = []
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
            if not choices:
                continue
            # A length stop annotates the final sampled-token event; it is not
            # a separate terminal event, so the final event remains timed.
            token_times.append(time.perf_counter())
            output_parts.append(choices[0].get("text") or "")
    ended = time.perf_counter()

    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens < 2 or len(token_times) < 2:
        raise RuntimeError(
            f"insufficient stream: usage={usage}, timed_events={len(token_times)}"
        )
    decode_seconds = token_times[-1] - token_times[0]
    result = {
        "formula": "(completion_tokens - 1) / (last_token_time - first_token_time)",
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt))),
        "completion_tokens": completion_tokens,
        "timed_events": len(token_times),
        "ttft_seconds": token_times[0] - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "wall_seconds": ended - started,
    }
    args.output.write_text("".join(output_parts), encoding="utf-8")
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
