"""Cost model.

API prices are USD per million tokens (Anthropic first-party list prices,
checked 2026-10). The local model has no per-token price; we charge it the
compute time of the machine it runs on.
"""
from __future__ import annotations

API_PRICES = {  # model: (input $/MTok, output $/MTok)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# 4 vCPU / 16 GB general-purpose cloud VM (e.g. AWS m7i.xlarge on-demand ~ $0.20/h).
CPU_HOUR_USD = 0.20


def api_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = API_PRICES.get(model, (0.0, 0.0))
    if pin == 0.0 and pout == 0.0:
        # Longest-key-first so versioned names match the right base model.
        for key, (kp_in, kp_out) in sorted(API_PRICES.items(), key=lambda kv: -len(kv[0])):
            if model.startswith(key):
                pin, pout = kp_in, kp_out
                break
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def compute_cost(seconds: float, hourly: float = CPU_HOUR_USD) -> float:
    return seconds / 3600 * hourly
