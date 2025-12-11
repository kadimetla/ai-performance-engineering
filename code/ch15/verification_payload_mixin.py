"""Shared verification helper for ch15 multi-GPU/disaggregated benchmarks.

This module now re-exports the core verification mixin so chapter examples stay
aligned with the canonical implementation in core/benchmark/verification_mixin.py.
"""

from core.benchmark.verification_mixin import VerificationPayload, VerificationPayloadMixin

__all__ = ["VerificationPayload", "VerificationPayloadMixin"]
