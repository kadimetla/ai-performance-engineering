import torch
import pytest

from core.benchmark.verification import coerce_input_signature, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin

try:
    from hypothesis import given, strategies as st  # type: ignore
except Exception:
    st = None


class _DummyBenchmark(VerificationPayloadMixin):
    def __init__(self):
        self._verification_payload = None

    def capture_verification_payload(self) -> None:
        x = torch.ones(2)
        y = x + 1
        self._set_verification_payload(
            inputs={"x": x},
            output=y,
            batch_size=1,
            parameter_count=2,
            output_tolerance=(1e-3, 1e-3),
        )


def test_verification_payload_round_trip() -> None:
    bench = _DummyBenchmark()
    bench.capture_verification_payload()

    inputs = bench.get_verify_inputs()
    assert set(inputs.keys()) == {"x"}

    out = bench.get_verify_output()
    assert torch.allclose(out, torch.ones(2) + 1)

    sig = bench.get_input_signature()
    assert sig.shapes["x"] == (2,)

    rtol, atol = bench.get_output_tolerance()
    assert (rtol, atol) == (1e-3, 1e-3)


def test_simple_signature_validation() -> None:
    sig = simple_signature(batch_size=4, dtype="float32", rows=2, cols=3)
    coerced = coerce_input_signature(sig)
    assert coerced.batch_size == 4
    assert coerced.shapes["workload"] == (2, 3)


@pytest.mark.skipif(st is None, reason="hypothesis not installed")
@given(st.integers(min_value=1, max_value=10000))  # type: ignore[arg-type]
def test_simple_signature_handles_any_batch_size(batch_size: int) -> None:
    sig = simple_signature(batch_size=batch_size, dtype="float32", length=8)
    assert sig.batch_size == batch_size
