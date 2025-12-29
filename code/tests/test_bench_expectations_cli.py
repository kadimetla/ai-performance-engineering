import json
import subprocess
import sys


def test_bench_expectations_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "expectations", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "expectations" in result.stdout.lower()


def test_bench_expectations_json_output():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "bench",
            "expectations",
            "--hardware",
            "b200",
            "--min-speedup",
            "1.0",
            "--goal",
            "speed",
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    summary = payload.get("summary", {})
    assert summary.get("hardware") == "b200"
    assert isinstance(payload.get("entries"), list)
    assert isinstance(payload.get("missing"), list)
