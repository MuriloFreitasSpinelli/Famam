import subprocess
import sys


def test_cli_runs_help():
    result = subprocess.run(
        [sys.executable, "-m", "src", "--help"],
        capture_output=True,
        text=True,
    )

    # CLI should not crash
    assert result.returncode == 0