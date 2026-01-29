"""
Script to run all unit tests.

Usage:
    python -m src_v4.tests.run_tests
    python -m src_v4.tests.run_tests -v           # Verbose
    python -m src_v4.tests.run_tests -k "encoder" # Filter by name
"""

import sys
import pytest


def main():
    """Run all tests."""
    # Default arguments
    args = [
        "src_v4/tests/",
        "-v",
        "--tb=short",
    ]

    # Add any command line arguments
    args.extend(sys.argv[1:])

    # Run pytest
    exit_code = pytest.main(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
