"""
Entry point for running src.client as a module.

Usage:
    python -m src.client             # Show menu to choose CLI
    python -m src.client experiment  # Run experiment CLI
    python -m src.client generate    # Run generation CLI
"""

import sys


def main():
    """Main entry point with CLI selection."""
    args = sys.argv[1:]

    if args:
        cmd = args[0].lower()
        if cmd in ('experiment', 'exp', 'e', '1'):
            from .experiment_cli import main as experiment_main
            return experiment_main()
        elif cmd in ('generate', 'gen', 'g', '2'):
            from .generation_cli import main as generation_main
            return generation_main()
        elif cmd in ('help', '-h', '--help'):
            print(__doc__)
            return 0

    # No args - show selection menu
    print("\n" + "=" * 50)
    print("  FAMAM Music Generation")
    print("=" * 50)
    print("\n  Select CLI mode:\n")
    print("    [1] Experiment CLI - Dataset & Training")
    print("    [2] Generation CLI - Music Generation")
    print("    [0] Exit")
    print("\n" + "=" * 50)

    while True:
        try:
            choice = input("\n  Select option: ").strip()
            if choice == '0':
                return 0
            elif choice == '1':
                from .experiment_cli import main as experiment_main
                return experiment_main()
            elif choice == '2':
                from .generation_cli import main as generation_main
                return generation_main()
            else:
                print("  Please enter 0, 1, or 2")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return 0


if __name__ == "__main__":
    sys.exit(main())
