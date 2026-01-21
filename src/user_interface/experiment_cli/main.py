"""Main entry point for the Experiment CLI."""

import sys

from .prompts import clear_screen, print_header, print_menu, get_choice
from .dataset_cli import run_dataset_creation
from .training_cli import run_tune_model
from .generation_cli import run_generate_music


def show_main_menu():
    """Display the main menu and return user choice."""
    print_header("Famam Experiment CLI")
    options = [
        "Create Dataset",
        "Tune & Train Model",
        "Generate Music",
        "Exit",
    ]
    print_menu(options)
    return get_choice(len(options))


def run_experiment_cli():
    """Run the main experiment CLI loop."""
    while True:
        try:
            clear_screen()
            choice = show_main_menu()

            if choice == 1:
                run_dataset_creation()
            elif choice == 2:
                run_tune_model()
            elif choice == 3:
                run_generate_music()
            elif choice == 4:
                print("\nGoodbye!")
                sys.exit(0)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Returning to main menu...")
            continue
        except EOFError:
            print("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    run_experiment_cli()
