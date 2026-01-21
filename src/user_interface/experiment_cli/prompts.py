"""Helper functions for CLI prompts."""

from pathlib import Path
from typing import Optional, List, TypeVar, Callable

T = TypeVar("T")


def clear_screen():
    """Clear terminal screen."""
    print("\033[H\033[J", end="")


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_menu(options: List[str], title: str = "Select an option:"):
    """Print a numbered menu."""
    print(title)
    print("-" * 40)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print()


def get_choice(max_choice: int, prompt: str = "Enter choice: ") -> int:
    """Get a valid menu choice from the user."""
    while True:
        try:
            choice = input(prompt).strip()
            if not choice:
                continue
            num = int(choice)
            if 1 <= num <= max_choice:
                return num
            print(f"Please enter a number between 1 and {max_choice}")
        except ValueError:
            print("Please enter a valid number")


def get_input(
    prompt: str,
    default: Optional[str] = None,
    validator: Optional[Callable[[str], bool]] = None,
    error_msg: str = "Invalid input",
) -> str:
    """Get string input with optional default and validation."""
    default_str = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{default_str}: ").strip()
        if not value and default is not None:
            return default
        if not value:
            print("This field is required")
            continue
        if validator and not validator(value):
            print(error_msg)
            continue
        return value


def get_int(
    prompt: str,
    default: Optional[int] = None,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    """Get integer input with optional bounds."""
    default_str = f" [{default}]" if default is not None else ""
    bounds = []
    if min_val is not None:
        bounds.append(f"min: {min_val}")
    if max_val is not None:
        bounds.append(f"max: {max_val}")
    bounds_str = f" ({', '.join(bounds)})" if bounds else ""

    while True:
        value = input(f"{prompt}{default_str}{bounds_str}: ").strip()
        if not value and default is not None:
            return default
        try:
            num = int(value)
            if min_val is not None and num < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and num > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return num
        except ValueError:
            print("Please enter a valid integer")


def get_float(
    prompt: str,
    default: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Get float input with optional bounds."""
    default_str = f" [{default}]" if default is not None else ""
    bounds = []
    if min_val is not None:
        bounds.append(f"min: {min_val}")
    if max_val is not None:
        bounds.append(f"max: {max_val}")
    bounds_str = f" ({', '.join(bounds)})" if bounds else ""

    while True:
        value = input(f"{prompt}{default_str}{bounds_str}: ").strip()
        if not value and default is not None:
            return default
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and num > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return num
        except ValueError:
            print("Please enter a valid number")


def get_bool(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "true", "1"):
            return True
        if value in ("n", "no", "false", "0"):
            return False
        print("Please enter y or n")


def get_optional_int(prompt: str, default: Optional[int] = None) -> Optional[int]:
    """Get optional integer input (empty for None)."""
    default_str = f" [{default}]" if default is not None else " [None]"
    while True:
        value = input(f"{prompt}{default_str} (empty for None): ").strip()
        if not value:
            return default
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            print("Please enter a valid integer or leave empty")


def get_optional_float(prompt: str, default: Optional[float] = None) -> Optional[float]:
    """Get optional float input (empty for None)."""
    default_str = f" [{default}]" if default is not None else " [None]"
    while True:
        value = input(f"{prompt}{default_str} (empty for None): ").strip()
        if not value:
            return default
        if value.lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            print("Please enter a valid number or leave empty")


def get_path(
    prompt: str,
    default: Optional[str] = None,
    must_exist: bool = False,
    is_dir: bool = False,
) -> str:
    """Get path input with optional existence check."""
    default_str = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{default_str}: ").strip()
        if not value and default:
            value = default
        if not value:
            print("This field is required")
            continue

        path = Path(value)
        if must_exist:
            if is_dir and not path.is_dir():
                print(f"Directory does not exist: {value}")
                continue
            elif not is_dir and not path.exists():
                print(f"Path does not exist: {value}")
                continue
        return value


def get_list_input(prompt: str, default: Optional[List[str]] = None) -> List[str]:
    """Get comma-separated list input."""
    default_str = f" [{', '.join(default)}]" if default else ""
    while True:
        value = input(f"{prompt}{default_str} (comma-separated): ").strip()
        if not value and default is not None:
            return default
        if not value:
            print("At least one item is required")
            continue
        return [item.strip() for item in value.split(",") if item.strip()]


def get_optional_list(prompt: str, default: Optional[List[str]] = None) -> Optional[List[str]]:
    """Get optional comma-separated list input."""
    default_str = f" [{', '.join(default)}]" if default else " [None]"
    value = input(f"{prompt}{default_str} (comma-separated, empty for None): ").strip()
    if not value:
        return default
    if value.lower() == "none":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def list_config_files(config_dir: Path) -> List[Path]:
    """List all JSON config files in a directory."""
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.json"))


def select_config_file(config_dir: Path, config_type: str) -> Optional[Path]:
    """Let user select a config file from a directory."""
    configs = list_config_files(config_dir)
    if not configs:
        print(f"No {config_type} config files found in {config_dir}")
        return None

    print(f"\nAvailable {config_type} configs:")
    print("-" * 40)
    for i, config in enumerate(configs, 1):
        print(f"  [{i}] {config.name}")
    print()

    choice = get_choice(len(configs), "Select config: ")
    return configs[choice - 1]


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask for confirmation."""
    return get_bool(prompt, default)
