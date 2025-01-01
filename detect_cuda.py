import argparse
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Final, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# File Configuration
PYPROJECT_FILE: Final[Path] = Path("pyproject.toml")

# PyTorch Source Configuration
PYTORCH_SOURCE: Final[Dict[str, str]] = {
    "name": "pytorch",
    "base_url": "https://download.pytorch.org/whl",
    "priority": "explicit",
}

# CUDA Version Mapping
CUDA_VERSIONS: Final[Dict[float, str]] = {
    12.4: "cu124",
    12.1: "cu121",
    11.8: "cu118",
}

# Regex Patterns for Dependencies
RE_PATTERNS: Final[Dict[str, str]] = {
    # Source configuration patterns
    "source_block": r'\[\[tool\.poetry\.source\]\].*?name\s*=\s*"pytorch".*?priority\s*=\s*"explicit"',
    "cuda_url": r'(url\s*=\s*"https://download\.pytorch\.org/whl/)cu\d+"',
    # Version extraction patterns
    "torch_version": r'torch\s*=\s*(?:\{\s*version\s*=\s*"(\^?[0-9]+\.[0-9]+\.[0-9]+)"[^}]*\}|"(\^?[0-9]+\.[0-9]+\.[0-9]+)")',
    "vision_version": r'torchvision\s*=\s*(?:\{\s*version\s*=\s*"(\^?[0-9]+\.[0-9]+\.[0-9]+)"[^}]*\}|"(\^?[0-9]+\.[0-9]+\.[0-9]+)")',
    # Dependency patterns - fixed naming
    "torch": r'(\s+)torch\s*=\s*(?:\{[^}]*\}|"[^"]+")',
    "torchvision": r'(\s+)torchvision\s*=\s*(?:\{[^}]*\}|"[^"]+")',
    # Cleanup patterns
    "multi_newlines": r"\n{3,}",
}

# Template strings for configuration
TEMPLATES: Final[Dict[str, str]] = {
    "source_section": f"""
    [[tool.poetry.source]]
        name = "{PYTORCH_SOURCE['name']}"
        url = "{PYTORCH_SOURCE['base_url']}/{{cuda_version}}"
        priority = "{PYTORCH_SOURCE['priority']}"
""",
    "cuda_dep": '\\1{package} = {{ version = "{version}", source = "{source}" }}',
    "cpu_dep": '\\1{package} = "{version}"',
}


def get_package_versions(content: str) -> Tuple[str, str]:
    """Extract torch and torchvision versions from pyproject.toml content."""
    torch_match = re.search(RE_PATTERNS["torch_version"], content)
    vision_match = re.search(RE_PATTERNS["vision_version"], content)

    if not torch_match or not vision_match:
        raise ValueError(
            "Could not find torch or torchvision versions in pyproject.toml"
        )

    # Get first non-None group (either source or direct format)
    torch_version = next(v for v in torch_match.groups() if v is not None)
    vision_version = next(v for v in vision_match.groups() if v is not None)

    logging.debug(f"Found torch version: {torch_version}")
    logging.debug(f"Found vision version: {vision_version}")

    return torch_version, vision_version


def format_dependency(package: str, version: str, indent: int = 8) -> str:
    """Format dependency string with proper indentation."""
    return " " * indent + f'{package} = "{version}"'


def get_cuda_version() -> Optional[float]:
    """Get CUDA version from nvidia-smi output."""
    try:
        logging.info("Checking for CUDA...")
        output = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT, text=True
        )
        logging.debug(f"nvidia-smi output: {output}")
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if match:
            version = float(match.group(1))
            logging.info(f"Found CUDA version: {version}")
            return version
    except FileNotFoundError:
        logging.warning(
            "CUDA not available: nvidia-smi not found (NVIDIA driver not installed)"
        )
    except subprocess.SubprocessError as e:
        logging.warning(f"CUDA not available: nvidia-smi failed to run ({str(e)})")
    except Exception as e:
        logging.warning(f"CUDA not available: unexpected error ({str(e)})")
    return None


def map_cuda_to_pytorch_version(cuda_version: Optional[float]) -> Optional[str]:
    """Map CUDA version to PyTorch wheel version."""
    if cuda_version is None:
        logging.info("No CUDA version provided")
        return None

    compatible_versions = [v for v in CUDA_VERSIONS.keys() if v <= cuda_version]
    if not compatible_versions:
        logging.info(
            f"No compatible PyTorch CUDA version found for CUDA {cuda_version}"
        )
        return None

    version = CUDA_VERSIONS[max(compatible_versions)]
    logging.info(f"Mapped CUDA {cuda_version} to PyTorch wheel {version}")
    return version


def get_current_cuda_version(content: str) -> Optional[str]:
    """Extract current CUDA version from pyproject.toml content."""
    cuda_match = re.search(RE_PATTERNS["cuda_url"], content)
    if (cuda_match):
        # Extract cu124 from URL and return it
        url = cuda_match.group(0)
        version_match = re.search(r"cu\d+", url)
        return version_match.group(0) if version_match else None
    return None


def get_user_choice(current_cuda: Optional[str], proposed_cuda: Optional[str]) -> str:
    """Get user choice for CUDA configuration."""
    print("\nCurrent configuration:")
    if current_cuda:
        print(f"PyTorch with {current_cuda}")
    else:
        print("CPU version (no CUDA)")

    if proposed_cuda:
        print(f"\nProposed configuration: PyTorch with {proposed_cuda}")
    else:
        print("\nProposed configuration: CPU version (no CUDA)")

    print("\nOptions:")
    print("1. Change to proposed version")
    print("2. Switch to CPU-only version")
    print("q. Quit without changes")

    while True:
        choice = input("\nEnter your choice (1/2/q): ").strip().lower()
        if choice in ["1", "2", "q"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or q")


def update_pyproject_toml(
    dry_run: bool = False, interactive: bool = True, force_cpu: bool = False
) -> Tuple[bool, bool]:
    """Update pyproject.toml based on GPU and CUDA availability."""
    try:
        cuda_version: None | float = None if force_cpu else get_cuda_version()

        if not PYPROJECT_FILE.exists():
            logging.error(f"File not found: {PYPROJECT_FILE}")
            return False, False

        logging.info(f"Reading {PYPROJECT_FILE}")
        content = PYPROJECT_FILE.read_text()
        original_content = content

        # Extract versions from content
        torch_version, vision_version = get_package_versions(content)

        # Extract current CUDA config
        current_cuda = get_current_cuda_version(content)

        pytorch_cuda = map_cuda_to_pytorch_version(cuda_version)

        if interactive and not dry_run:
            choice = get_user_choice(current_cuda, pytorch_cuda)
            if choice == "q":
                logging.info("Operation cancelled by user")
                return True, False
            if choice == "2":
                pytorch_cuda = None
        else:
            # Non-interactive mode: use proposed version
            logging.info(
                f"Non-interactive mode: {'using CPU' if force_cpu else 'selecting best CUDA version'}"
            )

        if pytorch_cuda:
            # Check if source section exists, add if missing
            has_source = bool(re.search(RE_PATTERNS["source_block"], content, re.DOTALL))
            if not has_source:
                # Add source section after [tool.poetry]
                source_section = TEMPLATES["source_section"].format(cuda_version=pytorch_cuda)
                content = re.sub(
                    r'(\[tool\.poetry\].*?)(\[tool\.poetry\.dependencies\])',
                    rf'\1{source_section}\2',
                    content,
                    flags=re.DOTALL
                )
            else:
                # Update existing source URL
                content = re.sub(
                    RE_PATTERNS["cuda_url"],
                    rf'\1{pytorch_cuda}"',
                    content,
                )

            # Update both torch and torchvision to use CUDA source
            for package in ["torch", "torchvision"]:
                version = torch_version if package == "torch" else vision_version
                pattern = RE_PATTERNS[package]
                replacement = TEMPLATES["cuda_dep"].format(
                    package=package,
                    version=version,
                    source=PYTORCH_SOURCE["name"]
                )
                content = re.sub(pattern, replacement, content)
        else:
            # CPU version: Update both packages to not use source
            for package in ["torch", "torchvision"]:
                version = torch_version if package == "torch" else vision_version
                pattern = RE_PATTERNS[package]
                replacement = TEMPLATES["cpu_dep"].format(
                    package=package,
                    version=version
                )
                content = re.sub(pattern, replacement, content)

        if content == original_content:
            logging.info("No changes needed")
            return True, False

        if dry_run:
            logging.info("Dry run - would write these changes:")
            logging.info(content)
            return True, True

        logging.info(f"Writing updated configuration to {PYPROJECT_FILE}")
        PYPROJECT_FILE.write_text(content)
        return True, True

    except KeyError as e:
        logging.error(f"Invalid pattern key: {str(e)}")
        return False, False
    except Exception as e:
        logging.error(f"Error updating pyproject.toml: {str(e)}")
        return False, False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update PyTorch CUDA configuration in pyproject.toml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making actual changes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without user prompts, automatically select best option",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only configuration",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the script."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Starting configuration update...")
    success, changed = update_pyproject_toml(
        dry_run=args.dry_run, interactive=not args.non_interactive, force_cpu=args.cpu
    )

    if not success:
        logging.info("Configuration failed")
    elif not changed:
        logging.info("Configuration unchanged")
    else:
        logging.info("Configuration completed successfully")

    return 0 if success else 1


if __name__ == "__main__":
    main()
