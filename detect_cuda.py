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
    "torch_version": r'torch\s*=\s*\{\s*version\s*=\s*"(\^?[0-9]+\.[0-9]+\.[0-9]+)"',
    "vision_version": r'torchvision\s*=\s*\{\s*version\s*=\s*"(\^?[0-9]+\.[0-9]+\.[0-9]+)"',
    # Dependency patterns - fixed naming
    "torch": r"(\s+)torch\s*=\s*\{[^}]*\}",
    "torchvision": r"(\s+)torchvision\s*=\s*\{[^}]*\}",
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

    return torch_match.group(1), vision_match.group(1)


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
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logging.info(f"CUDA detection failed: {str(e)}")
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


def get_user_choice(current_cuda: Optional[str], proposed_cuda: Optional[str]) -> str:
    """Get user choice for CUDA configuration."""
    print("\nCurrent configuration:")
    print(f"{'CPU version' if current_cuda is None else f'CUDA {current_cuda}'}")

    if proposed_cuda:
        print(f"\nProposed configuration: CUDA {proposed_cuda}")

    print("\nOptions:")
    print("1. Change to proposed version")
    print("2. Keep current version")
    print("3. Switch to CPU-only version")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def confirm_cpu_switch(torch_version: str, vision_version: str) -> bool:
    """Confirm switch to CPU-only version."""
    print("\nProposed CPU-only configuration:")
    print(f'torch = "{torch_version}"')
    print(f'torchvision = "{vision_version}"')

    while True:
        choice = input("\nProceed with CPU-only configuration? (y/n): ").strip().lower()
        if choice in ["y", "n"]:
            return choice == "y"
        print("Invalid choice. Please enter y or n.")


def update_pyproject_toml(dry_run=False, interactive=True) -> Tuple[bool, bool]:
    """Update pyproject.toml based on GPU and CUDA availability."""
    try:
        cuda_version = get_cuda_version()

        if not PYPROJECT_FILE.exists():
            logging.error(f"File not found: {PYPROJECT_FILE}")
            return False, False

        logging.info(f"Reading {PYPROJECT_FILE}")
        content = PYPROJECT_FILE.read_text()
        original_content = content

        # Extract versions from content
        torch_version, vision_version = get_package_versions(content)

        # Extract current CUDA config
        current_cuda_match = re.search(RE_PATTERNS["cuda_url"], content)
        current_cuda = (
            current_cuda_match.group(1).split("/")[-1] if current_cuda_match else None
        )

        pytorch_cuda = map_cuda_to_pytorch_version(cuda_version)

        if interactive and not dry_run:
            choice = get_user_choice(current_cuda, pytorch_cuda)

            if choice == "2":  # Keep current
                logging.info("Keeping current configuration")
                return True, False

            if choice == "3":  # Switch to CPU
                if not confirm_cpu_switch(torch_version, vision_version):
                    logging.info("CPU switch cancelled")
                    return False, False
                pytorch_cuda = None

        if pytorch_cuda:
            # Update CUDA URL in source section
            if re.search(RE_PATTERNS["source_block"], content, re.DOTALL):
                content = re.sub(
                    RE_PATTERNS["cuda_url"],
                    rf'\1{pytorch_cuda}"',
                    content,
                )

            # Update dependencies to use source
            for package in ["torch", "torchvision"]:
                version = torch_version if package == "torch" else vision_version
                pattern = RE_PATTERNS[package]  # Using new pattern keys
                replacement = TEMPLATES["cuda_dep"].format(
                    package=package, version=version, source=PYTORCH_SOURCE["name"]
                )
                content = re.sub(pattern, replacement, content)
        else:
            # CPU version: Update dependencies only, keep source section
            for package in ["torch", "torchvision"]:
                version = torch_version if package == "torch" else vision_version
                pattern = RE_PATTERNS[package]  # Using new pattern keys
                replacement = TEMPLATES["cpu_dep"].format(
                    package=package, version=version
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


def main() -> int:
    """Entry point for the script."""
    logging.info("Starting configuration update...")
    success, changed = update_pyproject_toml(dry_run=False, interactive=True)

    if not success:
        logging.info("Configuration failed")
    elif not changed:
        logging.info("Configuration unchanged")
    else:
        logging.info("Configuration completed successfully")

    return 0 if success else 1


if __name__ == "__main__":
    main()
