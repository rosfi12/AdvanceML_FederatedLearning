import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_package_versions(content: str) -> Tuple[str, str]:
    """Extract torch and torchvision versions from pyproject.toml content."""
    # Remove duplicate entries first
    content = re.sub(r'(\s+torch\s*=\s*"[^"]+"\n)+', r"\1", content)
    content = re.sub(r'(\s+torchvision\s*=\s*"[^"]+"\n)+', r"\1", content)

    # First try source-based format
    torch_source_pattern = r'torch\s*=\s*\{\s*version\s*=\s*"([^"]+)"'
    vision_source_pattern = r'torchvision\s*=\s*\{\s*version\s*=\s*"([^"]+)"'

    # Fallback patterns for direct version format
    torch_direct_pattern = r'torch\s*=\s*"([^"]+)"'
    vision_direct_pattern = r'torchvision\s*=\s*"([^"]+)"'

    # Try source-based format first
    torch_match = re.search(torch_source_pattern, content)
    vision_match = re.search(vision_source_pattern, content)

    # If not found, try direct version format
    if not torch_match:
        torch_match = re.search(torch_direct_pattern, content)
    if not vision_match:
        vision_match = re.search(vision_direct_pattern, content)

    if not torch_match or not vision_match:
        raise ValueError(
            "Could not find torch or torchvision versions in pyproject.toml"
        )

    return torch_match.group(1), vision_match.group(1)


def format_dependency(package: str, version: str, indent: int = 8) -> str:
    """Format dependency string with proper indentation."""
    return " " * indent + f'{package} = "{version}"'


def get_cuda_version() -> float | None:
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
    return None


def map_cuda_to_pytorch_version(cuda_version):
    """Map CUDA version to PyTorch wheel version."""
    if cuda_version is None:
        logging.info("No CUDA version provided")
        return None

    cuda_map: dict[float, str] = {
        12.4: "cu124",
        12.1: "cu121",
        11.8: "cu118",
    }

    compatible_versions: list[float] = [v for v in cuda_map.keys() if v <= cuda_version]
    if not compatible_versions:
        logging.info(
            f"No compatible PyTorch CUDA version found for CUDA {cuda_version}"
        )
        return None

    version: str = cuda_map[max(compatible_versions)]
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
        path = Path("pyproject.toml")

        if not path.exists():
            logging.error(f"File not found: {path}")
            return False, False

        logging.info(f"Reading {path}")
        content: str = path.read_text()
        original_content: str = content

        # Clean up duplicate dependencies first
        content = re.sub(r'(\s+torch\s*=\s*"[^"]+"\n)+', r"\1", content)
        content = re.sub(r'(\s+torchvision\s*=\s*"[^"]+"\n)+', r"\1", content)

        # Extract versions from cleaned content
        torch_version, vision_version = get_package_versions(content)

        # Extract current CUDA config
        current_cuda_match = re.search(
            r'url = "https://download\.pytorch\.org/whl/(cu\d+)"', content
        )
        current_cuda = current_cuda_match.group(1) if current_cuda_match else None

        pytorch_cuda = (
            map_cuda_to_pytorch_version(cuda_version) if cuda_version else None
        )

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

        # Update dependencies based on choice
        if pytorch_cuda:
            # Update CUDA URL in source section
            if re.search(
                r'\[\[tool\.poetry\.source\]\].*name = "pytorch"', content, re.DOTALL
            ):
                content = re.sub(
                    r'url = "https://download\.pytorch\.org/whl/cu\d+"',
                    f'url = "https://download.pytorch.org/whl/{pytorch_cuda}"',
                    content,
                )
            else:
                source_section = f"""
    [[tool.poetry.source]]
        name = "pytorch"
        url = "https://download.pytorch.org/whl/{pytorch_cuda}"
        priority = "explicit"
"""
                content = content.replace(
                    "[tool.poetry]", f"[tool.poetry]{source_section}"
                )

            # Update dependencies to use source
            dep_pattern = r'(\s+)torch\s*=\s*"[^"]+"'
            content = re.sub(
                dep_pattern,
                rf'\1torch = {{ version = "{torch_version}", source = "pytorch" }}',
                content,
            )
            dep_pattern = r'(\s+)torchvision\s*=\s*"[^"]+"'
            content = re.sub(
                dep_pattern,
                rf'\1torchvision = {{ version = "{vision_version}", source = "pytorch" }}',
                content,
            )
        else:
            # CPU version: Update dependencies only, keep source section
            dep_pattern = r"(\s+)torch\s*=\s*\{[^}]*\}"
            content = re.sub(dep_pattern, rf'\1torch = "{torch_version}"', content)
            dep_pattern = r"(\s+)torchvision\s*=\s*\{[^}]*\}"
            content = re.sub(
                dep_pattern, rf'\1torchvision = "{vision_version}"', content
            )

        content = re.sub(r"\n{3,}", "\n\n", content)

        if content == original_content:
            logging.info("No changes needed")
            return True, False

        if dry_run:
            logging.info("Dry run - would write these changes:")
            logging.info(content)
            return True, True

        logging.info(f"Writing updated configuration to {path}")
        path.write_text(content)
        return True, True

    except Exception as e:
        logging.error(f"Error updating pyproject.toml: {str(e)}")
        return False, False


def main():
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
