import json
import sys
from pathlib import Path
from typing import Optional


def convert_notebook_to_script(
    notebook_path: str, output_path: Optional[str | Path] = None
) -> None:
    """Convert Jupyter notebook to Python script."""

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Prepare output
    if output_path is None:
        output_path = Path(notebook_path).with_suffix(".py")

    script_content = []

    # Process cells
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            # Convert markdown to multi-line comment
            markdown = "".join(cell["source"])
            comment = f'"""\n{markdown}\n"""\n\n'
            script_content.append(comment)

        elif cell["cell_type"] == "code":
            # Add code directly
            code = "".join(cell["source"])
            if code.strip():  # Only add non-empty code cells
                script_content.append(f"{code}\n\n")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(script_content))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python notebook_to_script.py <notebook_path> [output_path]")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_notebook_to_script(notebook_path, output_path)
    print(f"Converted {notebook_path} to Python script")
