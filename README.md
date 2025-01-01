# AdvanceML_project5

>[!WARNING]
> Please follow the guide below before installing the local environment

## Installation

This project uses poetry to manage the environment. 

Before installing make sure that the pytorch is using a compatible cuda version for your machine.
There is already a friendly script in the project to update the `pyproject.toml` file to best fit your machine hardware capabilities.

Simply run the script from the root folder of the project with your python global environment
```bash
python detect_cuda.py
```

For more information the script can accept multiple arguments, try running
```bash
python detect_cuda.py -h
```

Anyway the basics idea is that the script will ask if you want to:
1. Update the package to match your cuda version
2. Use the cpu version of `pytorch`

After the script terminate you can safely install the environment with 
```bash
poetry install
```

### If the script doesn't work

If for some reason the script doesn't work, the best and fasted method is to go to the official [pytorch installation page](https://pytorch.org/get-started/locally/), and get there the source knowing you CUDA capabilities, or, if don't have cuda or want to use the cpu just go into the `pyproject.toml` file and remove the source from both `torch` and `torchvision`

Example with custom cuda version:
```toml
[...]
    [tool.poetry.dependencies]
        python = "^3.12"
        torch = { version = "^2.5.1", source = "pytorch" }
        torchvision = { version = "^0.20.1", source = "pytorch" }
[...]
```

Example without the cuda (cpu version):
```toml
[...]
    [tool.poetry.dependencies]
        python = "^3.12"
        torch = "^2.5.1"
        torchvision = "^0.20.1"
[...]
```

>[!NOTE]
> Removing the `source="pytorch"` and leaving the version inside the parenthesis, like `torch = { version = "^2.5.1"}`, would have produced the same result, it's a matter of personal preference at this point.