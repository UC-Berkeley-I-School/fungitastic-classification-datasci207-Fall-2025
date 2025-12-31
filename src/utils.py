import yaml
from pathlib import Path


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    """
    Save a Python object (dict, list, etc.) to a YAML file.

    Parameters
    ----------
    data : Any
        Python object to serialize (must be YAML-serializable)
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False
        )