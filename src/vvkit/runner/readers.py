"""Output readers for CSV, NPZ, HDF5, and custom formats."""

from pathlib import Path
from typing import Any

import numpy as np


def read_npz_output(npz_file: Path) -> dict[str, Any]:
    """Read solution and coordinates from a NumPy .npz file."""
    data = np.load(npz_file)
    fields = {}
    coords = {}
    for key in data.files:
        if key in ["x", "y", "z"]:
            coords[key] = data[key]
        elif key == "cell_measures":
            pass
        else:
            fields[key] = data[key]
    measures = data["cell_measures"] if "cell_measures" in data.files else None
    return {"fields": fields, "coords": coords, "cell_measures": measures}


def read_csv_output(
    csv_file: Path, coord_cols: list[str], field_cols: list[str]
) -> dict[str, Any]:
    """Read solution and coordinates from a CSV file."""
    arr = np.genfromtxt(csv_file, delimiter=",", names=True)
    names = arr.dtype.names or ()
    coords = {c: arr[c] for c in coord_cols if c in names}
    fields = {f: arr[f] for f in field_cols if f in names}
    return {"fields": fields, "coords": coords, "cell_measures": None}
