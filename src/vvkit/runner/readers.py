"""Output readers for CSV, NPZ, HDF5, and custom formats via entry points."""

import importlib.metadata
from collections.abc import Callable
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


def read_hdf5_output(
    h5_file: Path, coord_paths: dict[str, str], field_paths: dict[str, str]
) -> dict[str, Any]:
    """Read solution and coordinates from an HDF5 file."""
    import h5py

    fields = {}
    coords = {}
    with h5py.File(h5_file, "r") as f:
        for k, p in coord_paths.items():
            coords[k] = f[p][:]
        for k, p in field_paths.items():
            fields[k] = f[p][:]

        # Optional explicit cell measures in HDF5 if user stored them
        measures = f["/cell_measures"][:] if "/cell_measures" in f else None

    return {"fields": fields, "coords": coords, "cell_measures": measures}


def read_txt_output(
    txt_file: Path, coord_cols: list[str], field_cols: list[str]
) -> dict[str, Any]:
    """Read solution from plain-text columns without headers."""
    # Assuming the user provided columns in order, map names to indices for loadtxt
    arr = np.loadtxt(txt_file)
    coords = {}
    fields = {}

    # If the text file has 1D array of shape (N,), reshape to (N, 1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Note: text columns reader expects the user to pass columns mapped as integers in strings
    # e.g., coords={"x": "0"}, fields={"u": "1"}
    for k, col_idx_str in zip(coord_cols, coord_cols, strict=False):
        # We handle parsing if the user passes lists of ints or string ints
        try:
            col_idx = int(col_idx_str)
            coords[k] = arr[:, col_idx]
        except ValueError:
            pass  # Ignore invalid column mapping for text reader

    for k, col_idx_str in zip(field_cols, field_cols, strict=False):
        try:
            col_idx = int(col_idx_str)
            fields[k] = arr[:, col_idx]
        except ValueError:
            pass

    return {"fields": fields, "coords": coords, "cell_measures": None}


def get_reader(reader_type: str) -> Callable[..., dict[str, Any]]:
    """Get reader function by name, supporting built-ins and plugins."""
    builtins = {
        "npz": read_npz_output,
        "csv": read_csv_output,
        "hdf5": read_hdf5_output,
        "txt": read_txt_output,
    }
    if reader_type in builtins:
        return builtins[reader_type]

    # Check entry points for third-party readers
    eps = importlib.metadata.entry_points(group="vvkit.readers")
    for ep in eps:
        if ep.name == reader_type:
            return ep.load()  # type: ignore[no-any-return]

    raise ValueError(f"Unknown reader type: {reader_type}")
