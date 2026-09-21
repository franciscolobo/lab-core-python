# src/labcore/scrnaseq/io.py

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Callable

# --- Helper Functions (Internal) ---

def _as_str_array(x) -> np.ndarray:
    """Decode HDF5 datasets that may be stored as bytes (|S dtype)."""
    arr = np.array(x)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a) for a in arr])
    return arr.astype(str)

def _coerce_var_columns_for_h5ad(adata_obj, cols_bool=("mt", "ribo", "hb")):
    """Ensure QC flag columns are pure boolean to prevent H5 write errors."""
    # Fix in .var
    for c in cols_bool:
        if c in adata_obj.var.columns:
            adata_obj.var[c] = adata_obj.var[c].fillna(False).astype(bool)
    # Fix in .raw.var
    if adata_obj.raw is not None:
        for c in cols_bool:
            if c in adata_obj.raw.var.columns:
                adata_obj.raw.var[c] = adata_obj.raw.var[c].fillna(False).astype(bool)

# --- Public Functions ---

def read_cellbender_matrix_h5(path: str | Path, group_path: str = "matrix") -> ad.AnnData:
    """Read CellBender output H5 with a 10x-like CSC matrix under /matrix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        g = f[group_path]
        X_csc = sp.csc_matrix(
            (np.array(g["data"]), np.array(g["indices"]), np.array(g["indptr"])),
            shape=tuple(np.array(g["shape"]).tolist())
        )
        X = X_csc.T.tocsr()
        barcodes = _as_str_array(g["barcodes"])
        feat = g["features"]
        gene_names = _as_str_array(feat["name"])
        gene_ids = _as_str_array(feat["id"])
        var = pd.DataFrame(index=pd.Index(gene_ids.astype(str), name="gene_id"))
        var["gene_symbol"] = gene_names
        if "feature_type" in feat:
            var["feature_type"] = _as_str_array(feat["feature_type"])
        if "genome" in feat:
            var["genome"] = _as_str_array(feat["genome"])
        obs = pd.DataFrame(index=barcodes)

    if X.shape[0] != obs.shape[0]:
        raise ValueError(f"Cell dimension mismatch: X has {X.shape[0]} cells but barcodes has {obs.shape[0]}")
    if X.shape[1] != var.shape[0]:
        raise ValueError(f"Feature dimension mismatch: X has {X.shape[1]} vars but features has {var.shape[0]}")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names.name = "barcode"
    adata.var_names.name = "gene_id"
    adata.var_names_make_unique()
    return adata

def build_gene_map_from_h5_paths(h5_paths) -> pd.Series:
    """Build a union mapping from gene_id -> gene_symbol across many CellBender H5 files."""
    rows = []
    for p in map(Path, h5_paths):
        with h5py.File(p, "r") as f:
            feat = f["matrix/features"]
            gene_ids = _as_str_array(feat["id"]).astype(str)
            gene_symbols = _as_str_array(feat["name"]).astype(str)
        rows.append(pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_symbols}))

    all_map = pd.concat(rows, ignore_index=True).drop_duplicates()
    conflicts = all_map.groupby("gene_id")["gene_symbol"].nunique()
    bad = conflicts[conflicts > 1]
    if len(bad) > 0:
        examples = all_map[all_map["gene_id"].isin(bad.index)].sort_values("gene_id").head(10)
        raise ValueError(f"Found {len(bad)} gene_ids mapping to multiple symbols. Examples:\n{examples}")

    gene_map = all_map.set_index("gene_id")["gene_symbol"]
    return gene_map

def load_or_compute_adata(
    path: str | Path,
    compute_fn: Callable[[], "ad.AnnData"],
    overwrite: bool = False,
) -> "ad.AnnData":
    """Loads a cached AnnData object from disk, or computes and saves it.

    Wraps the common "if cached file exists, load it; otherwise run the
    pipeline and save the result" pattern used across preprocessing
    notebooks, so the branching logic doesn't need to be repeated at
    each checkpoint (raw loading, normalization, integration, etc.).

    Args:
        path: Path to the cached `.h5ad` file.
        compute_fn: Zero-argument callable that runs the pipeline and
            returns the resulting AnnData object when the cache is
            missing or `overwrite=True`. Typically a lambda or small
            wrapper closing over the actual pipeline call, e.g.
            ``lambda: scrnaseq.load_and_preprocess_from_manifest(...)``.
        overwrite: If `True`, ignore any existing cached file and
            recompute. Defaults to `False`.

    Returns:
        The loaded or newly computed AnnData object. When computed
        fresh, it is also written to `path` before being returned.

    Example:
        >>> adata = load_or_compute_adata(
        ...     PROCESSED_ADATA_PATH,
        ...     compute_fn=lambda: scrnaseq.load_and_preprocess_from_manifest(
        ...         manifest_path=MANIFEST_PATH,
        ...         min_genes=200,
        ...         max_pct_mito=10.0,
        ...     ),
        ...     overwrite=OVERWRITE,
        ... )
    """
    path = Path(path)

    if path.exists() and not overwrite:
        print(f"Loading cached data from: {path}")
        return ad.read_h5ad(path)

    print(f"Cached file not found (or overwrite=True). Running pipeline for: {path}")
    adata = compute_fn()

    print(f"Saving result to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)

    return adata
