# src/labcore/scrnaseq/qc.py

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# In src/labcore/scrnaseq/qc.py

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

def preprocess_sample(
    adata: AnnData, sample_meta: dict, min_genes: int = 200, min_cells_per_gene: int = 3,
    max_pct_mito: float = 20.0, max_total_counts: float = 5e4,
    mito_genes: list = None, hb_genes: list = None, ribo_genes: list = None
) -> AnnData:
    """
    Attach sample metadata, compute QC metrics, and filter. (Streamlined version)
    """
    # Attach metadata to every cell
    for k, v in sample_meta.items():
        adata.obs[k] = v

    # --- Calculate QC Flags ---
    # Ensure a gene_symbol column exists, falling back to var_names if not
    if "gene_symbol" not in adata.var.columns:
        adata.var["gene_symbol"] = adata.var_names

    sym_upper = adata.var["gene_symbol"].astype(str).str.upper()

    def _flag_genes(key, gene_list, default_prefix):
        # This logic ensures the column is always a pure boolean array
        if gene_list is not None:
            flags = sym_upper.isin([g.upper() for g in gene_list])
        else:
            flags = sym_upper.str.startswith(default_prefix)
        adata.var[key] = flags.fillna(False).to_numpy(dtype=bool)

    _flag_genes("mt", mito_genes, "MT-")
    _flag_genes("ribo", ribo_genes, ("RPL", "RPS"))
    _flag_genes("hb", hb_genes, ("HBB-", "HBA-"))

    # --- Calculate QC Metrics ---
    # This is the most important step for filtering
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )

    # --- Filter Cells and Genes ---
    # This is the main filtering logic based on the calculated metrics
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    
    # Perform subsetting based on obs metrics. Using .copy() is important.
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito, :].copy()
    adata = adata[adata.obs["total_counts"] < max_total_counts, :].copy()
    
    # The expensive re-calculation of var flags after filtering is REMOVED.
    # It's not necessary for the output of this function.

    return adata

def is_outlier(adata: AnnData, metric: str, nmads: float) -> pd.Series:
    """Check for outliers in a specific metric column of adata.obs."""
    metric_values = adata.obs[metric]
    median_val = metric_values.median()
    mad_val = (metric_values - median_val).abs().median()
    if mad_val == 0:
        return pd.Series(False, index=adata.obs.index)
    nmad_score = (metric_values - median_val).abs() / (mad_val * 1.4826)
    return nmad_score > nmads

def filter_outlier_cells(adata: AnnData, library_key: str, qc_metrics: list, nmads: float = 3.0) -> AnnData:
    """Filters outlier cells on a per-library basis using NMAD."""
    print(f"Original AnnData object shape: {adata.shape}")
    outlier_flags = pd.Series(False, index=adata.obs_names)

    for library, group_idx in adata.obs.groupby(library_key).groups.items():
        adata_library = adata[group_idx, :]
        lib_outliers = pd.Series(False, index=adata_library.obs_names)
        for metric in qc_metrics:
            lib_outliers |= is_outlier(adata_library, metric, nmads)
        outlier_flags.loc[group_idx] = lib_outliers

    n_outliers = outlier_flags.sum()
    if n_outliers > 0:
        adata_filtered = adata[~outlier_flags, :].copy()
        print(f"\nRemoved {n_outliers} total outlier cells.")
        print(f"Filtered AnnData object shape: {adata_filtered.shape}")
        return adata_filtered
    else:
        print("\nNo outlier cells were found to remove.")
        return adata.copy()

