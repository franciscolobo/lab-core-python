# src/labcore/scrnaseq/utils.py

import pandas as pd
import numpy as np
import scipy.sparse as sp

def attach_gene_symbols(adata_obj, gene_map, col="gene_symbol"):
    """Attach a gene_symbol column to adata_obj.var using var_names (gene IDs) as keys."""
    ids = pd.Index(adata_obj.var_names.astype(str))
    adata_obj.var[col] = ids.map(gene_map).astype("object")
    adata_obj.var[col] = adata_obj.var[col].fillna(ids.astype(str))
    return adata_obj

def map_markers_to_ids(adata, marker_df, symbol_col="gene_symbol", group_col="cell_type"):
    """Map gene symbols in marker_df to gene IDs in adata.var_names."""
    ref = pd.DataFrame({"gene_id": adata.var_names.astype(str), "gene_symbol": adata.var["gene_symbol"].astype(str)})
    merged = marker_df.merge(ref, left_on=symbol_col, right_on="gene_symbol", how="left")
    marker_ids = {ct: sub.dropna(subset=["gene_id"])["gene_id"].tolist() for ct, sub in merged.groupby(group_col)}
    missing = {ct: sub.loc[sub["gene_id"].isna(), symbol_col].tolist() for ct, sub in merged.groupby(group_col)}
    return marker_ids, missing, merged

def markers_df_to_dict(markers_df, symbol_col="gene_symbol", celltype_col="cell_type"):
    """Convert a marker dataframe to a dict of {cell_type: [symbols]}."""
    df = markers_df[[symbol_col, celltype_col]].dropna().drop_duplicates()
    df[symbol_col] = df[symbol_col].astype(str).str.strip()
    df[celltype_col] = df[celltype_col].astype(str).str.strip()
    return df.groupby(celltype_col)[symbol_col].apply(list).to_dict()

def adata_status(adata, counts_layer="counts"):
    """Print a concise status report for an AnnData object."""
    def _x_summary(X):
        vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
        if vals.size == 0: return {"dtype": str(vals.dtype), "nvals": 0}
        return {"dtype": str(vals.dtype), "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}

    print("=== AnnData Status Report ===")
    print(f"n_obs: {adata.n_obs:,}, n_vars: {adata.n_vars:,}")
    print(f"obs_names unique: {adata.obs_names.is_unique}, var_names unique: {adata.var_names.is_unique}")
    print(f"var columns: {list(adata.var.columns)}")
    print(f"X summary: {_x_summary(adata.X)}")
    print(f"layers: {list(adata.layers.keys())}")
    if counts_layer in adata.layers:
        print(f"  -> '{counts_layer}' summary: {_x_summary(adata.layers[counts_layer])}")
    print(f"obsm keys: {list(adata.obsm.keys())}")
    print(f"obsp keys: {list(adata.obsp.keys())}")
    if adata.raw: print(f"raw: present, with {adata.raw.n_vars:,} vars")

