# src/labcore/scrnaseq/utils.py

import pandas as pd
import numpy as np
import scipy.sparse as sp
from anndata import AnnData 

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

def set_rank_genes_symbols(
    adata: AnnData,
    gene_symbol_col: str = "gene_symbol"
) -> None:
    """
    Modifies the `adata.uns['rank_genes_groups']` result in place to use
    gene symbols instead of Ensembl IDs in the 'names' field.

    This is necessary for plotting functions like `sc.pl.rank_genes_groups`
    to display human-readable gene names.

    Args:
        adata: The AnnData object after running `sc.tl.rank_genes_groups`.
        gene_symbol_col: The column in `adata.var` that contains the gene symbols.
    """
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError("Please run `sc.tl.rank_genes_groups` before calling this function.")
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(f"Column '{gene_symbol_col}' not found in adata.var.")

    print("Replacing Ensembl IDs with gene symbols in `rank_genes_groups` results...")

    # Create a mapping from Ensembl ID (the index) to the gene symbol
    id_to_symbol_map = pd.Series(
        adata.var[gene_symbol_col].values,
        index=adata.var_names
    ).to_dict()

    # The 'names' field is a structured numpy array. We need to iterate through it.
    # We create a new array of the same shape and type to hold the new names.
    old_names = adata.uns['rank_genes_groups']['names']
    new_names = np.empty_like(old_names)

    # Iterate through the structured array by field (cluster names)
    for cluster in old_names.dtype.names:
        # For each cluster, map the Ensembl IDs to gene symbols
        new_names[cluster] = [
            id_to_symbol_map.get(ensembl_id, ensembl_id) # Fallback to ID if not found
            for ensembl_id in old_names[cluster]
        ]

    # Replace the old 'names' structured array with our new one
    adata.uns['rank_genes_groups']['names'] = new_names

    print("Update complete.")

def rank_genes_groups_df(
    adata: AnnData,
    group: str | None = None,
    gene_symbol_col: str = "gene_symbol"
) -> pd.DataFrame:
    """
    Converts the output of sc.tl.rank_genes_groups into a tidy pandas DataFrame,
    automatically including gene symbols if available.

    Args:
        adata: The AnnData object with `rank_genes_groups` results.
        group: The specific group (e.g., a cluster ID) to retrieve results for.
               If None, results for all groups are concatenated.
        gene_symbol_col: The column in `adata.var` that contains the gene symbols.

    Returns:
        A pandas DataFrame with the DGE results, including a gene symbol column.
    """
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError("Please run `sc.tl.rank_genes_groups` before calling this function.")

    results = adata.uns['rank_genes_groups']
    fields = [f for f in list(results.keys()) if f != 'params']

    if group:
        if group not in results[fields[0]].dtype.names:
            raise ValueError(f"Group '{group}' not found in rank_genes_groups results.")
        groups = [group]
    else:
        groups = list(results[fields[0]].dtype.names)

    all_dfs = []
    for g in groups:
        group_data = {field: results[field][g] for field in fields}
        df = pd.DataFrame(group_data)
        df['group'] = g
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    # --- NEW LOGIC TO AUTOMATICALLY ADD GENE SYMBOLS ---
    if gene_symbol_col in adata.var.columns:
        print(f"Adding gene symbols from column: '{gene_symbol_col}'")
        # Create a mapping from Ensembl ID (the 'names' column) to symbol
        id_to_symbol_map = adata.var[gene_symbol_col]
        final_df['gene_symbol'] = final_df['names'].map(id_to_symbol_map)

        # Reorder columns for clarity
        col_order = ['group', 'gene_symbol', 'names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores']
        # Filter for columns that actually exist in the dataframe
        final_df = final_df[[c for c in col_order if c in final_df.columns]]
    else:
        print(f"Warning: Gene symbol column '{gene_symbol_col}' not found. Returning table with IDs only.")
        # Reorder columns without symbols
        col_order = ['group', 'names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores']
        final_df = final_df[[c for c in col_order if c in final_df.columns]]

    return final_df

def get_top_markers(
    adata: AnnData,
    group_id: str,
    n_top: int = 10,
    sort_by: str = 'scores',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Extracts a DataFrame of the top N marker genes for a specific group.

    Args:
        adata: AnnData object after running rank_genes_groups.
        group_id: The ID of the group/cluster to get markers for.
        n_top: The number of top genes to return.
        sort_by: The column to sort by (e.g., 'scores', 'pvals_adj').
        ascending: The sort order. False for descending (best markers first).

    Returns:
        A pandas DataFrame with the top N marker genes for the specified group.
    """
    # Use our existing function to get the full table
    full_df = rank_genes_groups_df(adata)

    # Filter and sort
    group_df = full_df[full_df['group'] == str(group_id)]
    top_markers = group_df.sort_values(by=sort_by, ascending=ascending)

    return top_markers.head(n_top)

