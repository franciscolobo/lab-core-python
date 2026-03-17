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

# In src/labcore/scrnaseq/utils.py

def rank_genes_groups_df(
    adata: AnnData, 
    group: str | None = None,
    gene_symbol_col: str = "gene_symbol"
) -> pd.DataFrame:
    """
    Converts sc.tl.rank_genes_groups output into a tidy pandas DataFrame,
    including gene symbols and expression percentages (if available).

    Args:
        adata: AnnData object with `rank_genes_groups` results.
        group: Specific group/cluster ID to retrieve. If None, concatenates all.
        gene_symbol_col: Column in `adata.var` with gene symbols.

    Returns:
        A pandas DataFrame with DGE results.
    """
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError("Please run `sc.tl.rank_genes_groups` before calling this function.")

    results = adata.uns['rank_genes_groups']
    fields = [f for f in list(results.keys()) if f != 'params']
    
    if group:
        groups = [group]
    else:
        groups = list(results[fields[0]].dtype.names)

    all_dfs = []
    for g in groups:
        group_data = {field: results[field][g] for field in fields if field != 'pts'}
        df = pd.DataFrame(group_data)
        df['group'] = g
        
        # --- NEW LOGIC TO ADD EXPRESSION PERCENTAGES ---
        if 'pts' in results:
            pts_df = pd.DataFrame(results['pts'][g])
            pts_df.columns = ['pct_1']
            df = pd.concat([df, pts_df], axis=1)

        # To get pct.2, we calculate the expression in all other cells
        if 'pts' in results:
            other_groups = [og for og in groups if og != g]
            if other_groups:
                # Average the pts across all other groups
                other_pts = pd.DataFrame({og: results['pts'][og] for og in other_groups})
                df['pct_2'] = other_pts.mean(axis=1).values
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    df['diff'] = df['pct_1'] - df['pct_2']

    if gene_symbol_col in adata.var.columns:
        id_to_symbol_map = adata.var[gene_symbol_col]
        final_df['gene_symbol'] = final_df['names'].map(id_to_symbol_map)
        
    # Define the final column order, including the new percentage columns
    col_order = [
        'group', 'gene_symbol', 'names', 'logfoldchanges', 
        'pct_1', 'pct_2', 'pvals', 'pvals_adj', 'scores'
    ]
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


def print_top_markers(
    dge_df: pd.DataFrame,
    cluster_id: str,
    n_top: int = 10,
    display_cols: list[str] | None = None  # <-- NEW PARAMETER
):
    """
    Filters, sorts, and prints the top N marker genes for a specific cluster,
    allowing for custom column selection.

    Args:
        dge_df: The DataFrame of DGE results from rank_genes_groups_df.
        cluster_id: The ID of the cluster to display.
        n_top: The number of top genes to print.
        display_cols: An optional list of column names to display. If None, a
                      default set is used.
    """
    # --- THIS IS THE NEW LOGIC ---
    # If the user doesn't specify columns, use a helpful default set.
    if display_cols is None:
        display_cols = ['gene_symbol', 'logfoldchanges', 'pvals_adj', 'scores']

    try:
        # Filter for the specific cluster
        cluster_df = dge_df[dge_df['group'] == str(cluster_id)]

        # Sort by the 'scores' column to find the best markers
        top_markers = cluster_df.sort_values(by='scores', ascending=False)

        print(f"\n--- Top {n_top} DEGs for Cluster {cluster_id} ---")
        # Use the 'display_cols' variable to select which columns to print
        print(top_markers[display_cols].head(n_top).to_string())

    except KeyError:
        print(f"Error: One or more of the requested columns do not exist in the DataFrame.")
        print(f"Available columns are: {dge_df.columns.tolist()}")

