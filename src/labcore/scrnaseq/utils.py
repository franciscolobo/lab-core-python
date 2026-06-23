# src/labcore/scrnaseq/utils.py

import pandas as pd
import numpy as np
import scipy.sparse as sp
from anndata import AnnData 
import scanpy as sc

def describe_adata(adata, cell_index: int = 0, n_features: int = 100,
                   n_cells: int = 10) -> dict:
    """Print a concise structural summary of an AnnData object.

    Displays the object's shape, then each obs column individually with
    a sample of cells, and the expression profile of a single cell of
    interest.

    Args:
        adata: An AnnData object to inspect.
        cell_index: Integer position of the cell to profile. Defaults to 0
            (the first cell).
        n_features: Number of gene-expression pairs to print for the
            selected cell. Defaults to 100.
        n_cells: Number of example cells to show per obs column.
            Defaults to 10.

    Returns:
        A dict mapping gene names to expression values for the selected
        cell, so the caller can use the profile programmatically if needed.

    Example:
        >>> import scanpy as sc
        >>> from samap_extension import utils
        >>> adata = sc.read_h5ad("path/to/object.h5ad")
        >>> profile = utils.describe_adata(adata, cell_index=0, n_features=50)
    """
    SEP = "=" * 60

    # --- Shape ---
    print(f"{SEP}")
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"{SEP}\n")

    # --- obs columns, one per block ---
    print("Cell metadata (obs):")
    print(f"  {adata.n_obs:,} cells × {len(adata.obs.columns)} columns\n")

    sample = adata.obs.iloc[:n_cells]
    for col in adata.obs.columns:
        print(f"  --- {col} ---")
        col_data = sample[col]
        for cell_id, val in col_data.items():
            print(f"    {cell_id}:  {val}")
        print()

    # --- var columns, one per block ---
    print("Gene metadata (var):")
    print(f"  {adata.n_vars:,} genes × {len(adata.var.columns)} columns\n")

    if adata.var.columns.empty:
        print("  (no var columns)\n")
    else:
        var_sample = adata.var.iloc[:n_cells]
        for col in adata.var.columns:
            print(f"  --- {col} ---")
            for gene_id, val in var_sample[col].items():
                print(f"    {gene_id}:  {val}")
            print()

    # --- Single-cell expression profile ---
    cell_id = adata.obs_names[cell_index]
    cell_features = adata[cell_index, :].X

    if not isinstance(cell_features, np.ndarray):
        cell_features = cell_features.toarray()

    cell_profile = dict(zip(adata.var_names, cell_features.flatten()))

    print(f"{SEP}")
    print(f"Cell ID: {cell_id}")
    print(f"First {n_features} features:")
    print(list(cell_profile.items())[:n_features])

    return cell_profile


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


# In src/labcore/scrnaseq/utils.py

def rank_genes_groups_df(
    adata: AnnData, 
    group: str | None = None,
    gene_symbol_col: str = "gene_symbol"
) -> pd.DataFrame:
    """
    Converts sc.tl.rank_genes_groups output into a tidy pandas DataFrame,
    correctly calculating expression percentages within and outside the group.

    Args:
        adata: AnnData object with `rank_genes_groups` results (run with pts=True).
        group: Specific group/cluster ID to retrieve. If None, concatenates all.
        gene_symbol_col: Column in `adata.var` with gene symbols.

    Returns:
        A pandas DataFrame with DGE results.
    """
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError("Please run `sc.tl.rank_genes_groups` before calling this function.")

    results = adata.uns['rank_genes_groups']
    group_names = list(results['names'].dtype.names)
    groupby_key = results['params']['groupby']

    # --- NEW, ROBUST, AND SIMPLER LOGIC ---
    # 1. Get a boolean DataFrame where True means a gene is expressed in a cell.
    #    We use the 'counts' layer if it exists, as it's the most reliable source for this.
    layer_to_use = 'counts' if 'counts' in adata.layers else None
    expr_df = sc.get.obs_df(
        adata,
        keys=list(adata.var_names),
        layer=layer_to_use,
    ) > 0
    
    # 2. Add the cluster/group labels to this dataframe.
    expr_df[groupby_key] = adata.obs[groupby_key].values

    # 3. Use groupby().mean() to get the fraction of expressing cells for each gene in each cluster.
    #    The result is a DataFrame of shape (n_clusters, n_genes).
    pct_df = expr_df.groupby(groupby_key, observed=True).mean().T

    all_dfs = []
    groups_to_process = [group] if group else group_names
    
    for g in groups_to_process:
        # Extract standard results
        group_data = {
            field: results[field][g]
            for field in ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']
            if field in results
        }
        df = pd.DataFrame(group_data)
        df['group'] = g
        
        # Get pct.1 from our pre-calculated table
        df['pct.1'] = pct_df.loc[df['names'].values, g].values

        # Calculate pct.2
        other_groups = [og for og in group_names if og != g]
        other_sizes = adata.obs[groupby_key].value_counts()[other_groups]
        other_pcts = pct_df.loc[df['names'].values, other_groups]
        df['pct.2'] = np.average(other_pcts, axis=1, weights=other_sizes.values)
        
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Add diff and gene symbols
    final_df['diff'] = final_df['pct.1'] - final_df['pct.2']
    if gene_symbol_col in adata.var.columns:
        id_to_symbol_map = adata.var[gene_symbol_col]
        final_df['gene_symbol'] = final_df['names'].map(id_to_symbol_map)
    
    # Reorder columns
    col_order = [
        'group', 'gene_symbol', 'names', 'logfoldchanges', 
        'pct.1', 'pct.2', 'diff', 'pvals', 'pvals_adj', 'scores'
    ]
    final_df = final_df[[c for c in col_order if c in final_df.columns]]
    
    return final_df


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

