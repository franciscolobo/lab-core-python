# src/labcore/scrnaseq/workflows.py

import pandas as pd
import scanpy as sc
from anndata import AnnData

from .io import read_cellbender_matrix_h5, build_gene_map_from_h5_paths
from .qc import preprocess_sample, filter_outlier_cells

def load_and_preprocess_from_manifest(
    manifest_path: str,
    sample_id_col: str = "SampleID",
    path_col: str = "SamplePath",
    filter_outliers_nmads: float | None = None,
    **preprocess_kwargs
) -> AnnData:
    """
    Loads and preprocesses scRNA-Seq data from a manifest file, then concatenates
    the results into a single AnnData object.

    Args:
        manifest_path: Path to the sample manifest file (TSV/CSV).
        sample_id_col: Name of the column in the manifest containing biological sample IDs.
        path_col: Name of the column containing the path to each sample's H5 file.
        filter_outliers_nmads: If set to a float (e.g., 3.0), robust outlier
                               detection will be run on each sample individually
                               before concatenation.
        **preprocess_kwargs: Keyword arguments to be passed directly to the
                             `labcore.scrnaseq.preprocess_sample` function
                             (e.g., `mito_genes`, `max_pct_mito`).

    Returns:
        A single, concatenated AnnData object containing all preprocessed samples.
    """
    print(f"Reading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path, sep="\t")

    required_cols = {sample_id_col, path_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Manifest is missing required columns: {required_cols - set(df.columns)}")

    print("Building union gene map from all H5 files...")
    gene_map = build_gene_map_from_h5_paths(df[path_col].tolist())
    print(f"Gene map built with {len(gene_map)} unique entries.")

    adatas = []
    keys_for_concat = []

    for i, row in df.iterrows():
        sample_meta = row.to_dict()
        sample_id = sample_meta.get(sample_id_col, 'N/A')
        print(f"Processing sample: {sample_id} (row {i})")

        adata_processed = read_cellbender_matrix_h5(sample_meta[path_col])

        # Add all metadata from the manifest row into .obs
        for col, value in sample_meta.items():
            adata_processed.obs[col] = value

        # Attach gene symbols from the comprehensive map
        ids = pd.Index(adata_processed.var_names.astype(str))
        mapped_symbols = pd.Series(ids.map(gene_map), index=ids)
        adata_processed.var["gene_symbol"] = mapped_symbols.fillna(pd.Series(ids, index=ids))

        # Run initial per-sample preprocessing and filtering
        # Pass empty dict for sample_meta as we already added it
        adata_processed = preprocess_sample(adata_processed, sample_meta={}, **preprocess_kwargs)

        if filter_outliers_nmads:
            print(f"  -> Running robust outlier detection (NMADs={filter_outliers_nmads})...")
            adata_processed = filter_outlier_cells(
                adata=adata_processed,
                library_key=sample_id_col,
                qc_metrics=['log1p_n_genes_by_counts', 'log1p_total_counts', 'pct_counts_mt'],
                nmads=filter_outliers_nmads
            )

        if adata_processed.n_obs > 0:
            adatas.append(adata_processed)
            keys_for_concat.append(i) # Use the unique DataFrame index `i` as the key
        else:
            print(f"  -> Skipping sample {sample_id} as it has no cells after filtering.")

    print("\nConcatenating all samples...")
    if not adatas:
        raise ValueError("No samples were processed successfully. Aborting concatenation.")

    adata_full = sc.concat(
        adatas,
        join="outer",
        label="batch_index", # A new, clear name for the technical batch
        keys=keys_for_concat,  # Use the simple, guaranteed-unique integer keys
        index_unique="-",
    )

    # Re-apply the final gene symbol mapping to the concatenated object
    ids_full = pd.Index(adata_full.var_names.astype(str))
    mapped_symbols_full = pd.Series(ids_full.map(gene_map), index=ids_full)
    adata_full.var["gene_symbol"] = mapped_symbols_full.fillna(pd.Series(ids_full, index=ids_full))

    print("Workflow complete. Final object shape:", adata_full.shape)
    return adata_full

def run_downstream_analysis(
    adata_hvg: AnnData,
    full_adata: AnnData,
    use_rep: str = "X_pca",
    n_pcs: int = 30,
    n_neighbors: int = 15,
    res: float = 1.0,
) -> AnnData:
    """
    Runs downstream analysis (neighbors, UMAP, Leiden) and transfers results.

    Args:
        adata_hvg: The AnnData object subsetted to HVGs, with PCA/Harmony run.
        full_adata: The original, full AnnData object to transfer results to.
        use_rep: The representation to use for neighbor calculation (e.g., 'X_pca').
        n_pcs: Number of PCs to use for neighbor calculation.
        n_neighbors: Number of neighbors for the UMAP graph.

    Returns:
        The full AnnData object, updated with analysis results.
    """
    print(f"\n--- Running downstream analysis using '{use_rep}' ---")
    
    print(f"Computing neighbors with {n_neighbors} neighbors and {n_pcs} PCs...")
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.umap(adata_hvg, min_dist=0.3)
    sc.tl.leiden(adata_hvg, resolution=res)

    print("Transferring analysis results back to the main AnnData object.")
    full_adata.obsm[use_rep] = adata_hvg.obsm[use_rep]
    full_adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    full_adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    full_adata.obsp["distances"] = adata_hvg.obsp["distances"]
    full_adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")

    return full_adata
