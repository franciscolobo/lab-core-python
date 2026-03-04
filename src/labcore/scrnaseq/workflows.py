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
    filter_outliers_nmads: float | None = None, # <-- NEW PARAMETER
    **preprocess_kwargs
) -> AnnData:
    """
        Loads and preprocesses scRNA-Seq data from a manifest file.

    This workflow performs the following steps:
    1. Reads a sample manifest TSV/CSV file.
    2. Builds a comprehensive gene ID-to-symbol map from all H5 files.
    3. Iterates through each sample, loading the CellBender data.
    4. Attaches gene symbols and sample metadata.
    5. Runs the 'preprocess_sample' function on each sample with provided arguments.
    6. Concatenates all processed samples into a single AnnData object.

    Args:
        manifest_path: Path to the sample manifest file.
        sample_id_col: Name of the column in the manifest containing unique sample IDs.
        path_col: Name of the column containing the path to each sample's H5 file.
        filter_outliers_nmads: If set to a float (e.g., 3.0), robust outlier
                               detection will be run on each sample individually
                               before concatenation, using the specified number
                               of MADs as the threshold.

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
    for _, row in df.iterrows():
        sample_meta = row.to_dict()
        print(f"Processing sample: {sample_meta.get(sample_id_col, 'N/A')}")

        adata = read_cellbender_matrix_h5(sample_meta[path_col])

        ids = pd.Index(adata.var_names.astype(str))
        # This is the first fix (already done)
        mapped_symbols = pd.Series(ids.map(gene_map), index=ids)
        adata.var["gene_symbol"] = mapped_symbols.fillna(pd.Series(ids, index=ids))

        adata_processed = preprocess_sample(adata, sample_meta=sample_meta, **preprocess_kwargs)
        adatas.append(adata_processed)
        if filter_outliers_nmads:
            print(f"  -> Running robust outlier detection (NMADs={filter_outliers_nmads})...")
            # We use the existing filter function on this single-sample object.
            # The library_key is the sample_id_col, which will only have one group.
            adata_filtered = filter_outlier_cells(
                adata=adata_processed,
                library_key=sample_id_col,
                qc_metrics=['log1p_n_genes_by_counts', 'log1p_total_counts', 'pct_counts_mt'],
                nmads=filter_outliers_nmads
            )
            adatas.append(adata_filtered)
        else:
            adatas.append(adata_processed)

    print("\nConcatenating all samples...")
    if not adatas:
        raise ValueError("No samples were processed. Aborting concatenation.")

    adata_full = sc.concat(
        adatas,
        join="outer",
        label=sample_id_col,
        keys=[x.obs[sample_id_col].iloc[0] for x in adatas],
        index_unique="-",
    )

    # Ensure final object has a clean gene symbol column using the same, correct logic
    ids = pd.Index(adata_full.var_names.astype(str))
    mapped_symbols_full = pd.Series(ids.map(gene_map), index=ids)
    adata_full.var["gene_symbol"] = mapped_symbols_full.fillna(pd.Series(ids, index=ids))
    
    print("Workflow complete. Final object shape:", adata_full.shape)
    return adata_full


def run_downstream_analysis(
    adata_hvg: AnnData,
    full_adata: AnnData,
    use_rep: str = "X_pca",
    n_pcs: int = 30,
    n_neighbors: int = 15,
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
    sc.tl.leiden(adata_hvg, resolution=0.5)

    print("Transferring analysis results back to the main AnnData object.")
    full_adata.obsm[use_rep] = adata_hvg.obsm[use_rep]
    full_adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    full_adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    full_adata.obsp["distances"] = adata_hvg.obsp["distances"]
    full_adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")

    return full_adata
