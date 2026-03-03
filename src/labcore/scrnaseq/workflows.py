# src/labcore/scrnaseq/workflows.py

import pandas as pd
import scanpy as sc
from anndata import AnnData

from .io import read_cellbender_matrix_h5, build_gene_map_from_h5_paths
from .qc import preprocess_sample

def load_and_preprocess_from_manifest(
    manifest_path: str,
    sample_id_col: str = "SampleID",
    path_col: str = "SamplePath",
    **preprocess_kwargs
) -> AnnData:
    """
    Loads and preprocesses scRNA-Seq data from a manifest file.
    ... (docstring is unchanged) ...
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
