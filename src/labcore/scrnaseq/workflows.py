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

    # --- THIS IS THE SECOND, CRITICAL FIX ---
    # Ensure final object has a clean gene symbol column using the same, correct logic
    ids = pd.Index(adata_full.var_names.astype(str))
    mapped_symbols_full = pd.Series(ids.map(gene_map), index=ids)
    adata_full.var["gene_symbol"] = mapped_symbols_full.fillna(pd.Series(ids, index=ids))
    
    print("Workflow complete. Final object shape:", adata_full.shape)
    return adata_full

def run_standard_analysis(
    adata: AnnData,
    use_rep: str = "X_pca",
    n_top_genes: int = 3000,
    n_pcs: int = 30,
    n_neighbors: int = 15
) -> AnnData:
    """
    Runs a standard Scanpy analysis workflow on a given representation.

    This workflow performs:
    1. Finds highly variable genes.
    2. Subsets to HVGs, scales, and runs PCA.
    3. Computes neighbors and UMAP using the specified representation (`use_rep`).
    4. Clusters with Leiden.
    5. Transfers the new `obsm`, `obsp`, and `obs` data back to the full AnnData object.

    Args:
        adata: The full AnnData object.
        use_rep: The representation to use for neighbor calculation. 
                 Typically 'X_pca' for no integration or 'X_pca_harmony' for
                 Harmony-integrated analysis.
        n_top_genes: Number of highly variable genes to select.
        n_pcs: Number of principal components to compute.
        n_neighbors: Number of neighbors for the UMAP graph.
    
    Returns:
        The input AnnData object, updated with analysis results.
    """
    print(f"\n--- Running standard analysis using '{use_rep}' ---")
    
    # Store raw counts if not already present
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    # Find HVGs
    print("Finding highly variable genes using raw counts...")
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="counts",  # <--- ADD THIS ARGUMENT
    )
 
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=n_pcs, svd_solver="arpack")
    
    # If the requested representation doesn't exist on the HVG object (e.g. Harmony),
    # copy it from the full object.
    if use_rep not in adata_hvg.obsm:
        if use_rep in adata.obsm:
            print(f"Copying '{use_rep}' from full AnnData object to HVG subset.")
            adata_hvg.obsm[use_rep] = adata.obsm[use_rep]
        else:
            raise KeyError(f"Representation '{use_rep}' not found in adata.obsm. Run integration first.")

    # Compute neighbors and UMAP on the specified representation
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.umap(adata_hvg, min_dist=0.3)
    sc.tl.leiden(adata_hvg, resolution=0.5)

    # Transfer results from the HVG object back to the main adata object
    print("Transferring analysis results back to the main AnnData object.")
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    if "X_pca_harmony" in adata_hvg.obsm:
        adata.obsm["X_pca_harmony"] = adata_hvg.obsm["X_pca_harmony"]
    
    adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    adata.obsp["distances"] = adata_hvg.obsp["distances"]
    adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")

    print("--- Analysis complete ---")
    return adata
