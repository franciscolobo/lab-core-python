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

def run_scvi_integration(
    adata: AnnData,
    batch_key: str,
    continuous_covariate_keys: list[str] | None = None,
    categorical_covariate_keys: list[str] | None = None,
    counts_layer: str = "counts",
    n_top_genes: int = 3000,
    hvg_flavor: str = "seurat_v3",
    n_layers: int = 1,
    n_latent: int = 10,
    gene_likelihood: str = "nb",
    n_neighbors: int = 15,
    leiden_resolution: float = 1.0,
    train_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> AnnData:
    """
    Runs scVI integration and transfers the results back onto the full,
    all-gene AnnData object.

    HVGs are selected for training the scVI model, but unlike calling
    `sc.pp.highly_variable_genes(..., subset=True)` directly, the input
    `adata` is never subsetted in place — all genes are retained in the
    returned object, with only `.var["highly_variable"]` marking which
    genes scVI was trained on. `.obsm["X_scVI"]`, neighbors, UMAP, and
    Leiden clusters are computed on the HVG-restricted latent space and
    attached back onto the full object, mirroring the transfer pattern
    used by `run_downstream_analysis`.

    Args:
        adata: AnnData object with raw counts available in
            `adata.layers[counts_layer]`. Kept fully intact (all genes)
            in the returned object.
        batch_key: Column in `adata.obs` identifying the batch to
            integrate over (passed to `scvi.model.SCVI.setup_anndata`).
        continuous_covariate_keys: Optional list of columns in
            `adata.obs` holding continuous covariates to regress out
            inside the model (e.g. `["S_score", "G2M_score"]` for cell
            cycle). Passed through to `setup_anndata`. Defaults to
            `None` (no continuous covariates).
        categorical_covariate_keys: Optional list of columns in
            `adata.obs` holding categorical covariates (e.g. sample
            batch beyond `batch_key`, or genotype). Passed through to
            `setup_anndata`. Defaults to `None`.
        counts_layer: Layer in `adata.layers` containing raw counts.
            Defaults to `"counts"`.
        n_top_genes: Number of highly variable genes to select for
            training. Defaults to 3000.
        hvg_flavor: Flavor passed to `sc.pp.highly_variable_genes`.
            Defaults to `"seurat_v3"` (expects raw counts).
        n_layers: Number of hidden layers in the scVI encoder/decoder.
            Defaults to 1.
        n_latent: Dimensionality of the scVI latent space. Defaults
            to 10.
        gene_likelihood: Likelihood model used by scVI (e.g. `"nb"`,
            `"zinb"`, `"poisson"`). Defaults to `"nb"`.
        n_neighbors: Number of neighbors for the post-integration
            neighbor graph. Defaults to 15.
        leiden_resolution: Resolution passed to `sc.tl.leiden`.
            Defaults to 1.0.
        train_kwargs: Optional dict of extra keyword arguments forwarded
            to `model.train()` (e.g. `{"max_epochs": 400}`).
        model_kwargs: Optional dict of extra keyword arguments forwarded
            to `scvi.model.SCVI(...)` beyond `n_layers`, `n_latent`, and
            `gene_likelihood`.

    Returns:
        The input `adata`, retaining all genes, updated with:
            - `.var["highly_variable"]`: HVG flag used for training
            - `.obsm["X_scVI"]`: scVI latent representation
            - `.obsp["connectivities"]`, `.obsp["distances"]`: neighbor graph
            - `.obsm["X_umap"]`: UMAP coordinates
            - `.obs["leiden"]`: cluster assignments

    Raises:
        ValueError: If `counts_layer` is not found in `adata.layers`.
    """
    import scvi

    if counts_layer not in adata.layers:
        raise ValueError(
            f"Layer '{counts_layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    train_kwargs = train_kwargs or {}
    model_kwargs = model_kwargs or {}

    # --- Select HVGs without subsetting the full object ---
    print(f"Selecting top {n_top_genes} HVGs (flavor='{hvg_flavor}')...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
        layer=counts_layer,
        batch_key=batch_key,
        subset=False,
    )

    # --- Train scVI on a separate HVG-only copy ---
    adata_scvi = adata[:, adata.var["highly_variable"]].copy()

    print("Registering data with scVI...")
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer=counts_layer,
        batch_key=batch_key,
        continuous_covariate_keys=continuous_covariate_keys,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    print(
        f"Training scVI (n_layers={n_layers}, n_latent={n_latent}, "
        f"gene_likelihood='{gene_likelihood}')..."
    )
    model = scvi.model.SCVI(
        adata_scvi,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
        **model_kwargs,
    )
    model.train(**train_kwargs)

    # --- Transfer latent representation back onto the full object ---
    adata.obsm["X_scVI"] = model.get_latent_representation()

    print(f"Computing neighbors (n_neighbors={n_neighbors}) on X_scVI...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=n_neighbors)
    print("Computing UMAP...")
    sc.tl.umap(adata)
    print(f"Computing Leiden clusters (resolution={leiden_resolution})...")
    sc.tl.leiden(adata, resolution=leiden_resolution)

    return adata
