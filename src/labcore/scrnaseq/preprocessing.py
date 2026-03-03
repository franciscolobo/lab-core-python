import scanpy as sc
from anndata import AnnData

def score_cell_cycle(
    adata: AnnData,
    s_genes: list[str],
    g2m_genes: list[str],
) -> AnnData:
    """
    Scores cells for cell cycle phases (S and G2/M).
    ... (docstring is unchanged) ...
    """
    print("Scoring cell cycle phases...")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    adata.obs['phase'] = adata.obs['phase'].astype('category')
    print("Cell cycle scoring complete.")
    return adata

def preprocess_for_pca(
    adata: AnnData,
    n_top_genes: int = 3000,
    regress_vars: list[str] | None = None,
) -> AnnData:
    """
    Prepares an AnnData object for PCA by finding HVGs, regressing, and scaling.
    ... (docstring is unchanged) ...
    """
    print("\n--- Preprocessing for PCA ---")
    
    if 'counts' not in adata.layers:
        raise ValueError("A 'counts' layer with raw counts is required for HVG selection.")
        
    print("Finding highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        layer='counts',
        n_top_genes=n_top_genes,
        flavor='seurat_v3'
    )

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    if regress_vars:
        print(f"Regressing out the following variables: {regress_vars}")
        sc.pp.regress_out(adata_hvg, regress_vars)

    print("Scaling data...")
    sc.pp.scale(adata_hvg, max_value=10)
    
    return adata_hvg
