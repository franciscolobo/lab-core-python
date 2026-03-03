import scanpy as sc
from anndata import AnnData

def score_cell_cycle(
    adata: AnnData,
    s_genes: list[str],
    g2m_genes: list[str],
    gene_symbol_col: str = "gene_symbol",
) -> AnnData:
    """
    Scores cells for cell cycle phases (S and G2/M) using gene symbols.

    This function temporarily sets the AnnData index to the gene symbol column
    to run scanpy's `score_genes_cell_cycle`, then restores the original index.

    Args:
        adata: The AnnData object (should be log-normalized).
        s_genes: A list of gene symbols for the S phase.
        g2m_genes: A list of gene symbols for the G2/M phase.
        gene_symbol_col: The column in `adata.var` that contains the gene symbols.

    Returns:
        The input AnnData object, updated with 'S_score', 'G2M_score',
        and 'phase' columns in `.obs`.
    """
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(
            f"Column '{gene_symbol_col}' not found in adata.var. "
            "Cannot score cell cycle without gene symbols."
        )

    print("Scoring cell cycle phases...")

    # Store the original index
    original_var_names = adata.var_names.copy()
    
    # Temporarily set the index to gene symbols for scoring
    # Make sure the symbols are unique before setting as index
    if not adata.var[gene_symbol_col].is_unique:
        print("Warning: Gene symbols are not unique. Making them unique for scoring.")
        adata.var_names = adata.var[gene_symbol_col].astype(str) + "-" + adata.var.index.astype(str)
    else:
        adata.var_names = adata.var[gene_symbol_col].astype(str)
        
    # Run the scoring function
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,
        g2m_genes=g2m_genes
    )
    
    # --- IMPORTANT: Restore the original index ---
    adata.var_names = original_var_names
    
    # It's good practice to make the 'phase' column categorical
    adata.obs['phase'] = adata.obs['phase'].astype('category')
    
    print("Cell cycle scoring complete. Added 'S_score', 'G2M_score', and 'phase' to .obs.")
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
